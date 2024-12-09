import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.utils.validation import check_is_fitted
from typing import Tuple, Union, List
import json
import numpy as np
from challenge.feature_transformation import create_target, is_high_season, get_period_day
from challenge.data_validation import validate_model_features, validate_column_names, validate_categorical_values
import os

import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_VERSION = 'v1'
ARTIFACTS_PATH = f'challenge/artifacts/{MODEL_VERSION}'


class DelayModel:
    def __init__(self):
        try:
            with open(os.path.join(ARTIFACTS_PATH, 'config.json'), 'r') as file:
                self.config = json.load(file)
            logging.info("Configuration loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Configuration file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON configuration file: {e}")
            raise

        self._model_params = self.config.get("model_params")
        self._model_features = self.config.get("model_features")
        self._dataset_columns = self.config.get("dataset_columns")
        self._test_size = self.config.get("test_size")
        self._class_weights = {int(k): v for k, v in self.config.get("class_weights").items()}
        self._train_test_split_seed = self.config.get("train_test_split_seed", 42)
        self._training_shuffle_seed = self.config.get("training_shuffle_seed", 111)
        self._target_required_columns = self.config.get("target_required_columns")
        self._threshold_in_minutes = self.config.get("threshold_in_minutes")
        self._validation_score = self.config.get("validation_score", None)
        self._training_data_path = self.config.get("training_data_path")
        self._categorical_features_values = self.config.get("categorical_features_values")

        self._model = LogisticRegression(**self._model_params)
        self._model = LogisticRegression(class_weight=self._class_weights)
        self._model_is_fitted = False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def training_data_path(self):
        return self._training_data_path

    @training_data_path.setter
    def training_data_path(self, data_path):
        self._training_data_path = data_path

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.Series], 
                                                                                 pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        try:
            #data = shuffle(data, random_state=self._training_shuffle_seed)
            #data['PERIOD_DAY'] = data['Fecha-I'].apply(get_period_day)
            #data['HIGH_SEASON'] = data['Fecha-I'].apply(is_high_season)

            features = pd.concat([
                pd.get_dummies(data['OPERA'], prefix='OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
                pd.get_dummies(data['MES'], prefix='MES'),
                #pd.get_dummies(data['SIGLADES'], prefix = 'SIGLADES'),
                #pd.get_dummies(data['DIANOM'], prefix = 'DIANOM'),
                #pd.get_dummies(data['PERIOD_DAY'], prefix = 'PERIOD_DAY'),
                #data['HIGH_SEASON']
            ], axis=1)
        
            missing_cols = set(self._model_features) - set(features.columns)

            for col in missing_cols:
                features[col] = 0

            features = features[self._model_features]

            if target_column:
                target = create_target(data, self._threshold_in_minutes, self._target_required_columns)
                return features, pd.DataFrame(target, columns=[target_column])
            else:
                return features
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                features, target,
                test_size=self._test_size,
                random_state=self._train_test_split_seed
            )
            self.model.fit(x_train, y_train)
            self._model_is_fitted = True
            logging.info("Model trained successfully.")

            if self._validation_score:
                self.validate_model_accuracy(x_test, y_test, self._validation_score)
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        validate_model_features(features, self._model_features)

        if self._model_is_fitted:
            try:
                predictions = self.model.predict(features[self._model_features]).tolist()
                logging.info(f"Prediction completed. Total predictions: {len(predictions)}")
                return predictions
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                raise
        else:
            logging.info("Model not fitted. Running training pipeline...")
            self.run_training_pipeline()
            return self.predict(features)

    def validate_model_accuracy(self, x_test: pd.DataFrame, y_test: pd.Series, expected_score: float) -> None:
        """
        Validate the model's AUC score using test data.

        Args:
            x_test (pd.DataFrame): Test features.
            y_test (pd.Series): True target values for the test set (binary labels).
            expected_score (float): Expected AUC score for the model.

        Raises:
            ValueError: If the model's AUC score doesn't match the expected score up to all decimal points.
        """
        try:
            test_probs = self.model.predict_proba(x_test[self._model_features])[:, 1]
            actual_score = roc_auc_score(y_test, test_probs)

            if not actual_score == expected_score:
                raise ValueError(
                    f"Trained model AUC score ({actual_score}) doesn't match the expected score ({expected_score})"
                )
            logging.info(f"Model validation successful. AUC Score: {actual_score}")
        except Exception as e:
            logging.error(f"Error during model validation: {e}")
            raise

    def run_training_pipeline(self) -> None:
        """
        Runs the whole training pipeline and fits a model.

        """
        file_directory = os.path.dirname(os.path.abspath(__file__))
        data = pd.read_csv(os.path.join(file_directory, self._training_data_path))
        features, target = self.preprocess(data, target_column='delay')
        self.fit(features, target)

    def run_data_validations(self, features) -> None:
        """
        Run data validations to ensure the model can make predictions.

        """
        validate_column_names(features, self._dataset_columns)
        validate_categorical_values(features, self._categorical_features_values)
