import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from typing import Tuple, Union, List
import json
import numpy as np
from feature_transformation import create_target, is_high_season, get_period_day

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_VERSION = 'v1'
CONFIG_PATH = f'challenge/config/{MODEL_VERSION}/config.json'


class DelayModel:
    def __init__(self):
        try:
            with open(CONFIG_PATH, 'r') as file:
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
        self._class_weights = {int(k): v for k, v in self.config.get("class_weights").items()}
        self._test_size = self.config.get("test_size")
        self._train_test_split_seed = self.config.get("train_test_split_seed", 42)
        self._training_shuffle_seed = self.config.get("training_shuffle_seed", 111)
        self._target_required_columns = self.config.get("_target_required_columns")
        self._threshold_in_minutes = self.config.get("threshold_in_minutes")
        self._validation_score = self.config.get("validation_score")
        self._training_data_path = self.config.get("training_data_path")

        self._model = LogisticRegression(**self._model_params)

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

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
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

            if target_column:
                target = create_target(data, self._threshold_in_minutes, self._target_required_columns)
                return features, target
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
                features[self._model_features], target,
                test_size=self._test_size,
                random_state=self._train_test_split_seed
            )
            self.model.set_params(class_weight=self._class_weights)
            self.model.fit(x_train, y_train)
            logging.info("Model trained successfully.")

            # Validate if a validation score is provided
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
        try:
            predictions = self.model.predict(features[self._model_features]).tolist()
            logging.info(f"Prediction completed. Total predictions: {len(predictions)}")
            return predictions
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

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


if __name__ == '__main__':
    delay_model = DelayModel()
    df = pd.read_csv('./data/data.csv')#delay_model.training_data_path)
    features, target = delay_model.preprocess(df, target_column='si')
    delay_model.fit(features, target)