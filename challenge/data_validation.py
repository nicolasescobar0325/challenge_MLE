def validate_target_required_columns(features, target_required_columns):
    features_columns = list(features.columns)
    for target_required_column in target_required_columns:
        if target_required_column not in features_columns:
            raise ValueError(f'Target required column {target_required_column} not found in input data')

def validate_column_names(features, dataset_columns) -> None:
    features_columns = list(features.columns)
    for column in dataset_columns:
        if column not in features_columns:
            raise ValueError(f"Column {column} not found in input data.")

def validate_categorical_values(features, categorical_features_values) -> None:
    for feature in features:
        feature_values = [str(col) for col in features[feature].unique()]
        accepted_values = categorical_features_values.get(feature)
        if accepted_values:
            for value in feature_values:
                if value not in accepted_values:
                    raise ValueError(f"Value {value} not in accepted values for feature {feature}.")
                
def validate_model_features(features, model_features) -> None:
    features_columns = list(features.columns)
    for column in model_features:
        if column not in features_columns:
            raise ValueError(f"Column {column} not found in input data.")