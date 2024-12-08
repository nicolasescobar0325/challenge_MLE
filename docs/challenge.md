Phase 1: Code review for file "exploration.ipynb"

Version of matplotlib is not allowing Jupyter to display barplots because of the class arguments. x= and y= added to every barplot to fix  this problem.

Xgboost library does not make part of any requirements file. Adding version 1.5.0 to requirements.txt

training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111). The shuffle output is independent from the columns in the dataset, so it is unnecessary to filter the columns here (removing).

Also, the dataset for training doesn't seem to be including the created features 'period_day' and 'is_high_season' and also the Data Scientist didn't provide an explanation in why to discard those features so it might be consired as a bug. 

period_day is categorical, so it should be processed with OneHotEnconding
is_high_season is binary, so it doesn't need to be processed
min_diff is calculated with the actual flight datetime so it shouldn't be included in the model

features = pd.concat([
    pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
    pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
    pd.get_dummies(data['MES'], prefix = 'MES')], 
    axis = 1
)
target = data['delay']

features definition is not using 'SIGLADES', 'DIANOM', 'period_day' or 'is_high_season'. For this use-case we will discard SIGLADES because of high dimentionality and also we are not certain if there are destine locations not included in the provided dataset (to include it or not will require a discussion with the Data Scientist).
After including 'period_day' and 'is_high_season', the top10 of features change.

period_day custom feature does not cover the case of 00:00 -TO-REVIEW-

SUGGESTED IMPROVEMENTS: 
The data scientist uses XGBOOST model to extract the 10 most important features. While this method can help to reduce the dimentionality of the dataset it also makes an assumption that the most important features for XGBoost are the same than the LinearModel, which may not hold true since each algorithm relies on completely different modeling approaches. Also, the method used to evaluate feature_importances in xgboost has a big downfall because it does not account for feature interactions/dependencies, a more reliable method to use here is SHAP.

Model selection: 
Since we are not supposed to improve the modeling process, we will select the model based in efficiency. LinearRegression is a better option since it has a lower response latency for prediction, it also is a less complex model and a more interpretable one. XGBoost is a better option if there is future development that might require model complexity but since this is not mentioned in the case it won't be taken into account. 

Phase 2: Create model.py



