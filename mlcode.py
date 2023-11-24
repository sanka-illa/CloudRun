# Import necessary libraries
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint

# Initialize a BigQuery client
client = bigquery.Client()

# Set dataframes
PROJECT_ID = 'playpen-1edb0b'
DATASET_ID = 'Vertex'
TABLE_ID = 'EPCValid'
TABLE_ID1 = 'EPCInvalid'

sql = f"""
SELECT 
  *
FROM
  '{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}'
"""

sql1 = f"""
SELECT 
  *
FROM
  '{PROJECT_ID}.{DATASET_ID}.{TABLE_ID1}'
"""

# Load EPCValid dataset from BigQuery
epc_valid = client.query(sql).result().to_dataframe()

# Load EPCInvalid dataset from BigQuery
epc_invalid = client.query(sql1).result().to_dataframe()

# Split dataset into training and validation sets
X = epc_valid.drop(columns=['co2_emissions_current'])
y = epc_valid['co2_emissions_current']
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess the data
categorical_cols = ['property_type','built_form','main_fuel_rename','mainheat_env_eff','hot_water_env_eff','floor_env_eff','windows_env_eff','walls_env_eff','roof_env_eff','mainheatc_env_eff','lighting_env_eff']
numeric_cols = ['total_floor_area', 'construction_year']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numeric_transformer, numeric_cols)
    ])

X_train_transformed = preprocessor.fit_transform(X_train)
X_validation_transformed = preprocessor.transform(X_validation)

# Hyperparameter tuning
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

rf = RandomForestRegressor()
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train_transformed, y_train)

best_params = random_search.best_params_

# Train model with best parameters
best_rf = RandomForestRegressor(**best_params)
best_rf.fit(X_train_transformed, y_train)

# Evaluate model
predictions = best_rf.predict(X_validation_transformed)
mse = mean_squared_error(y_validation, predictions)
r2 = r2_score(y_validation, predictions)

# Create a DataFrame with the scores
scores_df = pd.DataFrame({
    'MSE': [mse],
    'R2_Score': [r2]
})

# Export MSE and R² scores to BigQuery
destination_table = 'Vertex.EPCEvaluation'
job_config = bigquery.LoadJobConfig()
job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
job_config.autodetect = True

# Upload the DataFrame
job1 = client.load_table_from_dataframe(scores_df, destination_table, job_config=job_config)
job1.result()  # Wait for the job to complete

# Predict on epc_invalid
epc_invalid_transformed = preprocessor.transform(epc_invalid.drop(columns=['co2_emissions_current']))
epc_invalid['co2_emissions_predicted'] = best_rf.predict(epc_invalid_transformed)

# Export predictions to BigQuery
destination_table_predictions = "Vertex.EPCInvalidFixed"
job = client.load_table_from_dataframe(epc_invalid, destination_table_predictions)
job.result()
