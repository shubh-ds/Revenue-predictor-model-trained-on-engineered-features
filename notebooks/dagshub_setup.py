import mlflow
import dagshub

# Set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/shubhamyadav2442/Revenue-predictor-model-trained-on-engineered-features.mlflow')

# Initialize dagshub
dagshub.init(repo_owner='shubhamyadav2442', repo_name='Revenue-predictor-model-trained-on-engineered-features', mlflow=True)

# Log dummy parameters
with mlflow.start_run():
  mlflow.log_param('dummy parameter name', 'value')
  mlflow.log_metric('dummy metric name', 1)