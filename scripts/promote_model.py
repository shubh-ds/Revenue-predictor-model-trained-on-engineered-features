# Import libraries
import os
import mlflow
import logging

# Logging configuration
logger = logging.getLogger('model_promotion')
logger.setLevel('DEBUG')

# Set console logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Set format for logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handlers to loggers
logger.addHandler(console_handler)
print()
logger.debug('------------------------------------------- MODEL PROMOTION STARTED -------------------------------------------------------')

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "shubhamyadav2442"
    repo_name = "Revenue-predictor-model-trained-on-engineered-features"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    # Set model name
    model_name = "Revenue_Predictor_LinearRegressor"

    # Get staging model version
    staging_model = client.get_model_version_by_alias(model_name, "staging")
    staging_version = staging_model.version

    try:
        # Get current production model's version if it exists
        current_prod_model = client.get_model_version_by_alias(model_name, "production")
        current_prod_version = current_prod_model.version

        # Update alias to previous production
        client.set_registered_model_alias(model_name, "previous_production", current_prod_version)
    except Exception:
        current_prod_version = None

    # Promote current staging model to production
    client.set_registered_model_alias(model_name, "production", staging_version)

    # Remove staging alias after promotion
    client.delete_registered_model_alias(model_name, "staging")

    print(
        f"***************** Model version {staging_version} promoted to production *************************"
        + (f", previous production was version {current_prod_version}" if current_prod_version else "")
    )
    

if __name__ == "__main__":
    promote_model()