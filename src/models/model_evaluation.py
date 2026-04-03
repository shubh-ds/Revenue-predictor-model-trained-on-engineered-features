# Import libraries
import numpy as np
import pandas as pd
import os
import yaml
import logging
import pickle
import mlflow
import dagshub
import json

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score


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

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Set console logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Set file logger
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

# Set format for logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to loggers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
print()
logger.debug('------------------------------------------- MODEL EVALUATION STARTED -------------------------------------------------------')


def load_model(path: str) -> LinearRegression:
    '''Loads pre-trained Linear Regression model from a defined path'''
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', path)
        return model

    except Exception as e:
        logger.error('Unexpected error occurred while loading the pre-trained linear regression model: %s', e)
        raise

def load_data(path: str) -> pd.DataFrame:
    '''Loads data from a defined path'''

    try:
        df = pd.read_csv(path)
        logger.debug('Data loaded from %s', path)
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def evaluate_model(model: LinearRegression, train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
    '''Evaluates the performance of LR model using train and test dataset'''

    try:
        # Features trained on
        features = train_data.iloc[:, :-1].columns

        # Predict on train and test
        train_pred = model.predict(train_data.iloc[:, :-1])
        test_pred = model.predict(test_data.iloc[:, :-1])[0]

        # Evaluate performance
        train_r2 = r2_score(train_data['revenue_index'], train_pred)

        # Actual test value
        test_actual = test_data['revenue_index'].iloc[0]

        # Percentage error on test point
        test_ape_pct = 100 * abs(test_pred - test_actual) / test_actual

        # Model learned parameters
        print('Intercept:', model.intercept_)

        # Coefficients
        print('Coefficients:')
        for feature, coef in zip(features, model.coef_):
            print(feature, ':', coef)

        # Model evaluation metrics
        print('\nIn-sample R-squared:', train_r2)
        print('2022 Q4 actual revenue_index:', test_actual)
        print('2022 Q4 predicted revenue_index:', test_pred)
        print('2022 Q4 absolute percentage error:', test_ape_pct)

        metrics_dict = {
            'In-sample R-squared': train_r2,
            '2022 Q4 actual revenue_index': test_actual,
            '2022 Q4 predicted revenue_index': test_pred,
            '2022 Q4 absolute percentage error': test_ape_pct
        }

        return metrics_dict

    except Exception as e:
        logger.error('Error during model evauation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    '''Save the evaluation metric as a JSON'''

    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_uri, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_uri': model_uri, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    # Load train data and get features model is trained on
    train_data = load_data('./data/processed/train_data.csv')
    features = train_data.iloc[:, :-1].columns
    features_string = ', '.join(features)

    mlflow.set_experiment("DVC-ML-pipeline")
    with mlflow.start_run(description=f"Linear regression model trained on {features_string}") as run:  # Start an MLflow run
        try:
            # Load model
            model = load_model('./models/model.pkl')

            # Load train and test data
            train_data = load_data('./data/processed/train_data.csv')
            test_data = load_data('./data/processed/test_data.csv')

            # Log train data
            training_dataset = mlflow.data.from_pandas(
                train_data,
                targets="revenue_index",
                name="train_dataset"
            )
            mlflow.log_input(training_dataset, context="training")

            # Evaluate model
            metrics = evaluate_model(model, train_data, test_data)

            # Save metrics
            save_metrics(metrics, './reports/metrics.json')

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model to MLflow
            result = mlflow.sklearn.log_model(model, name="model")

            # Save experiment/model info locally
            save_model_info(run.info.run_id, result.model_uri, "model", './reports/experiment_info.json')

            # Log the metrics file to MLflow
            mlflow.log_artifact('./reports/metrics.json')

            # Log the experiment info file to MLflow
            mlflow.log_artifact('./reports/experiment_info.json')

            # Log the evaluation errors log file to MLflow
            mlflow.log_artifact('model_evaluation_errors.log')


            logger.debug('Model evaluation process completed successfully !')

        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()