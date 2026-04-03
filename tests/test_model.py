# Import libraries
import unittest
import mlflow
import os
import pandas as pd
import logging
from sklearn.metrics import r2_score

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
logger.debug('------------------------------------------ MODEL TESTING STARTED -----------------------------------------------------')

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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

        # Load the new model from MLflow model registry
        cls.new_model_name = "Revenue_Predictor_LinearRegressor"
        cls.new_model_uri = f'models:/{cls.new_model_name}@staging'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load train data
        cls.train_data = pd.read_csv('data/processed/train_data.csv')

        # Load test data
        cls.test_data = pd.read_csv('data/processed/test_data.csv')

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_performance(self):

        # Extract independant features and target from the train data
        train_features = self.train_data.iloc[:,0:-1]
        train_target = self.train_data.iloc[:,-1]

        # Predict on train and test data
        train_pred = self.new_model.predict(train_features)
        test_pred = self.new_model.predict(self.test_data.iloc[:, :-1])[0]

        # Evaluate performance metric for the new model on train data
        train_r2 = r2_score(train_target, train_pred)

        # Target test value
        test_target = self.test_data.iloc[:, -1].iloc[0]

        # Percentage error on test point
        test_ape_pct = 100 * abs(test_pred - test_target) / test_target

        # Define expected thresholds for the performance metrics
        expected_r2 = 0.94
        maximum_test_ape_pct = 20

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(train_r2, expected_r2, f'In-sample R2 score should be at least {expected_r2}')
        self.assertLessEqual(test_ape_pct, maximum_test_ape_pct, f'Percentage error on test point should be less than {maximum_test_ape_pct}')

if __name__ == "__main__":
    unittest.main()