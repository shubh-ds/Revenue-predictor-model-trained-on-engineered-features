# Import libraries
import unittest
from flask_app.app import app
import logging

# Logging configuration
logger = logging.getLogger('flask_app_testing')
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
logger.debug('------------------------------------------ FLASK APP TESTING STARTED -----------------------------------------------------')

class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Revenue Predictor</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data={
                                                        'sum_spend_per_user': '133.08',
                                                        'avg_weekly_active_users_index': '212.96'
                                                    })
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()