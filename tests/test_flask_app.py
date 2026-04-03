# Import libraries
import unittest
from flask_app.app import app

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