# Import libraries
from flask import Flask, render_template, request
import mlflow
import dagshub
import os

# # DagsHub credentials
# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "shubhamyadav2442"
# repo_name = "Revenue-predictor-model-trained-on-engineered-features"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/shubhamyadav2442/Revenue-predictor-model-trained-on-engineered-features.mlflow')

# Make Flask object
app = Flask(__name__)

client = mlflow.MlflowClient()

# Load production model from registry
model_name = 'Revenue_Predictor_LinearRegressor'
try:
    model_uri = f'models:/{model_name}@production'
    model = mlflow.pyfunc.load_model(model_uri)

    production_model = client.get_model_version_by_alias(model_name, 'production')
    production_model_version = production_model.version

    print(f'***************** Production model successfully retrieved from model registry (model version: {production_model_version}) ***************')

except Exception as e:
    print('Unexpected error occurred while retrieving production model from model registry')
    raise

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    sum_spend_per_user = request.form['sum_spend_per_user']
    avg_weekly_active_users_index = request.form['avg_weekly_active_users_index']

    # Predict
    result = round((model.predict([[float(sum_spend_per_user), float(avg_weekly_active_users_index)]])[0]), 3)
    return render_template('index.html', result = str(result))

app.run(debug=True)