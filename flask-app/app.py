# Import libraries
from flask import Flask, render_template, request
import mlflow
import dagshub

# Set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/shubhamyadav2442/Revenue-predictor-model-trained-on-engineered-features.mlflow')

# Initialize dagshub
dagshub.init(repo_owner='shubhamyadav2442', repo_name='Revenue-predictor-model-trained-on-engineered-features', mlflow=True)

# Make Flask object
app = Flask(__name__)

# Load model from registry
model_name = 'Revenue_Predictor_LinearRegressor'
model_version = 1

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

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