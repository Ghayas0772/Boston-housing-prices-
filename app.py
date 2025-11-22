from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load('model/RandomForest_BostonHousing.joblib')

# Define feature order
features = ['ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','PTRATIO','CRIM_log','TAX_log','LSTAT_log']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values
        input_data = [float(request.form[feat]) for feat in features]
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)
        
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    # Make Flask accessible outside the container
    app.run(host='0.0.0.0', port=5000, debug=True)

