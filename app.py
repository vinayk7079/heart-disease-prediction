from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        float(request.form['age']),
        int(request.form['sex']),
        int(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        int(request.form['fbs']),
        int(request.form['restecg']),
        float(request.form['thalach']),
        int(request.form['exang']),
        float(request.form['oldpeak']),
        int(request.form['slope']),
        float(request.form['ca']),
        int(request.form['thal'])
    ]

    # Scale the input features
    features_scaled = scaler.transform([features])

    # Predict probability
    probability = model.predict_proba(features_scaled)[0][1] * 100  # Probability of heart disease

    # Render result page
    return render_template('result.html', probability=probability)

if __name__ == '__main__':
    app.run(debug=True)