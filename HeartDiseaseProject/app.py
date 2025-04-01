from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('stacked_model.pkl', 'rb') as f:
    stacked_model = pickle.load(f)

meta_model = stacked_model["meta_model"]
base_models = stacked_model["base_models"]

# Flask app initialization
app = Flask(__name__)

# Index page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        input_data = {
            'age': float(request.form['age']),
            'sex': int(request.form['sex']),
            'chest pain type': int(request.form['chest_pain_type']),
            'resting bp s': float(request.form['resting_bp']),
            'cholesterol': float(request.form['cholesterol']),
            'fasting blood sugar': int(request.form['fasting_blood_sugar']),
            'resting ecg': int(request.form['resting_ecg']),
            'max heart rate': float(request.form['max_heart_rate']),
            'exercise angina': int(request.form['exercise_angina']),
            'oldpeak': float(request.form['oldpeak']),
            'ST slope': int(request.form['st_slope']),
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Generate predictions from base models
        base_predictions = np.zeros((1, len(base_models)))  # To store predictions from base models

        # Get predictions from each base model
        for i, model in enumerate(base_models):
            base_predictions[:, i] = model.predict(input_df)

        # Final prediction using the meta-model (get the probability)
        final_prediction = meta_model.predict(base_predictions)  # Returns 0 or 1
        final_prediction_proba = meta_model.predict_proba(base_predictions)[:, 1]  # Get the probability of having heart disease

        # Extract scalar probability value (if it is an array)
        final_prediction_proba_value = final_prediction_proba[0] if isinstance(final_prediction_proba, np.ndarray) else final_prediction_proba

        # Print the prediction probability in the terminal (PyCharm)
        print(f"Meta-model prediction probability: {final_prediction_proba_value:.2f}")

        # Display result
        if final_prediction == 1:
            result = f"The patient has heart disease. (Probability: {final_prediction_proba_value:.2f})"
            background_image = 'heartdisease.jpg'  # Set image for heart disease
        else:
            result = f"The patient does not have heart disease. (Probability: {final_prediction_proba_value:.2f})"
            background_image = 'goodheart.jpg'  # Set image for no heart disease

        # Pass result and background image to template
        return render_template('result.html', result=result, background_image=background_image)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
