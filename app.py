import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__, template_folder='templates', static_folder='static')

# Ensure these paths are correct
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
encoders_path = os.path.join(os.path.dirname(__file__), 'label_encoders.pkl')
disease_encoder_path = os.path.join(os.path.dirname(__file__), 'disease_encoder.pkl')

# Load the model, label encoders, and disease encoder
model = pickle.load(open(model_path, "rb"))
le_dict = pickle.load(open(encoders_path, "rb"))
disease_encoder = pickle.load(open(disease_encoder_path, "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    input_features = [request.form.get(key) for key in request.form.keys()]

    # Ensure no empty input values
    if '' in input_features:
        return render_template("index.html", prediction_text="Error: Please fill in all fields.")

    # Convert categorical features to numerical values
    feature_names = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Age", "Gender", "Blood Pressure", "Cholesterol Level"]
    encoded_features = []
    for feature, name in zip(input_features, feature_names):
        if name in le_dict:
            # If the feature is categorical, transform it
            encoded_features.append(le_dict[name].transform([feature])[0])
        else:
            # If the feature is numerical, convert it to float
            encoded_features.append(float(feature))

    features = np.array([encoded_features])
    prediction = model.predict(features)
    prediction_decoded = disease_encoder.inverse_transform(prediction)
    return render_template("index.html", prediction_text="The predicted disease is {}".format(prediction_decoded[0]))

if __name__ == "__main__":
    flask_app.run(debug=True)
