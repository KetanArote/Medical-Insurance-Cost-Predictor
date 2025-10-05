# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and column order
bundle = joblib.load("insurance_gbr.joblib")
model = bundle['model']
feature_order = bundle['columns']

# Mappings (same as training)
sex_map = {'female': 0, 'male': 1}
smoker_map = {'no': 0, 'yes': 1}
region_map = {'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_form():
    # Get user inputs as strings (for display)
    age = request.form["age"]
    sex = request.form["sex"]
    bmi = request.form["bmi"]
    children = request.form["children"]
    smoker = request.form["smoker"]
    region = request.form["region"]

    # Encode for model prediction
    row = pd.DataFrame([{
        "age": float(age),
        "sex": sex_map[sex],
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker_map[smoker],
        "region": region_map[region]
    }])[feature_order]

    # Predict
    pred = float(model.predict(row)[0])

    # Pass prediction + original values to template
    return render_template(
        "index.html",
        prediction=round(pred, 2),
        age=age,
        sex=sex,
        bmi=bmi,
        children=children,
        smoker=smoker,
        region=region
    )

@app.route("/api/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(force=True)
    row = pd.DataFrame([{
        "age": float(payload["age"]),
        "sex": sex_map[payload["sex"]],
        "bmi": float(payload["bmi"]),
        "children": int(payload["children"]),
        "smoker": smoker_map[payload["smoker"]],
        "region": region_map[payload["region"]]
    }])[feature_order]

    pred = float(model.predict(row)[0])
    return jsonify({"predicted_charge": pred})

if __name__ == '__main__':
    app.run(debug=True)
