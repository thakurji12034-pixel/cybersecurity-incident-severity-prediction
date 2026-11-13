from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# 1. Load trained model and feature columns
model = joblib.load("models/random_forest.joblib")
feature_cols = joblib.load("models/feature_columns.joblib")
request_logs = []


# 2. Health check endpoint
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Cybersecurity Incident Prediction API",
        "model_loaded": model is not None
    })


# 3. Model metrics endpoint
@app.route("/metrics", methods=["GET"])
def metrics():
    df = pd.read_csv("outputs/model_comparison.csv")
    return jsonify(df.to_dict(orient="records"))


# 4. Request logs endpoint
@app.route("/logs", methods=["GET"])
def logs():
    return jsonify(request_logs[::-1])


# 5. Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "No input provided"}), 400

    # Get incident record and convert to DataFrame
    record = data.get("record", data)
    df = pd.DataFrame([record])

    # Extract time features if Timestamp is present
    if "Timestamp" in df.columns and df["Timestamp"].notna().any():
        t = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Hour"], df["DayOfWeek"], df["Month"] = t.dt.hour, t.dt.day_of_week, t.dt.month

    # Treat code-like columns as text strings
    id_cols = [
        "OrgId", "DetectorId", "AlertTitle", "ApplicationId",
        "OAuthApplicationId", "ApplicationName", "OSFamily",
        "OSVersion", "CountryCode", "State", "City"
    ]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Match columns with training features
    df = df.reindex(columns=feature_cols)

    # Predict class and probabilities
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    prob_dict = {str(c): round(float(p), 4) for c, p in zip(model.classes_, probabilities)}

    # Record log
    request_logs.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "method": "POST",
        "endpoint": "/predict",
        "status": 200,
        "prediction": str(prediction)
    })
    if len(request_logs) > 50:
        request_logs.pop(0)

    return jsonify({
        "success": True,
        "prediction": str(prediction),
        "probabilities": prob_dict,
        "message": "Prediction completed successfully"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Flask API on 0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
