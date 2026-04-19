from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np

app = Flask(__name__)

# -----------------------------
# Load model safely (Render fix)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "churn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Health check (Render uses this)
# -----------------------------
@app.route("/health")
def health():
    return "OK"

# -----------------------------
# Prediction from HTML form
# -----------------------------
@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        tenure = int(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        contract = int(request.form["contract"])
        internet_service = int(request.form["internet_service"])
        support_calls = int(request.form["support_calls"])
        payment_method = int(request.form["payment_method"])

        features = np.array([[age, gender, tenure, monthly_charges,
                              contract, internet_service, support_calls,
                              payment_method]])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        result = "Customer Will Churn ❌" if prediction == 1 else "Customer Will Stay ✅"
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# JSON API (for testing/postman)
# -----------------------------
@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        features = np.array([[ 
            data["age"],
            data["gender"],
            data["tenure"],
            data["monthly_charges"],
            data["contract"],
            data["internet_service"],
            data["support_calls"],
            data["payment_method"]
        ]])

        scaled = scaler.transform(features)
        prediction = int(model.predict(scaled)[0])

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# Local run only (not used by Render)
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
