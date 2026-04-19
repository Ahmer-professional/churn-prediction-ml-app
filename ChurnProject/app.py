from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# -------------------------------
# Load ML model safely
# -------------------------------
model = None
try:
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)

# -------------------------------
# Health check route (REQUIRED by Render)
# -------------------------------
@app.route("/health")
def health():
    return "OK"

# -------------------------------
# Home page
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Prediction API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        credit_score = int(data["credit_score"])
        age = int(data["age"])
        tenure = int(data["tenure"])
        balance = float(data["balance"])
        products = int(data["products"])
        active_member = int(data["active_member"])
        salary = float(data["salary"])

        features = np.array([[credit_score, age, tenure, balance, products, active_member, salary]])
        prediction = model.predict(features)[0]

        result = "Customer Will Leave ❌" if prediction == 1 else "Customer Will Stay ✅"
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

# -------------------------------
# IMPORTANT: Bind PORT for Render
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
