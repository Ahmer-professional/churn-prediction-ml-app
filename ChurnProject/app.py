from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# -----------------------------
# Lazy loading (CRITICAL FIX)
# -----------------------------
model = None
scaler = None

def load_artifacts():
    global model, scaler
    
    if model is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "model", "churn_model.pkl")
        scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

        model = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))

# -----------------------------
# Health route (Render check)
# -----------------------------
@app.route("/health")
def health():
    return "OK"

# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        load_artifacts()   # Load model here

        age = int(request.form["age"])
        gender = int(request.form["gender"])
        tenure = int(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        contract = int(request.form["contract"])
        internet_service = int(request.form["internet_service"])
        support_calls = int(request.form["support_calls"])
        payment_method = int(request.form["payment_method"])

        input_data = np.array([[age, gender, tenure, monthly_charges,
                                contract, internet_service, support_calls, payment_method]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        result = "Churn" if prediction == 1 else "No Churn"

        return render_template("index.html",
                               prediction_text=f"Result: {result}")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")

# -----------------------------
# Run locally only
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
