from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return "OK"

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

        features = [[age, gender, tenure, monthly_charges, contract,
                     internet_service, support_calls, payment_method]]

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        result = "Customer Will Churn ❌" if prediction == 1 else "Customer Will Stay ✅"
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
