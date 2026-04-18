from flask import Flask, request, render_template
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('model/churn_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Collect all 8 features from form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        tenure = int(request.form['tenure'])
        monthly_charges = float(request.form['monthly_charges'])
        contract = int(request.form['contract'])
        internet_service = int(request.form['internet_service'])
        support_calls = int(request.form['support_calls'])
        payment_method = int(request.form['payment_method'])

        # Convert to array with 8 features
        input_data = np.array([[age, gender, tenure, monthly_charges,
                                contract, internet_service, support_calls, payment_method]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        return render_template('index.html', prediction_text=f"Result: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
