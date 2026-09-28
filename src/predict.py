import pandas as pd
import joblib


MODEL_PATH = "models/telco_churn_model.pkl"


def predict_churn(customer_data):
    model = joblib.load(MODEL_PATH)

    prediction = model.predict(customer_data)[0]
    probability = model.predict_proba(customer_data)[0, 1]

    return {
        "Prediction": prediction,
        "Churn Probability": probability
    }


if __name__ == "__main__":

    new_customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 475.0
    }])

    result = predict_churn(new_customer)

    print(result)