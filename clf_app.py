import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib



# Load the saved objects
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')  # If model includes the full pipeline, you don't need preprocessor separately



# 2. Input UI
st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ['Male', 'Female'])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['Yes', 'No'])
tenure = st.number_input("Tenure (in months)", min_value=0.0)
PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No'])
OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No'])
DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No'])
TechSupport = st.selectbox("Tech Support", ['Yes', 'No'])
StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No'])
StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
PaymentMethod = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# 3. Define input DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [int(SeniorCitizen)],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})


# 5. Prediction function
def predictiveclf():
    processed_input = preprocessor.transform(input_data)
    prediction = model.predict(processed_input)
    return "Churn" if prediction[0] == 1 else "Not Churn"

# 6. Trigger prediction
if st.button("Predict"):
    try:
        result = predictiveclf()
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
