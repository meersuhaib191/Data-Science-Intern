import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("loan_model.pkl", "rb"))

st.title("Loan Approval Prediction System")

st.write("Enter applicant details to check loan approval status.")

# User Inputs

income = st.number_input("Applicant Income", min_value=0)

credit_score = st.number_input("Credit Score", min_value=300, max_value=900)

loan_amount = st.number_input("Loan Amount")

years_employed = st.number_input("Years Employed")

points = st.number_input("Points")

# Prediction Button
if st.button("Predict Loan Approval"):

    input_data = np.array([[income, credit_score, loan_amount, years_employed, points]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")