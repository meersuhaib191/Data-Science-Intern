import streamlit as st
import pickle
import numpy as np

# load model
model = pickle.load(open("salary_model.pkl","rb"))

st.title("Salary Prediction App")

st.write("Enter years of experience to predict salary")

experience = st.slider("Years of Experience",0,20)

if st.button("Predict Salary"):

    prediction = model.predict([[experience]])

    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")