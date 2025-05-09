import streamlit as st
import pickle
import pandas as pd

with open("LinearRegressionModel.pkl", "rb") as file:
    model = pickle.load(file)

unique_df = pd.read_csv("unique_values.csv")
st.title("Car Price Prediction App 🚗💰")

name = st.selectbox("Car Name", unique_df["name"].dropna().unique())
company = st.selectbox("Company", unique_df["company"].dropna().unique())
year = st.selectbox("Year", unique_df["year"].dropna().unique())
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100, value=100)
fuel_type = st.selectbox("Fuel Type", unique_df["fuel_type"].dropna().unique())

if st.button("Predict Price"):
    new_data = pd.DataFrame({
        'name': [name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })
    prediction = model.predict(new_data)[0]
    st.success(f"Estimated Car Price: **₹{round(prediction, 2)}**")
