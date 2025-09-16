import streamlit as st
import pandas as pd
import numpy as np
import pickle


scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Social Media Ads Classifier")
st.write("This app predicts whether a user is likely to make a purchase based on their demographic information.")


st.sidebar.header("User Input Features")

def user_input():
    age = st.sidebar.slider("Age", 18, 60, 30)
    estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=10000, max_value=1000000, value=50000)
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    if gender == "Male":
        gender_binary = 1 
    else:
        gender_binary=0
    input_data = pd.DataFrame({
        "Age": [age],
        "EstimatedSalary": [estimated_salary],
        "Gender_male":[gender_binary]
    })
    return input_data


input_data = user_input()


st.write("### User Input:")
st.write(input_data)


scaled_input = scaler.transform(input_data)


prediction = model.predict(scaled_input)[0]

st.write("### Prediction Result:")
if prediction == 1:
    st.success("The user is **likely to make a purchase**.")
else:
    st.error("The user is **not likely to make a purchase**.")

    