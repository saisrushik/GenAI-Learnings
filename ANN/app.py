import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

#laod the trained model
model = tf.keras.models.load_model('trained_model.h5')

#Load the trained model, scaler onehot encoder pickle files
model = load_model('model.h5')

#load scaler and encoder
with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

#load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Steamlit app
st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn probability.")

#Input fields
credit_score = st.number_input("Credit Score")
geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100)
estimated_salary = st.number_input("Estimated Salary")
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10)
balance = st.number_input("Balance")
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])

# Example input data
input_data = {
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard':   [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

#Convert geography using onehot encoder
geo_encoded = onehot_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

#Create dataframe
input_df = pd.DataFrame([input_data])

#Convert gender using label encoder
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])


#Combine encoded geography with input dataframe
input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
input_df

#Scale the input data
scaled_input = scaler.transform(input_df)

#Predict churn probability
pred = model.predict(scaled_input)
pred_probability = pred[0][0]


if pred_probability >= 0.5:
    st.error(f"The customer is likely to churn with a probability of {pred_probability:.2f}")
else:
    st.success(f"The customer is unlikely to churn with a probability of {pred_probability:.2f}")