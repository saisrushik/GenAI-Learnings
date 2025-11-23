import streamlit as st
import pandas as pd


st.title("Streamlit Text Input")

name = st.text_input("Enter your name:", )

age = st.slider("Select your age:", 0, 100, 25)


if name:
    st.write(f"Hello, {name}!")

st.write(f"You are {age} years old.")

options = ["Python", "Java", "JavaScript", "Selenium"]
choice = st.selectbox("Select your favorite programming language:", options)    
st.write(f"You selected: {choice}")


data = {
    "Name": ["John", "Alice", "Bob"],
    "Age": [28, 24, 22],
    "City": ["New York", "Los Angeles", "Chicago"]
}

df = pd.DataFrame(data)
st.write("Here is a sample dataframe:")
st.write(df)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"])
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Here is the uploaded dataframe:")
    st.write(uploaded_df)
