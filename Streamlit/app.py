import streamlit as st
import numpy as np
import pandas as pd

st.title("Hello, Streamlit!")
st.write("This is a simple Streamlit application.")

df = pd.DataFrame({
    'Column 1': [1, 2, 3, 4],
    'Column 2': [10, 20, 30, 40]
})

#Display the dataframe
st.write("Here is a random dataframe:")
st.write(df)

#Create line chart
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)

st.line_chart(df)