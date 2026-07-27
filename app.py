import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Indian Road Accident Severity Prediction")

st.write("Enter accident details")

input_data = {}

for col in columns:
    input_data[col] = st.number_input(col, value=0.0)

df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(df)

    st.success(f"Predicted Accident Severity: {prediction[0]}")
