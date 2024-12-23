import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Load pickled model and preprocessing objects
with open('D:\VS Code Project\churn_model.pkl', 'rb') as model_file:
    ann = pickle.load(model_file)

with open('D:\VS Code Project\scaler.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

with open('D:\VS Code Project\label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

with open('D:\VS Code Project\column_transformer.pkl', 'rb') as ct_file:
    ct = pickle.load(ct_file)

# Streamlit app
st.title("Churn Prediction App")

# User inputs
st.sidebar.header("Input Features")
credit_score = st.sidebar.number_input("Credit Score", min_value=0)
geography = st.sidebar.selectbox("Geography", ("France", "Germany", "Spain"))
gender = st.sidebar.selectbox("Gender", ("Female", "Male"))
age = st.sidebar.number_input("Age", min_value=0)
tenure = st.sidebar.number_input("Tenure", min_value=0)
balance = st.sidebar.number_input("Balance", min_value=0.0, format="%.2f")
num_of_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=4)
has_cr_card = st.sidebar.selectbox("Has Credit Card", (0, 1))
is_active_member = st.sidebar.selectbox("Is Active Member", (0, 1))
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, format="%.2f")

# Transform user input
user_data = np.array([[credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])
user_data[:, 2] = le.transform(user_data[:, 2])
user_data = np.array(ct.transform(user_data))
user_data = sc.transform(user_data)

# Predict churn
if st.button("Predict"):
    prediction = ann.predict(user_data)
    prediction = (prediction > 0.5)
    result = "Churn" if prediction else "No Churn"
    st.write(f"The prediction is: **{result}**")
