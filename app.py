import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
def load_data():
    return pd.read_csv('medical_insurance.csv')

# Data Preprocessing
def preprocess_data(df):
    df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    df.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)
    df['BMI_smoker'] = df['bmi'] * df['smoker']
    return df

# Load and preprocess the data
medical_df = preprocess_data(load_data())

# Prepare data for training
X = medical_df.drop('charges', axis=1)
y = medical_df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=2)
rf.fit(X_train, y_train)

# Sidebar
st.sidebar.title("Dashboard")
sidedash = st.sidebar.selectbox("Select Page", ["Home", "Medical Insurance Charges Prediction"])

# Home page
if sidedash == "Home":
    st.title("Medical Insurance Charges Prediction System")
    st.markdown("""
    <style>
       .stApp{
           background-image: url('https://i.pinimg.com/originals/c5/16/ff/c516ff9163fefeaa5974fc7c8855cd02.jpg');
           background-size: cover;
           background-position: center center;
           height: 100%;
       }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    Welcome to the **Medical Insurance Charges Prediction App**!

    This app helps you predict the cost of your medical insurance based on certain details. Simply enter your age, sex, BMI, the number of children, smoking status, and region to get an estimate of your insurance charges.

    ### How It Works:
    1. **Input Your Details**: You will be asked to enter your personal information such as age, sex, BMI, smoking habits, number of children, and the region where you live.
    2. **Click on the "Predict" Button**: Our app will predict your estimated medical insurance charges in just a few seconds.
    3. **Simple & Quick**: See your prediction and make informed decisions about your insurance options.

    ### Ready to get your medical insurance estimate? 
    Go to the **Medical Insurance Charges Prediction** Page in the sidebar to get started.
    """, unsafe_allow_html=True)

# Medical Insurance Charges Prediction page
elif sidedash == "Medical Insurance Charges Prediction":
    st.title("Medical Insurance Charges Prediction")
    st.write("Enter the details below to predict the medical insurance charges:")
    st.markdown("""
    <style>
       .stApp{
           background-image: url('https://i.pinimg.com/originals/c5/16/ff/c516ff9163fefeaa5974fc7c8855cd02.jpg');
           background-size: cover;
           background-position: center center;
           height: 100%;
       }
    </style>
    """, unsafe_allow_html=True)
    # Input fields for prediction
    age = st.number_input("Age", min_value=0, max_value=120)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    bmi = st.number_input("BMI", min_value=0.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", options=["Yes", "No"])
    region = st.selectbox("Region", options=["Southeast", "Southwest", "Northwest", "Northeast"])

    # Prediction logic
    if st.button("Predict"):
        sex = 0 if sex == "Male" else 1
        smoker = 0 if smoker == "Yes" else 1
        region = ["Southeast", "Southwest", "Northwest", "Northeast"].index(region)
        
        # Prepare input features and make prediction
        input_features = np.array([age, sex, bmi, children, smoker, region, bmi * smoker]).reshape(1, -1)
        predicted_cost = rf.predict(input_features)
        
        st.subheader(f"Predicted Medical Insurance Charges: **${predicted_cost[0]:.2f}**")
