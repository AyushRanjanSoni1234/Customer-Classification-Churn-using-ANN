import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import joblib
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

label_encoders = label_encoders.fit(['Male', 'Female'])

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

## Streamlit App
st.title("Customer Churn Prediction")

## User Input
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoders.classes_)
age = st.number_input('Age', min_value=18, max_value=92, value=30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)  
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoders.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geography_encoded = onehot_encoder.transform([[geography]])
geography_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

# Combine input data with one-hot encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geography_df], axis=1)

# Scale numerical features
input_data_scaled = scaler.transform(input_data)

# Prediction Chunk
if st.button('Predict Churn'):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]
    if churn_probability > 0.5:
        st.error(f'The customer is likely to churn with a probability of {churn_probability:.2f}')
    else:
        st.success(f'The customer is unlikely to churn with a probability of {churn_probability:.2f}')