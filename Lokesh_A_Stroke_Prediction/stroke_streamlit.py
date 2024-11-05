import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
with open("stroke_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

scaler = StandardScaler()
encoder = LabelEncoder()

st.title("Stroke Prediction App")

# Input fields for user to enter data
st.header("Enter Patient Details:")
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100, value=67)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=120.0)
bmi = st.number_input('BMI', min_value=0.0, value=25.0)
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Dictionary to hold input data for prediction
input_data = {
    'gender': gender,
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'work_type': work_type,
    'Residence_type': residence_type,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': smoking_status
}

input_df = pd.DataFrame([input_data])

# Encoding categorical variables
input_df['gender'] = encoder.fit_transform(input_df['gender'])
input_df['ever_married'] = encoder.fit_transform(input_df['ever_married'])
input_df['work_type'] = encoder.fit_transform(input_df['work_type'])
input_df['Residence_type'] = encoder.fit_transform(input_df['Residence_type'])
input_df['smoking_status'] = encoder.fit_transform(input_df['smoking_status'])

# Scaling numerical features
numerical_features = ['age', 'avg_glucose_level', 'bmi']
input_df[numerical_features] = scaler.fit_transform(input_df[numerical_features])

# Prediction button
if st.button("Predict Stroke"):
    prediction = model.predict(input_df)[0]
    result = "Stroke" if prediction == 1 else "No Stroke"
    st.write(f"**{result}**")