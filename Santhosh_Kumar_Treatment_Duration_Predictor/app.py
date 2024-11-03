import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Constants for input validation
INPUT_RANGES = {
    'Age': (18, 80),
    'BMI': (16.0, 45.0),
    'Disease_Severity': (1, 10),
    'Heart_Rate': (50, 120),
    'Glucose_Level': (70, 200),
    'Oxygen_Saturation': (90, 100),
    'Treatment_Compliance': (1, 5)
}

def load_model():
    """Load the trained model and scaler"""
    with open('treatment_duration_model.pkl', 'rb') as f:
        model, scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

def perform_risk_analysis(input_data):
    """Perform comprehensive risk analysis"""
    risk_factors = []
    risk_explanations = []
    recommendations = []
    
    # BMI Analysis
    if input_data['BMI'] >= 30:
        risk_factors.append("High BMI")
        risk_explanations.append("BMI ≥ 30 indicates obesity")
        recommendations.append("Consider nutrition consultation and structured exercise program")
    
    # Glucose Level Analysis
    if input_data['Glucose_Level'] > 126:
        risk_factors.append("Elevated Glucose")
        risk_explanations.append("Fasting glucose > 126 mg/dL indicates diabetes risk")
        recommendations.append("Regular glucose monitoring and dietary management recommended")
    
    # Blood Pressure Analysis
    if input_data['Blood_Pressure'] == 'High':
        risk_factors.append("High Blood Pressure")
        risk_explanations.append("High blood pressure increases treatment complexity")
        recommendations.append("Regular blood pressure monitoring and medication review needed")
    
    # Smoking Status Analysis
    if input_data['Smoking_Status'] == 'Current':
        risk_factors.append("Current Smoker")
        risk_explanations.append("Smoking may complicate treatment and recovery")
        recommendations.append("Smoking cessation program recommended")
    
    # Exercise Level Analysis
    if input_data['Exercise_Level'] == 'Sedentary':
        risk_factors.append("Sedentary Lifestyle")
        risk_explanations.append("Low physical activity may slow recovery")
        recommendations.append("Gradual increase in physical activity with professional guidance")
    
    # Treatment Compliance Analysis
    if input_data['Treatment_Compliance'] <= 2:
        risk_factors.append("Low Treatment Compliance")
        risk_explanations.append("Poor compliance history may affect treatment success")
        recommendations.append("Consider patient education program and simplified treatment plan")
    
    # Heart Rate Analysis
    if input_data['Heart_Rate'] > 100:
        risk_factors.append("Elevated Heart Rate")
        risk_explanations.append("Resting heart rate > 100 bpm indicates stress")
        recommendations.append("Consider stress management techniques and cardiac evaluation")
    
    # Oxygen Saturation Analysis
    if input_data['Oxygen_Saturation'] < 95:
        risk_factors.append("Low Oxygen Saturation")
        risk_explanations.append("O2 saturation < 95% may indicate respiratory issues")
        recommendations.append("Regular monitoring of oxygen levels and respiratory assessment")
    
    return risk_factors, risk_explanations, recommendations

def create_prediction_input():
    """Create input fields for prediction"""
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=45)
        bmi = st.number_input("BMI", min_value=16.0, max_value=45.0, value=25.0)
        severity = st.slider("Disease Severity", 1, 10, 5)
        heart_rate = st.number_input("Heart Rate (bpm)", 50, 120, 75)
        
    with col2:
        glucose = st.number_input("Glucose Level (mg/dL)", 70, 200, 100)
        oxygen = st.number_input("Oxygen Saturation (%)", 90, 100, 97)
        compliance = st.slider("Treatment Compliance", 1, 5, 3)
    
    st.subheader("Medical History")
    col3, col4 = st.columns(2)
    
    with col3:
        chronic = st.checkbox("Chronic Condition")
        previous = st.checkbox("Previous Treatments")
        diabetes = st.checkbox("Diabetes")
        
    with col4:
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
    
    st.subheader("Lifestyle Factors")
    col5, col6 = st.columns(2)
    
    with col5:
        blood_pressure = st.selectbox("Blood Pressure Category",
                                    ['Normal', 'Elevated', 'High'])
        exercise = st.selectbox("Exercise Level",
                              ['Sedentary', 'Light', 'Moderate', 'Active'])
    
    with col6:
        smoking = st.selectbox("Smoking Status",
                             ['Never', 'Former', 'Current'])
    
    return {
        'Age': age,
        'BMI': bmi,
        'Disease_Severity': severity,
        'Heart_Rate': heart_rate,
        'Glucose_Level': glucose,
        'Oxygen_Saturation': oxygen,
        'Treatment_Compliance': compliance,
        'Chronic_Condition': int(chronic),
        'Previous_Treatments': int(previous),
        'Diabetes': int(diabetes),
        'Hypertension': int(hypertension),
        'Heart_Disease': int(heart_disease),
        'Blood_Pressure': blood_pressure,
        'Exercise_Level': exercise,
        'Smoking_Status': smoking
    }

def display_risk_analysis(risk_factors, risk_explanations, recommendations):
    """Display risk analysis results in an organized manner"""
    if risk_factors:
        st.subheader("🚨 Risk Analysis")
        
        # Display risk factors and explanations
        st.markdown("### Identified Risk Factors:")
        for factor, explanation in zip(risk_factors, risk_explanations):
            with st.expander(f"📌 {factor}"):
                st.write(explanation)
        
        # Display recommendations
        st.markdown("### 📋 Recommendations:")
        for rec in recommendations:
            st.markdown(f"- {rec}")
            
        # Calculate overall risk level
        risk_level = len(risk_factors)
        if risk_level >= 4:
            st.error("⚠️ High Risk - Close monitoring recommended")
        elif risk_level >= 2:
            st.warning("⚠️ Moderate Risk - Regular monitoring advised")
        else:
            st.info("⚠️ Low Risk - Standard monitoring")
    else:
        st.success("✅ No significant risk factors identified")

def main():
    st.set_page_config(layout="wide", page_title="Treatment Duration Predictor")
    st.title("Healthcare Treatment Duration Predictor")
    
    try:
        model, scaler, feature_names = load_model()
        
        page = st.sidebar.selectbox("Choose a page", 
                                  ["Make Prediction", 
                                   "About"])
        
        if page == "Make Prediction":
            input_data = create_prediction_input()
            
            if st.button("Analyze Patient"):
                # Prepare input data for prediction
                input_df = pd.DataFrame([input_data])
                input_encoded = pd.get_dummies(input_df, 
                                            columns=['Blood_Pressure', 
                                                    'Exercise_Level', 
                                                    'Smoking_Status'])
                
                # Align columns with training data
                for col in feature_names:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                input_encoded = input_encoded[feature_names]
                
                # Scale features and predict
                input_scaled = scaler.transform(input_encoded)
                prediction = model.predict(input_scaled)[0]
                
                # Display results in columns
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### 🏥 Treatment Duration Prediction")
                    st.success(f"### {prediction:.0f} days")
                
                with col2:
                    # Perform risk analysis
                    risk_factors, risk_explanations, recommendations = perform_risk_analysis(input_data)
                    display_risk_analysis(risk_factors, risk_explanations, recommendations)
        
        else:  # About page
            st.header("About This Application")
            st.write("""
            This application predicts the duration of medical treatments based on various patient characteristics and medical history. It uses a machine learning model trained on healthcare data to make predictions and perform risk analysis.
            
            The system analyzes:
            - Basic patient information (age, BMI)
            - Medical measurements (heart rate, glucose level, oxygen saturation)
            - Medical history (chronic conditions, previous treatments)
            - Lifestyle factors (exercise level, smoking status)
            
            The risk analysis component evaluates multiple factors to identify potential complications and provide targeted recommendations.
            
            Please note that this is a predictive tool and should not replace professional medical judgment.
            """)
            
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model and feature names files are in the correct location.")

if __name__ == "__main__":
    main()