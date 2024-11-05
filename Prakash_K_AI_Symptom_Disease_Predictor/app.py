import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained(r'D:\datascience\python\project praactice\symptom_diagnosis_model')
tokenizer = BertTokenizer.from_pretrained(r'D:\datascience\python\project praactice\symptom_diagnosis_model')

# Load the precaution data
precaution_data = pd.read_csv(r'D:\datascience\python\project praactice\Disease precaution.csv')

# Streamlit app
st.title("Symptom-to-Diagnosis Predictor")

# User input for symptoms
user_input = st.text_area("Enter your symptoms (comma separated):")


symptom_data = pd.read_csv(r'D:\datascience\python\project praactice\DiseaseAndSymptoms.csv')

# Extract unique disease names to create labels
labels = symptom_data['Disease'].unique().tolist()

# Function to predict diagnosis
def predict_diagnosis(symptoms):
    # Process symptoms and make prediction
    inputs = tokenizer(symptoms, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Map the predicted index to the corresponding disease label
    disease = labels[predictions.item()]
    return disease

# Predict button
if st.button("Predict Diagnosis"):
    if user_input:
        # Combine symptoms into a single string
        symptoms = user_input.split(',')
        combined_symptoms = ' '.join(symptom.strip() for symptom in symptoms)
        
        # Get diagnosis
        diagnosis = predict_diagnosis(combined_symptoms)
        
        # Show the diagnosis
        st.success(f"Predicted Diagnosis: {diagnosis}")
        
        # Show precautions related to the diagnosis
        precautions = precaution_data[precaution_data['Disease'] == diagnosis]
        st.write("Precautions:")
        for _, row in precautions.iterrows():
            st.write(f"- {row['Precaution_1']}")
            st.write(f"- {row['Precaution_2']}")
            st.write(f"- {row['Precaution_3']}")
            st.write(f"- {row['Precaution_4']}")
    else:
        st.error("Please enter symptoms to predict the diagnosis.")
