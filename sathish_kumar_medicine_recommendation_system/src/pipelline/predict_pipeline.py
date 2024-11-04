import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from logger import logging
from exception import CustomException
from utils import load_object

symtoms_data = pd.read_csv("S:/DS/Prince_Chhirolya/ChhirolyaTech_AI_ML_Intern_Candidates_Assignments/sathish_kumar_medicine_recommendation_system/data/symtoms_df.csv")
precautions_data = pd.read_csv("S:/DS/Prince_Chhirolya/ChhirolyaTech_AI_ML_Intern_Candidates_Assignments/sathish_kumar_medicine_recommendation_system/data/precautions_df.csv")
workout_data = pd.read_csv("S:/DS/Prince_Chhirolya/ChhirolyaTech_AI_ML_Intern_Candidates_Assignments/sathish_kumar_medicine_recommendation_system/data/workout_df.csv")
description_data = pd.read_csv("S:/DS/Prince_Chhirolya/ChhirolyaTech_AI_ML_Intern_Candidates_Assignments/sathish_kumar_medicine_recommendation_system/data/description.csv")
medications_data = pd.read_csv('S:/DS/Prince_Chhirolya/ChhirolyaTech_AI_ML_Intern_Candidates_Assignments/sathish_kumar_medicine_recommendation_system/data/medications.csv')
diet_data = pd.read_csv("S:/DS/Prince_Chhirolya/ChhirolyaTech_AI_ML_Intern_Candidates_Assignments/sathish_kumar_medicine_recommendation_system/data/diets.csv")


df = pd.read_csv("S:/DS/Prince_Chhirolya/ChhirolyaTech_AI_ML_Intern_Candidates_Assignments/sathish_kumar_medicine_recommendation_system/data/Training.csv")
symptoms_dict = {col: idx for idx, col in enumerate(df.columns[:-1])} 

def helper(dis):
    try:

        description = description_data[description_data['Disease'] == dis]['Description']
        descriptions = " ".join([i for i in description])

        precaution = precautions_data[precautions_data['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        precautions = [j for j in precaution.values]

        medication = medications_data[medications_data['Disease'] == dis]['Medication']
        medications = [k for k in medication.values]

        diet = diet_data[diet_data['Disease'] == dis]['Diet']
        diets = [l for l in diet.values]

        workout = workout_data[workout_data['disease'] == dis] ['workout']


        return descriptions,precautions,medications,diets,workout

    except Exception as e:
        raise CustomException(e,sys)
    
# Assuming symptoms_dict is the dictionary that maps symptoms to indices
def get_predicted_value(patient_symptoms):
    try:
        input_vector = np.zeros(len(symptoms_dict))  # Initialize input vector
        model_path=os.path.join("artifacts","model.pkl")
        label_path=os.path.join('artifacts','class_mapping.pkl')
        model=load_object(file_path=model_path)
        label=load_object(file_path=label_path)
        for symptom in patient_symptoms:
            if symptom in symptoms_dict:
                input_vector[symptoms_dict[symptom]] = 1
            else:
                logging.warning(f"Symptom '{symptom}' not found in symptoms dictionary.")
                # Optionally: add additional handling here for missing symptoms

        # Predict disease
        predicted_disease = label[model.predict([input_vector])[0]]
        return predicted_disease

    except Exception as e:
        raise CustomException(e, sys)
    


# # Model Prediction function
# def get_predicted_value(patient_symptoms):
#     try:
#         model_path=os.path.join("artifacts","model.pkl")
#         label_path=os.path.join('artifacts','class_mapping.pkl')
#         print("Before Loading")
#         model=load_object(file_path=model_path)
#         label=load_object(file_path=label_path)
#         print("After Loading")
#         input_vector = np.zeros(len(symptoms_dict))
#         for item in patient_symptoms:
#             input_vector[symptoms_dict[item]] = 1
#         return label[model.predict([input_vector])[0]]

#     except Exception as e:
#         raise CustomException(e,sys)

