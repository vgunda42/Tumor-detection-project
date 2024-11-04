import sys
import pandas as pd
import numpy as np
import os
from src.exception import CustomException
from src.utils import load_object

# load databasedataset===================================
symtoms_data = pd.read_csv("ChhirolyaTech_AI_ML_Intern_Candidates_Assignments\sathish_kumar_medicine_recommendation_system\data\symtoms_df.csv")
precautions_data = pd.read_csv("ChhirolyaTech_AI_ML_Intern_Candidates_Assignments\sathish_kumar_medicine_recommendation_system\data\precautions_df.csv")
workout_data = pd.read_csv("ChhirolyaTech_AI_ML_Intern_Candidates_Assignments\sathish_kumar_medicine_recommendation_system\data\workout_df.csv")
description_data = pd.read_csv("ChhirolyaTech_AI_ML_Intern_Candidates_Assignments\sathish_kumar_medicine_recommendation_system\data\description.csv")
medications_data = pd.read_csv('ChhirolyaTech_AI_ML_Intern_Candidates_Assignments\sathish_kumar_medicine_recommendation_system\data\medications.csv')
diet_data = pd.read_csv("ChhirolyaTech_AI_ML_Intern_Candidates_Assignments\sathish_kumar_medicine_recommendation_system\data\diets.csv")


df = pd.read_csv("ChhirolyaTech_AI_ML_Intern_Candidates_Assignments\sathish_kumar_medicine_recommendation_system\data\Training.csv")
symptoms_dict = {col: idx for idx, col in enumerate(df.columns)}
symptoms_dict


def helper(dis):
    try:

        description = description_data[description_data['Disease'] == dis]['Description']
        description = " ".join([i for i in description])

        precautions = precautions_data[precautions_data['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        precautions = [i for i in precautions.values]

        medications = medications_data[medications_data['Disease'] == dis]['Medication']
        medications = [i for i in medications.values]

        diet = diet_data[diet_data['Disease'] == dis]['Diet']
        diet = [i for i in diet.values]

        workout = workout_data[workout_data['disease'] == dis] ['workout']


        return description,precautions,medications,diet,workout

    except Exception as e:
        raise CustomException(e,sys)


# Model Prediction function
def get_predicted_value(patient_symptoms):
    try:
        model_path=os.path.join("artifacts","model.pkl")
        label_path=os.path.join('artifacts','label_encoded_data.pkl')
        print("Before Loading")
        model=load_object(file_path=model_path)
        label=load_object(file_path=label_path)
        print("After Loading")
        input_vector = np.zeros(len(symptoms_dict))
        for item in patient_symptoms:
            input_vector[symptoms_dict[item]] = 1
        return label[model.predict([input_vector])[0]]

    except Exception as e:
        raise CustomException(e,sys)

