Stroke Prediction Model A machine learning project that uses patient data to predict the likelihood of a stroke. This application leverages a trained model with a Streamlit interface, allowing users to input health-related data and receive predictions in real-time.

# Table of Contents
Project Overview Dataset Installation Usage Model Training Technologies Used Contributing

# Project Overview
The goal of this project is to help predict strokes based on a set of patient attributes such as age, gender, hypertension, heart disease, BMI, and other health-related factors. The prediction model is built using Random Forest Classifier (or other chosen algorithms) and is deployed as an interactive web app using Streamlit.

# Dataset
Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset# The dataset used for training the model includes the following features: id: Unique identifier gender: Gender of the patient age: Age of the patient hypertension: 1 if the patient has hypertension, 0 otherwise heart_disease: 1 if the patient has heart disease, 0 otherwise ever_married: "Yes" or "No" work_type: Type of occupation Residence_type: "Urban" or "Rural" avg_glucose_level: Average glucose level in blood bmi: Body mass index smoking_status: Smoking status of the patient

Target: stroke: 1 if the patient had a stroke, 0 otherwise

Note: This dataset is fictional and for educational purposes only. Ensure that you have the dataset in your project directory.

# Clone the repository:
git clone https://github.com/yourusername/Stroke_Prediction.git cd Stroke_Prediction Set up a virtual environment: venv\Scripts\activate

Install dependencies: pip install -r requirements.txt

Make sure requirements.txt includes the following libraries: scikit-learn pandas streamlit

# Usage
Run the Streamlit app: streamlit run stroke_streamlit.py

Input Features: Enter patient details through the inputs provided in the Streamlit app. Click "Predict Stroke" to see the prediction result.

# Prediction Output:
The app will display the likelihood of stroke based on the provided data.

# Model Training:
Load and Preprocess Data: Load the data and perform any necessary preprocessing. Train the Model: Train using RandomForestClassifier (or another algorithm of choice). Save the Model: Save the trained model using pickle: import pickle with open("model.pkl", "wb") as file: pickle.dump(model, file)

# Technologies Used:
Python scikit-learn: For model building and evaluation Streamlit: For building the web interface Pandas: For data handling

# Contribution
Contributions are welcome! Please fork this repository and submit a pull request with your changes.
