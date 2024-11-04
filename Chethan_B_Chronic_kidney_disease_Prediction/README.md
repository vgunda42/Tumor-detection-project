Chronic Kidney Disease (CKD) Prediction Project :

Overview
This project aims to predict Chronic Kidney Disease (CKD) progression in patients using clinical data and demographic information. The project includes data preprocessing, feature encoding, model training with a Random Forest classifier, and a web-based application built with Streamlit for prediction. This app allows healthcare professionals or users to input various health metrics to predict the likelihood of CKD.

Project Structure
README.md: Project overview, setup instructions, and usage guide.
src/: Contains Python scripts, including data preprocessing, model training, and the Streamlit app.
data/: Contains the raw and preprocessed CKD dataset.
results/: Stores generated results and model outputs.
Installation Instructions
To set up the project environment and install dependencies, run:

bash
Copy code
pip install -r requirements.txt
Dependencies:

pandas for data manipulation
scikit-learn for model building
joblib for model persistence
streamlit for building the web app
pyngrok for accessing the Streamlit app publicly
Data
The dataset, kidney_disease.csv, contains health metrics and CKD diagnoses for patients. The data undergoes preprocessing steps like handling missing values, encoding categorical features, and scaling numerical features.

Usage
1. Preprocess the Data
Load and preprocess the dataset. Key preprocessing steps include:

Encoding categorical features.
Filling missing values.
Scaling numerical features.
After preprocessing, the data is saved as preprocessed_kidney_disease.csv.

2. Train the Model
Run the RandomForestClassifier to train the model on the processed data. The model is saved as ckd_model.pkl for later use in predictions.

3. Run the Streamlit App
Start the Streamlit app to make CKD predictions based on user input. The app provides an interactive interface for entering relevant patient metrics and health parameters.

To run the app locally:

bash
Copy code
streamlit run app.py
4. Access the App via ngrok
To make the app publicly accessible, use ngrok:

bash
Copy code
from pyngrok import ngrok

# Start ngrok tunnel
public_url = ngrok.connect(8501, "http")
print("ngrok tunnel URL:", public_url)
Features of the Web Application
Interactive Input Fields: Accepts health and demographic data such as blood pressure, hemoglobin, serum creatinine, and more.
Real-Time Prediction: Predicts CKD likelihood based on input metrics.
Result Display: Displays whether the patient is at risk of CKD or likely healthy based on the prediction.
Results
The Random Forest model performs well on the CKD dataset, allowing accurate predictions of CKD progression. The user-friendly interface in the Streamlit app makes it accessible for real-time use.

Important Note
The ngrok tunnel URL changes each time it's restarted, so update the URL as needed to access the app externally.
