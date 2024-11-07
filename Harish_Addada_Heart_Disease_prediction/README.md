# Heart Disease Prediction

## Project Overview
The Heart Disease Risk Prediction project is an advanced machine learning initiative aimed at predicting the likelihood of heart disease in individuals based on clinical and demographic data. By leveraging a structured dataset and integrating machine learning techniques, this project provides a robust, explainable solution.<br />
<br />
__dataset:__
The dataset used here dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.<br />

## Structure of this file
<pre>
|__ Harish_Addada_Heart_Disease_Prediction
|  |__ images
|  |  |__ Output1.png
|  |  |__ Output2.png
|  |  |__ Visualization1.png
|  |  |__ Visualization2.png
|  |  |__ Visualization3.png
|  |  |__ Visualization4.png
|  |__ Heart disease Prediction.ipynb (source code - jupyter notebook file) 
|  |__ README.md
|  |__ app.py (flaskapi)  
|  |__ heart.csv (dataset)
|  |__ heart_model.pkl (pickle file)
|  |__ requirements.txt
</pre> 

## Dataset information

- age : age in years
- sex : 1 = male, 0 = Female
- cp : chest pain type (4 values)
- trestbps : resting blood pressure (in mm Hg on admission to the hospital)
- chol : serum cholestoral in mg/dl
- fbs : fasting blood sugar > 120 mg/dl
- restecg : resting electrocardiographic results (values 0,1,2)
- thalach : maximum heart rate achieved
- exang : exercise induced angina
- oldpeak : ST depression induced by exercise relative to rest
- slope : the slope of the peak exercise ST segment
- ca : number of major vessels (0-3) colored by flourosopy
- thal : 0 = none; 1 = normal; 2 = fixed defect; 3 = reversable defect
- target : 0 = no disease; 1 = heart disease.

## Classification Model Selection
- here i used LogisticRegression, LDA, KNN, RandomForestClassifier
- since this is a small dataset, for KNN and randomforestclassifier i got overfitting
- so then i have 2 options LogisticRegression and LDA
- Both performed well but i got good accuracy score from logistic regression so finalised this model for prediction.

## How it works?
- I created a predictive classification model using python which i deployed using flaskapi with the help of pickle. 
- It runs on http://localhost:5000/predict , after running flaskapi i used postman in which it takes the json data as input and gives predictive output from model.
- You can see the outputs1 and 2 which i added in image folder.


## Prerequisites
- python
- scikit-learn
- these libraries should be installed in the system:
- scikit-learn
- pandas
- numpy
- seaborn
- flask
- matplotlib
- pickle


