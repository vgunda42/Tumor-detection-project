# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the trained model
with open('heart_model.pkl', 'rb') as f:
    data = pickle.load(f)
    clf_lr = data['model']
    scaler = data['scaler']
print(clf_lr)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Validate that all necessary fields are present
        required_fields = ['age', 'sex(0=F,1=M)', 'chestpain(0-3)','RestingBloodPressure','cholestoral(mg/dl)','restingECG(0-2)','MaxHeartRate','exang','oldpeak','slope','No-ofMajorVessele(0-3)','thal(0-2)']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
    
        input_data = {        
            'age': data['age'],                      
            'sex' : data['sex(0=F,1=M)'],                              
            'cp' : data['chestpain(0-3)'],            
            'trestbps': data['RestingBloodPressure'],                      
            'chol' : data['cholestoral(mg/dl)'],                               
            'restecg': data['restingECG(0-2)'],                      
            'thalach' : data['MaxHeartRate'],                              
            'exang' : data['exang'], 
            'oldpeak': data['oldpeak'], 
            'slope' : data['slope'],                     
            'ca' : data['No-ofMajorVessele(0-3)'],                              
            'thal' : data['thal(0-2)']
        }

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        from sklearn.preprocessing import StandardScaler
        # Scale the input data
        input_df_scaled = scaler.transform(input_df)

        # Make prediction using the model
        prediction = clf_lr.predict(input_df_scaled)


        if prediction == 1:
            str = "Having Heart Disease"
        else:
            str = "No Heart Disease"

        # Return the prediction as a JSON response
        return jsonify({"predicted": str})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

