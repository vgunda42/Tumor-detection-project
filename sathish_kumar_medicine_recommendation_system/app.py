from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle

from src.pipelline.predict_pipeline import helper,get_predicted_value

# flask app
app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')

        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')



# # about view funtion and path
# @app.route('/about')
# def about():
#     return render_template("about.html")
# # contact view funtion and path
# @app.route('/contact')
# def contact():
#     return render_template("contact.html")

# # developer view funtion and path
# @app.route('/developer')
# def developer():
#     return render_template("developer.html")

# # about view funtion and path
# @app.route('/blog')
# def blog():
#     return render_template("blog.html")


if __name__ == '__main__':

    app.run(debug=True)