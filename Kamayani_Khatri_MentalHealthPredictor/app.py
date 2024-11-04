import streamlit as st
import pandas as pd
import joblib

# Loading trained model
model = joblib.load('mental_health_model.pkl')

# App title
st.title("Employee Mental Health Predictor")

# Input fields for user data
age = st.number_input("Age", min_value=18, max_value=100, value=30)
years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
hours_worked_per_week = st.number_input("Hours Worked Per Week", min_value=0, max_value=168, value=40)
number_of_virtual_meetings = st.number_input("Number of Virtual Meetings", min_value=0, max_value=20, value=5)
satisfaction_with_remote_work = st.selectbox("Satisfaction with Remote Work (1-3)", [1, 2, 3])
work_life_balance_rating = st.selectbox("Work-Life Balance Rating (1-5)", [1, 2, 3, 4, 5])
company_support = st.selectbox("Company Support for Remote Work (1-5)", [1, 2, 3, 4, 5])
stress_level = st.selectbox("Stress Level (0-4)", [0, 1, 2, 3, 4])
social_isolation_rating = st.selectbox("Social Isolation Rating (0-2)", [0, 1, 2])

# Gender input
gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
gender_male = 1 if gender == "Male" else 0
gender_non_binary = 1 if gender == "Non-binary" else 0
gender_prefer_not_to_say = 1 if gender == "Prefer not to say" else 0

# Job role input
job_role = st.selectbox("Job Role", ["Designer", "HR", "Marketing", "Project Manager", "Sales", "Software Engineer"])
job_role_designer = 1 if job_role == "Designer" else 0
job_role_hr = 1 if job_role == "HR" else 0
job_role_marketing = 1 if job_role == "Marketing" else 0
job_role_project_manager = 1 if job_role == "Project Manager" else 0
job_role_sales = 1 if job_role == "Sales" else 0
job_role_software_engineer = 1 if job_role == "Software Engineer" else 0

# Industry input
industry = st.selectbox("Industry", ["Education", "Finance", "Healthcare", "IT", "Manufacturing", "Retail"])
industry_education = 1 if industry == "Education" else 0
industry_finance = 1 if industry == "Finance" else 0
industry_healthcare = 1 if industry == "Healthcare" else 0
industry_it = 1 if industry == "IT" else 0
industry_manufacturing = 1 if industry == "Manufacturing" else 0
industry_retail = 1 if industry == "Retail" else 0

# Work location input 
work_location = st.selectbox("Work Location", ["Onsite", "Remote"])
work_location_onsite = 1 if work_location == "Onsite" else 0
work_location_remote = 1 if work_location == "Remote" else 0

# Access to mental health resources
access_to_mental_health_resources = st.selectbox("Access to Mental Health Resources (0: No, 1: Yes)", [0, 1])

# Productivity change
productivity_change = st.selectbox("Productivity Change", ["Increase", "No Change"])
productivity_change_increase = 1 if productivity_change == "Increase" else 0
productivity_change_no_change = 0 if productivity_change_increase == 1 else 1

# Sleep quality
sleep_quality = st.selectbox("Sleep Quality", ["Good", "Poor"])
sleep_quality_good = 1 if sleep_quality == "Good" else 0
sleep_quality_poor = 0 if sleep_quality_good == 1 else 1

# Physical Activity input
physical_activity = st.selectbox("Physical Activity", ["Daily", "Weekly"])
physical_activity_daily = 1 if physical_activity == "Daily" else 0
physical_activity_weekly = 1 if physical_activity == "Weekly" else 0

# Region input
region = st.selectbox("Region", ["Asia", "Europe", "North America", "Oceania", "South America"])
region_asia = 1 if region == "Asia" else 0
region_europe = 1 if region == "Europe" else 0
region_north_america = 1 if region == "North America" else 0
region_oceania = 1 if region == "Oceania" else 0
region_south_america = 1 if region == "South America" else 0

# Preparing input DataFrame based on model's expected columns in the correct order
input_data = pd.DataFrame({
    'Age': [age],
    'Years_of_Experience': [years_of_experience],
    'Hours_Worked_Per_Week': [hours_worked_per_week],
    'Number_of_Virtual_Meetings': [number_of_virtual_meetings],
    'Work_Life_Balance_Rating': [work_life_balance_rating],
    'Stress_Level': [stress_level],
    'Social_Isolation_Rating': [social_isolation_rating],
    'Satisfaction_with_Remote_Work': [satisfaction_with_remote_work],
    'Company_Support_for_Remote_Work': [company_support],
    'Physical_Activity_Daily': [physical_activity_daily],
    'Physical_Activity_Weekly': [physical_activity_weekly],
    'Gender_Male': [gender_male],
    'Gender_Non-binary': [gender_non_binary],
    'Gender_Prefer not to say': [gender_prefer_not_to_say],
    'Job_Role_Designer': [job_role_designer],
    'Job_Role_HR': [job_role_hr],
    'Job_Role_Marketing': [job_role_marketing],
    'Job_Role_Project Manager': [job_role_project_manager],
    'Job_Role_Sales': [job_role_sales],
    'Job_Role_Software Engineer': [job_role_software_engineer],
    'Industry_Education': [industry_education],
    'Industry_Finance': [industry_finance],
    'Industry_Healthcare': [industry_healthcare],
    'Industry_IT': [industry_it],
    'Industry_Manufacturing': [industry_manufacturing],
    'Industry_Retail': [industry_retail],
    'Work_Location_Onsite': [work_location_onsite],
    'Work_Location_Remote': [work_location_remote],
    'Access_to_Mental_Health_Resources_Yes': [access_to_mental_health_resources],
    'Productivity_Change_Increase': [productivity_change_increase],
    'Productivity_Change_No Change': [productivity_change_no_change],
    'Sleep_Quality_Good': [sleep_quality_good],
    'Sleep_Quality_Poor': [sleep_quality_poor],
    'Region_Asia': [region_asia],
    'Region_Europe': [region_europe],
    'Region_North America': [region_north_america],
    'Region_Oceania': [region_oceania],
    'Region_South America': [region_south_america],
})

# Ensuring input data matches the training column order
input_data = input_data[['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 
                          'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating', 
                          'Stress_Level', 'Social_Isolation_Rating', 
                          'Satisfaction_with_Remote_Work', 
                          'Company_Support_for_Remote_Work', 
                          'Physical_Activity_Daily', 'Physical_Activity_Weekly', 
                          'Gender_Male', 'Gender_Non-binary', 
                          'Gender_Prefer not to say', 'Job_Role_Designer', 
                          'Job_Role_HR', 'Job_Role_Marketing', 
                          'Job_Role_Project Manager', 
                          'Job_Role_Sales', 'Job_Role_Software Engineer', 
                          'Industry_Education', 'Industry_Finance', 
                          'Industry_Healthcare', 'Industry_IT', 
                          'Industry_Manufacturing', 'Industry_Retail', 
                          'Work_Location_Onsite', 'Work_Location_Remote', 
                          'Access_to_Mental_Health_Resources_Yes', 
                          'Productivity_Change_Increase', 
                          'Productivity_Change_No Change', 
                          'Sleep_Quality_Good', 'Sleep_Quality_Poor', 
                          'Region_Asia', 'Region_Europe', 
                          'Region_North America', 'Region_Oceania', 
                          'Region_South America']]

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Displaying the results
    if prediction[0] == 0:
        st.success("Predicted Mental Health Status: No Issues - Keep it up!! :) ")
    else:
        st.error("Predicted Mental Health Status: Issues Detected - Provide Help!!")

  
