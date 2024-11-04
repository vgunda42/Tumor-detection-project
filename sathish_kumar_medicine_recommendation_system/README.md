# Medicine Recommendation System

## Overview
The Medicine Recommendation System is an AI-driven application designed to assist users in identifying potential health conditions based on their reported symptoms. The system provides detailed information about the predicted disease, including its description, precautions, medications, dietary recommendations, and suggested workouts.

## Features
- **Symptom Analysis:** Users can input their symptoms, and the system will predict potential diseases based on the provided information.
- **Disease Information:** For each predicted disease, the system provides:
  - Description of the disease
  - Precautions to be taken
  - Recommended medications
  - Dietary suggestions
  - Suggested workouts

## Technologies Used
- **Python**: The primary programming language for the application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Machine Learning**: Utilized to predict diseases based on symptoms.
- **Flask** or **Streamlit**: For developing the web application interface.

## Data
The application uses several datasets to provide accurate recommendations:
- `symptoms_df.csv`: Contains a list of symptoms and corresponding diseases.
- `precautions_df.csv`: Lists precautions associated with each disease.
- `medications.csv`: Details the medications recommended for each disease.
- `diets.csv`: Offers dietary recommendations based on diseases.
- `workout_df.csv`: Suggests workouts suitable for different health conditions.
- `description.csv`: Provides detailed descriptions of each disease.

Ensure the datasets are located in the `/data` directory as specified in the code.

### Example
1. Input your symptoms in the designated field.
2. Click on the "Predict" button.
3. The system will display the predicted disease along with detailed information, precautions, medications, diets, and workouts.


## Author
**Sathish Kumar**

For any queries or further information, feel free to reach out via email at [2310sathishkumarsk@gmail.com](mailto:2310sathishkumarsk@gmail.com).

