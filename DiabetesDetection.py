import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb 

st.set_page_config(page_icon="🩸")

# Load the saved Diabetes Detection model
model = pickle.load(open('diabetes.pkl', 'rb'))

# List of features that you used during training
trained_features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Gender labels
gender_labels = {
    0: 'Female',
    1: 'Male'
}

# Smoking history labels
smoking_labels = {
    0: 'No Info',
    1: 'never',
    2: 'former',
    3: 'current',
    4: 'not current',
    5: 'ever'
}

def main():
    st.markdown(
        "<h1 style='text-align: center;'>Diabetes Detection App</h1>",
        unsafe_allow_html=True,
)

    # User input
    age = st.slider("Enter your age:", min_value=1, max_value=100, value=50)
    gender = st.radio("Select your gender:", options=list(gender_labels.values()))
    hypertension = st.radio("Do you have hypertension?", options=["NO", "YES"])
    heart_disease = st.radio("Do you have heart disease?", options=["NO", "YES"])
    smoking_history = st.selectbox("Select smoking history:", options=list(smoking_labels.values()))
    bmi = st.slider("Enter your BMI(Body Mass Index):", min_value=10, max_value=50, value=25)
    HbA1c_level = st.slider("Enter your HbA1c(Hemoglobin A1c) level:", min_value=4.0, max_value=10.0, value=5.7)
    blood_glucose_level = st.slider("Enter your blood glucose level:", min_value=70, max_value=300, value=120)

    # Convert categorical inputs to numerical
    gender_numeric = [key for key, value in gender_labels.items() if value == gender][0]
    smoking_numeric = [key for key, value in smoking_labels.items() if value == smoking_history][0]

    # Numerical Convert
    hypertension_numeric = 1 if hypertension == "YES" else 0
    heart_disease_numeric = 1 if heart_disease == "YES" else 0

    # Create a DataFrame with user inputs
    user_data = pd.DataFrame({
    'gender': [gender_numeric],
    'age': [age],
    'hypertension': [hypertension_numeric],
    'heart_disease': [heart_disease_numeric],
    'smoking_history': [smoking_numeric],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level]
    }, columns=trained_features)

    prediction = model.predict(user_data)

    # Display prediction
# Display prediction with formatted box
    prediction = f"<p style='color: {'red' if prediction[0] == 1 else 'green'}; text-align: center; font-weight: bold; width: 50%; margin: 0 auto; padding: 10px; border: 2px solid {'red' if prediction[0] == 1 else 'green'}; border-radius: 5px;'>Prediction: {'Diabetes Present' if prediction[0] == 1 else 'No Diabetes'}</p>"
    st.markdown(prediction, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
