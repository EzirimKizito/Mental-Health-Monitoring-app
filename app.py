import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load models
model_regress = joblib.load('model_regress.pkl')
model_classif = joblib.load('model_classif.pkl')

# Load all encoders
with open('all_label_encoders.pkl', 'rb') as file:
    loaded_encoders = pickle.load(file)

def transform_input(feature_name, input_value):
    """Transform user input using the corresponding LabelEncoder."""
    encoder = loaded_encoders[feature_name]
    if input_value in encoder.classes_:
        encoded_value = encoder.transform([input_value])[0]
    else:
        st.error(f"Invalid input for {feature_name}. Please enter a valid value from {encoder.classes_}.")
        return None
    return encoded_value

def predict_mental_health_score(input_data):
    """Predict using the regression model with correct feature names."""
    # Create a DataFrame with the same feature names as used in training
    feature_names = model_regress.feature_names_in_  # Assuming the regressor is trained and has this attribute
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model_regress.predict(input_df)
    return prediction[0]

def predict_mental_stability(input_data):
    """Predict using the classification model with correct feature names."""
    # Create a DataFrame with the same feature names as used in training
    feature_names = model_classif.feature_names_in_  # Assuming the classifier is trained and has this attribute
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model_classif.predict(input_df)
    # Fetch the first element of the prediction array
    prediction_label = prediction[0]
    # Define a mapping from model output to descriptive labels
    stability_mapping = {
        0: 'Low Stability',
        1: 'Moderate Stability',
        2: 'High Stability'
    }
    # Fetch the descriptive label from the mapping using the prediction label
    descriptive_label = stability_mapping.get(prediction_label, "Unknown Stability")
    return descriptive_label


def main():
    st.title("Mental Health Prediction System")

    # Sidebar for navigation
    app_mode = st.sidebar.selectbox(
        "Choose the prediction mode:",
        ["Mental Health Score Prediction", "Mental Stability Prediction"]
    )

    # User input section
    st.subheader("User Input Parameters")
    input_data = []

    # Define features and options for user input
    features = {
        'Gender': ['Male', 'Female'],
        'Preferred form of social interaction': ['text messages', 'social apps', 'phone calls'],
        'Satisfaction from social interaction': (1, 5),
        'Feelings of loneliness and isolation': ['Yes', 'Maybe', 'No', "I'm not sure"],
        'Availability of support system': (1, 5),
        'Barriers to feeling connected': ['Yes', 'No'],
        'Cordiality in relationships': (1, 5),
        'Preferred method of seeking support': ['text messages', 'face to face conversations', 'social apps', 'via online messages'],
        'Reluctance to seek support': (1, 5),
        'Major conflicts in relationships': ['Yes', 'No'],
        'Impact of conflict on mental health': ['Made me want to isolate more', 'Made me more defensive', 'Made me indifferent', 'Made me more assertive'],
        'Frequency of tests/exams': ['Very Frequent', 'Not Frequent', 'Frequent'],
        'Time for activities outside academics': ['Not at all. Academic work keeps me fully occupied.', 'Not really but I try to make out the time', 'Yes, I have ample time'],
        'Negative effects due to academic commitments': ['Somewhat', 'Yes', 'No'],
        'Hours of sleep per night': (0, 15),
        'Disruption in sleep due to stress': ['Yes', 'I think so', 'No'],
        'Stress level about academics': (1, 5),
        'Feeling supported during stress': ['Yes', 'No'],
        'Favorite way of de-stressing': ['Sleeping', 'Eating', 'Taking a stroll', 'Playing games', 'Watching movies'],
        'Body response to academic stress': ['Nothing changes', 'My physique deteriorates'],
        'Comfort in seeking help when overwhelmed': ['No', 'Yes'],
        'Awareness of mental health resources': ['No', 'Yes'],
        'Use of mental health services': ['No', 'Yes'],
        'Alcohol consumption frequency': ['Not often', 'Never', 'Once in a while'],
        'Bottles of alcohol per session': (0, 10),
        'Alcohol consumption during academic sessions': ['High', 'Very low'],
        'Experimentation with recreational drugs': ['No', 'Yes'],
        'Frequency of recreational drug use': ['Rarely', 'Never', 'Not often'],
        'Introduction to substances': ['Academic stress/frustrations', 'Friends', 'Other', 'Social media'],
        'Substance use behaviors in family/social circles': ['Yes', 'No'],
        'Risk perception of substance use': (0, 5),
        'Negative effects from substance use': ['Yes', 'No'],
        'Substance use as a stress cope': ['No', 'Yes'],
        'Changes in academics due to substance use': ['Yes', 'No'],
        'Seeking help for substance issues': ['Yes', 'No'],
        'Worry about financial situation': ['Very often', 'Often', 'Rarely', 'Not often'],
        'Stress level about financial stability': (1, 5),
        'Struggles with basic needs due to finances': ['Yes', 'No'],
        'Financial situation compared to last year': ['Stayed the same', 'Improved', 'Worsened'],
        'Confidence in managing finances': (1, 5),
        'Borrowing money due to financial difficulties': ['No', 'Yes'],
        'Having a savings plan': ['Yes', 'No'],
        'Impact of financial situation on daily life': ['Makes me very cautious', 'Keeps me on my toes', 'Gives me freedom of choice', 'Helps me plan better'],
        'Feelings of shame related to financial status': ['Yes', 'No'],
        'Satisfaction with income and financial stability': (1, 5),
        'Health symptoms from financial stress': ['Yes', 'No'],
        'Awareness of financial support resources': ['No', 'Yes']
    }

    # Generate UI elements for each feature and collect user input
    for feature, options in features.items():
        if isinstance(options, tuple):  # If the options is a tuple, it represents a range
            input_data.append(st.number_input(f'{feature} (Range: {options[0]}-{options[1]})', min_value=options[0], max_value=options[1]))
        elif isinstance(options, list):
            selected_option = st.selectbox(f'Select {feature}', options)
            encoded_value = transform_input(feature, selected_option)
            input_data.append(encoded_value)
        else:
            st.error("Error in feature configuration.")

    if st.button("Predict"):
        if None not in input_data:  # Ensure no invalid inputs
            if app_mode == "Mental Health Score Prediction":
                prediction = predict_mental_health_score(input_data)
                st.success(f"The predicted mental health score is {prediction:.2f}")
            elif app_mode == "Mental Stability Prediction":
                descriptive_label = predict_mental_stability(input_data)
                st.success(f"The predicted mental stability category is: {descriptive_label}")

if __name__ == "__main__":
    main()
