import streamlit as st
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
data = pd.read_csv('Medicalpremium.csv')

# Define features and target variable
X = data.drop('PremiumPrice', axis=1)  # Features (all columns except target)
y = data['PremiumPrice']  # Target variable

# Feature scaling for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Streamlit Web App
st.title("Medical Insurance Premium Predictor")

# Input fields for user to enter data
age = st.number_input('Age', min_value=18, max_value=100, value=30)
diabetes = st.checkbox('Diabetes')
blood_pressure = st.checkbox('Blood Pressure Problems')
transplants = st.checkbox('Any Transplants')
chronic_diseases = st.checkbox('Any Chronic Diseases')
height = st.number_input('Height (in cm)', min_value=100, max_value=250, value=170)
weight = st.number_input('Weight (in kg)', min_value=30, max_value=200, value=70)
allergies = st.checkbox('Known Allergies')
cancer_history = st.checkbox('History of Cancer in Family')
surgeries = st.number_input('Number of Major Surgeries', min_value=0, max_value=10, value=0)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Diabetes': [1 if diabetes else 0],
    'BloodPressureProblems': [1 if blood_pressure else 0],
    'AnyTransplants': [1 if transplants else 0],
    'AnyChronicDiseases': [1 if chronic_diseases else 0],
    'Height': [height],
    'Weight': [weight],
    'KnownAllergies': [1 if allergies else 0],
    'HistoryOfCancerInFamily': [1 if cancer_history else 0],
    'NumberOfMajorSurgeries': [surgeries]
})

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Button to trigger prediction and start processing
if st.button('Predict Premium Price'):
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define objective function for Optuna
    def objective(trial):
        try:
            # Hyperparameter tuning
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 4, 16),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
            model = RandomForestRegressor(**params, random_state=42)
            
            # Cross-validation for robust performance evaluation
            cv_score = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
            return cv_score

        except Exception as e:
            print(f"Trial failed: {e}")
            return -np.inf  # Return a very low score if there's an error

    # Create and run the Optuna study
    study = optuna.create_study(direction='maximize')

    # Early stopping and reducing the number of trials for faster optimization
    study.optimize(objective, n_trials=20, timeout=600, n_jobs=-1)

    # Get the best parameters from the study
    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # Make prediction using the best model
    premium_price = best_model.predict(input_data_scaled)[0]
    st.header(f"Estimated Premium Price: Rs. {premium_price:.2f}")

    # Display best parameters in Streamlit
    # st.write(f"Best Parameters: {best_params}")
