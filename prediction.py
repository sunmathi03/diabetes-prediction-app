import joblib
import pandas as pd

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example patient
sample = pd.DataFrame({
    "Age":[45],
    "BMI":[32],
    "Glucose":[170],
    "BloodPressure":[140],
    "ExerciseHours":[1],
    "FamilyHistory":[1]
})

# Validation checks
if sample["Age"][0] < 0 or sample["Age"][0] > 120:
    raise ValueError("Invalid Age")

if sample["BMI"][0] < 10 or sample["BMI"][0] > 80:
    raise ValueError("Invalid BMI")

if sample["Glucose"][0] < 40 or sample["Glucose"][0] > 600:
    raise ValueError("Invalid Glucose")

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]

print("Prediction:", "Diabetes" if prediction == 1 else "No Diabetes")
print("Risk Probability:", round(probability * 100, 2), "%")