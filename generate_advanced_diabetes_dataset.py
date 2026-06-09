import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

# Features
age = np.random.randint(18, 81, n)

bmi = np.round(np.random.normal(28, 6, n), 1)
bmi = np.clip(bmi, 16, 50)

glucose = np.random.randint(70, 251, n)

blood_pressure = np.random.randint(90, 181, n)

exercise_hours = np.round(np.random.uniform(0, 10, n), 1)

family_history = np.random.choice([0, 1], size=n, p=[0.65, 0.35])

cholesterol = np.random.randint(120, 351, n)

sleep_hours = np.round(np.random.uniform(4, 10, n), 1)

smoking = np.random.choice([0, 1], size=n, p=[0.75, 0.25])

hba1c = np.round(np.random.uniform(4.0, 12.0, n), 1)

# Risk score
risk_score = (
    0.03 * age +
    0.08 * bmi +
    0.05 * glucose +
    0.02 * blood_pressure +
    0.25 * family_history +
    0.03 * cholesterol +
    0.10 * smoking +
    0.50 * hba1c -
    0.10 * exercise_hours -
    0.08 * sleep_hours
)

# Add randomness
risk_score += np.random.normal(0, 5, n)

# Convert to target variable
threshold = np.percentile(risk_score, 60)

diabetes = (risk_score > threshold).astype(int)

# Create dataframe
df = pd.DataFrame({
    "Age": age,
    "BMI": bmi,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "ExerciseHours": exercise_hours,
    "FamilyHistory": family_history,
    "Cholesterol": cholesterol,
    "SleepHours": sleep_hours,
    "Smoking": smoking,
    "HbA1c": hba1c,
    "Diabetes": diabetes
})

# Save CSV
df.to_csv("advanced_diabetes_dataset_5000.csv", index=False)

print("Dataset Generated Successfully")
print("Shape:", df.shape)
print(df.head())