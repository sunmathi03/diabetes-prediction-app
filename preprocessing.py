import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("advanced_diabetes_dataset_5000.csv")

print("Original Shape:", df.shape)

# Check null values
print("\nNull Values:")
print(df.isnull().sum())

# Check duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values (if any)
imputer = SimpleImputer(strategy='mean')
df[:] = imputer.fit_transform(df)

# Scale numerical columns
numerical_cols = [
    'Age',
    'BMI',
    'Glucose',
    'BloodPressure',
    'ExerciseHours',
    'Cholesterol',
    'SleepHours',
    'HbA1c'
]

scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
joblib.dump(scaler, "scaler_5000.pkl")
# Save cleaned dataset
df.to_csv("preprocessed_diabetes_dataset.csv", index=False)

print("\nPreprocessing Completed!")
print("Final Shape:", df.shape)
print(df.head())
print(df['Diabetes'].value_counts())