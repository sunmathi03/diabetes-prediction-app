import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ==========================
# LOAD MODEL & SCALER
# ==========================

model = joblib.load("logistic_diabetes_5000.pkl")
scaler = joblib.load("scaler_5000.pkl")

# ==========================
# PATIENT DATA
# ==========================

patient = pd.DataFrame({
    "Age": [45],
    "BMI": [32],
    "Glucose": [170],
    "BloodPressure": [140],
    "ExerciseHours": [2],
    "FamilyHistory": [1],
    "Cholesterol": [240],
    "SleepHours": [6],
    "Smoking": [1],
    "HbA1c": [8.5]
})

# ==========================
# SCALE ONLY NUMERICAL COLUMNS
# ==========================

numerical_cols = [
    "Age",
    "BMI",
    "Glucose",
    "BloodPressure",
    "ExerciseHours",
    "Cholesterol",
    "SleepHours",
    "HbA1c"
]

patient_processed = patient.copy()

patient_processed[numerical_cols] = scaler.transform(
    patient[numerical_cols]
)

# ==========================
# PREDICTION
# ==========================

probability = model.predict_proba(patient_processed)[0][1]

threshold = 0.40

prediction = 1 if probability >= threshold else 0

print("\n===== PREDICTION RESULT =====")

print(f"Probability : {probability*100:.2f}%")
print(f"Threshold   : {threshold}")

if prediction == 1:
    print("Prediction  : Diabetes Risk Detected")
else:
    print("Prediction  : No Diabetes Risk")

# ==========================
# LOAD BACKGROUND DATA
# ==========================

background = pd.read_csv(
    "preprocessed_diabetes_dataset.csv"
)

X_background = background.drop(
    "Diabetes",
    axis=1
)

# ==========================
# SHAP EXPLAINER
# ==========================

explainer = shap.LinearExplainer(
    model,
    X_background
)

shap_values = explainer.shap_values(
    patient_processed
)

# ==========================
# FEATURE CONTRIBUTIONS
# ==========================

print("\n===== TOP FACTORS =====")

feature_names = patient.columns.tolist()

contributions = list(
    zip(
        feature_names,
        shap_values[0]
    )
)

contributions.sort(
    key=lambda x: abs(x[1]),
    reverse=True
)

for feature, value in contributions:

    if value > 0:
        effect = "Increased Risk"
    else:
        effect = "Reduced Risk"

    print(
        f"{feature:15s} "
        f"{value:+.4f} "
        f"({effect})"
    )

# ==========================
# SHAP WATERFALL PLOT
# ==========================

explanation = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=patient_processed.iloc[0].values,
    feature_names=feature_names
)

shap.plots.waterfall(
    explanation,
    show=False
)

plt.savefig(
    "patient_shap_waterfall.png",
    bbox_inches="tight"
)

print("\nSHAP plot saved as patient_shap_waterfall.png")