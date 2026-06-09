import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("preprocessed_diabetes_dataset.csv")

# Features and target
X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Probability prediction
y_prob = model.predict_proba(X_test)[:, 1]

# Try multiple thresholds
thresholds = [0.50, 0.45, 0.40, 0.35, 0.30]

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    print("\n" + "="*50)
    print(f"Threshold = {threshold}")

    print("Accuracy :", round(accuracy_score(y_test, y_pred),4))
    print("Precision:", round(precision_score(y_test, y_pred),4))
    print("Recall   :", round(recall_score(y_test, y_pred),4))
    print("F1 Score :", round(f1_score(y_test, y_pred),4))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))