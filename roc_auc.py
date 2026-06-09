import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("preprocessed_diabetes_dataset.csv")

# Features and Target
X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Probability Prediction
y_prob = model.predict_proba(X_test)[:, 1]

# ROC-AUC Score
auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", round(auc, 4))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()