import joblib

model = joblib.load("logistic_diabetes_5000.pkl")

print(model.n_features_in_)

try:
    print(model.feature_names_in_)
except:
    print("No feature names stored")