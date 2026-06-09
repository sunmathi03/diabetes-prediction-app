import pandas as pd

df = pd.read_csv("preprocessed_diabetes_dataset.csv")

print("Columns:")
print(df.columns.tolist())

print("\nNumber of Features:")
print(len(df.columns) - 1)