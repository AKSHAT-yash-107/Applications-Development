import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("logistic_regression_dataset.csv")
df.columns = df.columns.str.strip()

print("\nDataset Preview:")
print(df.head())

# last column assumed as target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# PREDICT
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# PROBABILITY OUTPUT
# -----------------------------
probs = model.predict_proba(X_test)

print("\nPrediction Probabilities (first 5 rows):")
print(probs[:5])
