import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("fraud.csv")
print(df.head())
X = df.drop("label", axis=1)
y = df["label"]
model = RandomForestClassifier(class_weight="balanced")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(class_weight="balanced", n_estimators=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
new = [[1500, 12, 28, 0, 1, 0]]  # example transaction
new = scaler.transform(new)

print(model.predict(new))
print(model.predict_proba(new))
import matplotlib.pyplot as plt

importances = model.feature_importances_
plt.barh(X.columns, importances)
plt.show()
