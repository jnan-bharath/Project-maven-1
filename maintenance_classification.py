import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier


file_path = "MACHINE HEALTH & MAINTENANCE.xlsx"
df = pd.read_excel(file_path)

df = df.dropna()

target_column = "MaintenanceFlag"

X = df.drop(columns=[target_column])
y = df[target_column]

datetime_cols = X.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns

for col in datetime_cols:
    X[col] = pd.to_datetime(X[col])
    X[col + "_year"] = X[col].dt.year
    X[col + "_month"] = X[col].dt.month
    X[col + "_day"] = X[col].dt.day
    X[col + "_hour"] = X[col].dt.hour
    X = X.drop(columns=[col])

categorical_cols = X.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train_res, y_train_res)

rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("\n----- RANDOM FOREST RESULTS -----")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Precision:", precision_score(y_test, rf_preds))
print("Recall:", recall_score(y_test, rf_preds))
print("F1 Score:", f1_score(y_test, rf_preds))

cb = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    verbose=False,
    class_weights=[1, 2]
)

cb.fit(X_train_res, y_train_res)

cb_preds = cb.predict(X_test)
cb_probs = cb.predict_proba(X_test)[:, 1]

print("\n----- CATBOOST RESULTS -----")
print("Accuracy:", accuracy_score(y_test, cb_preds))
print("Precision:", precision_score(y_test, cb_preds))
print("Recall:", recall_score(y_test, cb_preds))
print("F1 Score:", f1_score(y_test, cb_preds))

future_data = X_test.copy()
future_data["Predicted_Probability"] = cb_probs

top_20_risk = future_data.sort_values(
    by="Predicted_Probability", ascending=False
).head(20)

print("\n----- TOP 20 HIGH-RISK ROWS -----")
print(top_20_risk)

top_20_risk.to_excel("Top_20_High_Risk_Machines.xlsx", index=False)