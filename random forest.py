import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = "MACHINE HEALTH & MAINTENANCE.xlsx"
df = pd.read_excel(file_path)

le_plant = LabelEncoder()
le_machine = LabelEncoder()

df['Plant'] = le_plant.fit_transform(df['Plant'])
df['MachineID'] = le_machine.fit_transform(df['MachineID'])

X = df[['Temperature',
        'Vibration',
        'Pressure',
        'EnergyConsumption',
        'ProductionUnits',
        'Plant',
        'MachineID']]

y = df['DefectCount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE  :", round(mae, 4))
print("RMSE :", round(rmse, 4))
print("R2   :", round(r2, 4))

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n")
print(importance)
