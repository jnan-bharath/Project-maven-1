# model.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ────────────────────────────────────────────────
# 1. Load the data
# ────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel('MACHINE HEALTH & MAINTENANCE (1).xlsx')

# ────────────────────────────────────────────────
# 2. Define features and target
# ────────────────────────────────────────────────
features = [
    'Temperature',
    'Vibration',
    'Pressure',
    'EnergyConsumption',
    'ProductionUnits',
    'Plant',
    'MachineID'
]

target = 'DefectCount'

X = df[features]
y = df[target]

print(f"Dataset shape: {df.shape}")
print(f"Number of unique Plants: {df['Plant'].nunique()}")
print(f"Number of unique MachineIDs: {df['MachineID'].nunique()}")

# ────────────────────────────────────────────────
# 3. Preprocessing pipeline
# ────────────────────────────────────────────────
categorical_features = ['Plant', 'MachineID']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'   # keep numerical columns as they are
)

# ────────────────────────────────────────────────
# 4. Full modeling pipeline
# ────────────────────────────────────────────────
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
print(df['DefectCount'].value_counts().sort_index())
print(df['DefectCount'].describe())

# ────────────────────────────────────────────────
# 5. Train / test split
# ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples     : {len(X_test):,}")

# ────────────────────────────────────────────────
# 6. Train the model
# ────────────────────────────────────────────────
print("Training model...")
model.fit(X_train, y_train)

# ────────────────────────────────────────────────
# 7. Predict on test set
# ────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ────────────────────────────────────────────────
# 8. Evaluate
# ────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50)
print("Regression Results")
print("="*50)
print(f"Mean Absolute Error (MAE)  : {mae:.4f}")
print(f"Root Mean Squared Error    : {rmse:.4f}")
print("="*50)

# Optional: show a few predictions vs actual
print("\nSample of 10 predictions vs actual:")
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
}).reset_index(drop=True)
print(comparison.head(10))
# ────────────────────────────────────────────────
# 8. Evaluate & Print Results
# ────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ────────────────────────────────────────────────
# Beautiful terminal output
# ────────────────────────────────────────────────
print("\n" + "═" * 60)
print("               REGRESSION MODEL PERFORMANCE")
print("═" * 60)
print(f"Mean Absolute Error (MAE)    : {mae:>10.4f}   (average prediction error)")
print(f"Root Mean Squared Error (RMSE): {rmse:>10.4f}   (typical error magnitude)")
print("═" * 60)
print(f"Test set size                : {len(y_test):>10,d} samples")
print(f"Random seed used             : 42")
print("═" * 60)

# Optional: quick interpretation
print("\nInterpretation:")
print("• MAE ≈ 1.38 → on average, predictions are off by about 1.4 defects")
print("• RMSE ≈ 1.75 → larger errors are penalized more")
print("• Most DefectCount values are low (0–5) → these errors are moderate")
print("═" * 60)