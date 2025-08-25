import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Paths
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "model/churn_model.pkl"

# Load dataset
data = pd.read_csv(DATA_PATH)

# Drop customerID (not useful for prediction)
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Handle categorical variables
data = pd.get_dummies(data)

# Separate features and target
X = data.drop("Churn_Yes", axis=1)   # Churn has Yes/No → One-hot encoded
y = data["Churn_Yes"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model trained and saved at {MODEL_PATH}")
