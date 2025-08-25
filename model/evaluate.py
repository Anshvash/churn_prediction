import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID (not useful for prediction)
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Encode categorical variables
data = pd.get_dummies(data)

# Features and target
X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

# Train-test split (same as in train.py)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the saved model
model = joblib.load("model/churn_model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy:.2f}")

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n🔄 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
