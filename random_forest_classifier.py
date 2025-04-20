import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load example fraud detection dataset (replace with your dataset)
# Expected format: features like transaction_amount, time_of_day, etc., and a label column 'is_fraud'
data = pd.read_csv("fraud_data.csv")
X = data.drop(columns=["is_fraud"])
y = data["is_fraud"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("Feature Importance:\n", feature_importance)

# Save the model
joblib.dump(model, "random_forest_fraud_classifier.pkl")
print("Model saved as random_forest_fraud_classifier.pkl")
