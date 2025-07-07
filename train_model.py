import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load the features from the CSV file
df = pd.read_csv("features_dataset.csv")

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df)

# Save the trained model
joblib.dump(model, "eye_anomaly_model.pkl")
print("✅ Model trained and saved as eye_anomaly_model.pkl")
