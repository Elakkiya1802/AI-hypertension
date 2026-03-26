import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("hypertension.csv")

# --------- ENCODE CATEGORICAL COLUMNS ---------
le = LabelEncoder()

df["BP_History"] = le.fit_transform(df["BP_History"])
df["Medication"] = le.fit_transform(df["Medication"])
df["Exercise_Level"] = le.fit_transform(df["Exercise_Level"])
df["Smoking_Status"] = le.fit_transform(df["Smoking_Status"])

# Convert Yes/No to 1/0
df["Family_History"] = df["Family_History"].map({"Yes": 1, "No": 0})
df["Has_Hypertension"] = df["Has_Hypertension"].map({"Yes": 1, "No": 0})

# --------- FEATURES AND TARGET ---------
X = df.drop("Has_Hypertension", axis=1)
y = df["Has_Hypertension"]

# Feature scaling (important)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
