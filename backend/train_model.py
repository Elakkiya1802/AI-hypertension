import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# keep_default_na=False is IMPORTANT: the text "None" in the Medication column is a
# real category ("no medication"), but pandas would otherwise read it as a missing value.
df = pd.read_csv("hypertension.csv", keep_default_na=False)

# --------- ENCODE CATEGORICAL COLUMNS ---------
# These mappings are IDENTICAL to encode_input() in app.py.
# (The old script used LabelEncoder, which numbers categories alphabetically, e.g.
#  Hypertension=0, Normal=1 - while the API sent Normal=0. That mismatch made the
#  model misread users' inputs and produced the "family history dominates" behaviour.)
df["BP_History"] = df["BP_History"].str.lower().map(
    {"normal": 0, "prehypertension": 1, "hypertension": 2}
)
df["Medication"] = (df["Medication"].str.lower() != "none").astype(int)
df["Family_History"] = (df["Family_History"].str.lower() == "yes").astype(int)
df["Exercise_Level"] = df["Exercise_Level"].str.lower().map(
    {"low": 0, "moderate": 1, "high": 2}
)
df["Smoking_Status"] = (df["Smoking_Status"].str.lower() != "non-smoker").astype(int)
df["Has_Hypertension"] = (df["Has_Hypertension"].str.lower() == "yes").astype(int)

assert not df.isna().any().any(), "Unmapped/missing values found after encoding"

# --------- FEATURES AND TARGET ---------
X = df.drop("Has_Hypertension", axis=1)
y = df["Has_Hypertension"]

# Feature scaling (kept so app.py's scaler.transform() flow is unchanged)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,   # smoother, less "all-or-nothing" probabilities
    random_state=42,
)
model.fit(X_train, y_train)

# Accuracy
print("Test accuracy      :", round(model.score(X_test, y_test), 4))
cv = cross_val_score(
    RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42),
    X_scaled, y, cv=5, scoring="roc_auc",
)
print("5-fold CV ROC-AUC  :", round(cv.mean(), 4))

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
