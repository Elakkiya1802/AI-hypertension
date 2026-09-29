from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import pickle
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import itertools
import math

app = FastAPI()

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------- DATABASE ----------
def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS health_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        Age INTEGER,
        Salt_Intake REAL,
        Stress_Score INTEGER,
        BP_History INTEGER,
        Sleep_Duration REAL,
        BMI REAL,
        Medication INTEGER,
        Family_History INTEGER,
        Exercise_Level INTEGER,
        Smoking_Status INTEGER,
        risk TEXT,
        advice TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

create_tables()

# ---------- REQUEST MODELS ----------
class Register(BaseModel):
    email: str
    password: str
    role: str   # "user" or "admin"

class Login(BaseModel):
    email: str
    password: str

class HealthInput(BaseModel):
    Age: int
    Salt_Intake: float
    Stress_Score: int
    BP_History: str
    Sleep_Duration: float
    BMI: float
    Medication: str
    Family_History: str
    Exercise_Level: str
    Smoking_Status: str

# ---------- AUTH HELPERS ----------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_default_admin():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE role='admin'")
    admin_exists = cursor.fetchone()

    if not admin_exists:
        cursor.execute(
            "INSERT INTO users (email, password, role) VALUES (?, ?, ?)",
            ("admin@gmail.com", hash_password("admin123"), "admin")
        )
        conn.commit()

    conn.close()

create_default_admin()

# ---------- LOAD MODEL & SCALER ----------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------- ENCODING (FIXED TO MATCH YOUR DATASET) ----------
def encode_input(data: HealthInput):
    bp_map = {
        "normal": 0,
        "prehypertension": 1,
        "hypertension": 2
    }

    exercise_map = {
        "low": 0,
        "moderate": 1,
        "high": 2
    }

    encoded = {
        "Age": data.Age,
        "Salt_Intake": data.Salt_Intake,
        "Stress_Score": data.Stress_Score,
        "BP_History": bp_map.get(data.BP_History.lower(), 0),
        "Sleep_Duration": data.Sleep_Duration,
        "BMI": data.BMI,
        "Medication": 1 if data.Medication.lower() != "none" else 0,
        "Family_History": 1 if data.Family_History.lower() == "yes" else 0,
        "Exercise_Level": exercise_map.get(data.Exercise_Level.lower(), 1),
        "Smoking_Status": 1 if data.Smoking_Status.lower() != "non-smoker" else 0
    }
    return pd.DataFrame([encoded])

# ---------- EXPLANATION + PERSONALISED ADVICE ----------
FEATURES = [
    "Age", "Salt_Intake", "Stress_Score", "BP_History", "Sleep_Duration",
    "BMI", "Medication", "Family_History", "Exercise_Level", "Smoking_Status",
]

# A "healthy reference person" (encoded exactly like encode_input()).
# Each feature's contribution = how much of the user's risk comes from that feature
# compared with this baseline.
REFERENCE = {
    "Age": 30, "Salt_Intake": 5.0, "Stress_Score": 2, "BP_History": 0,
    "Sleep_Duration": 7.5, "BMI": 22.0, "Medication": 0, "Family_History": 0,
    "Exercise_Level": 2, "Smoking_Status": 0,
}

DRIVER_MIN = 0.05        # a factor must add >= 5 percentage points of risk to be called a driver
MAX_DRIVERS = 4
NON_DRIVERS = {"Medication"}   # being on medication is a marker of treatment, not a cause

# Exact Shapley values need every on/off combination of the 10 features (2^10 = 1024 rows,
# scored in one batch). These tables are computed once at start-up.
_N = len(FEATURES)
_MASKS = np.array(list(itertools.product([0, 1], repeat=_N)), dtype=bool)
_IDX = np.arange(2 ** _N)
_SIZE = _MASKS.sum(axis=1)
_PAIRS = []
for _i in range(_N):
    _bit = 1 << (_N - 1 - _i)
    _lo = _IDX[(_IDX & _bit) == 0]
    _w = np.array([
        math.factorial(s) * math.factorial(_N - s - 1) / math.factorial(_N)
        for s in _SIZE[_lo]
    ])
    _PAIRS.append((_lo, _lo | _bit, _w))


def explain_prediction(df):
    """Return {feature: contribution to probability} versus the healthy reference person."""
    user = df.iloc[0][FEATURES].to_numpy(dtype=float)
    ref = np.array([REFERENCE[f] for f in FEATURES], dtype=float)
    rows = np.where(_MASKS, user, ref)                       # True -> user's value
    batch = pd.DataFrame(rows, columns=FEATURES)
    v = model.predict_proba(scaler.transform(batch))[:, 1]
    return {
        f: float(np.sum(w * (v[hi] - v[lo])))
        for f, (lo, hi, w) in zip(FEATURES, _PAIRS)
    }


def _reason_label(f, d):
    labels = {
        "Age": f"Age ({d.Age}): blood pressure tends to rise with age",
        "Salt_Intake": f"High salt intake ({d.Salt_Intake:g} g/day)",
        "Stress_Score": f"High stress level ({d.Stress_Score}/10)",
        "BP_History": f"Blood pressure history: {d.BP_History.strip().lower()}",
        "Sleep_Duration": f"Short sleep ({d.Sleep_Duration:g} hours/night)",
        "BMI": f"Higher body weight (BMI {d.BMI:.1f})",
        "Family_History": "Family history of hypertension",
        "Exercise_Level": "Low physical activity",
        "Smoking_Status": "Smoking",
    }
    return labels.get(f, f)


def _tips(d):
    """One tip per factor, ONLY when that factor is actually unhealthy for this user."""
    t = {}
    if d.Smoking_Status.lower() != "non-smoker":
        t["Smoking_Status"] = "Quit smoking - it raises blood pressure and strains the heart. Ask your doctor about cessation support."
    if d.Salt_Intake > 6:
        t["Salt_Intake"] = (f"Cut your salt intake from {d.Salt_Intake:g} g/day towards 5-6 g/day: "
                            "limit processed, packaged and pickled foods and avoid extra salt at the table.")
    if d.BMI >= 25:
        t["BMI"] = f"Your BMI is {d.BMI:.1f}. Gradual weight loss (even a few kilos) can noticeably lower blood pressure."
    if d.Sleep_Duration < 7:
        t["Sleep_Duration"] = f"You sleep about {d.Sleep_Duration:g} hours a night. Aim for 7-8 hours on a regular schedule."
    if d.Stress_Score >= 7:
        t["Stress_Score"] = f"Your stress level is high ({d.Stress_Score}/10). Try breathing exercises, meditation, walks or regular breaks."
    if d.Exercise_Level.lower() == "low":
        t["Exercise_Level"] = "Aim for about 30 minutes of moderate activity (e.g. brisk walking) on most days."
    bp = d.BP_History.lower()
    if bp == "hypertension":
        t["BP_History"] = "You have a history of hypertension: monitor your blood pressure regularly and follow up with your doctor."
    elif bp == "prehypertension":
        t["BP_History"] = "Your blood pressure has been in the pre-hypertension range: check it regularly and keep a log."
    if d.Medication.lower() != "none":
        t["Medication"] = "Continue any prescribed medication as directed and do not change doses without your doctor."
    if d.Family_History.lower() == "yes":
        t["Family_History"] = ("Hypertension runs in your family, so have your blood pressure checked at least once a year, "
                               "even when your other factors look good.")
    if d.Age >= 50:
        t["Age"] = "Blood pressure tends to rise with age, so regular check-ups matter more as you get older."
    return t


def build_explanation_and_advice(d, risk, prob, phi):
    """Build (advice_text, reasons_list). Never raises: falls back to simple advice if phi is None."""
    drivers = []
    if phi:
        drivers = sorted(
            [(f, p) for f, p in phi.items() if f not in NON_DRIVERS and p >= DRIVER_MIN],
            key=lambda x: -x[1],
        )[:MAX_DRIVERS]
    total = sum(p for _, p in drivers) or 1.0

    reasons = [
        {"factor": f, "label": _reason_label(f, d), "impact_pct": round(100 * p / total, 1)}
        for f, p in drivers
    ]

    lines = []
    if phi is not None:
        lines.append(f"Why this prediction (estimated risk {prob * 100:.0f}% - {risk}):")
        if reasons:
            for n, r in enumerate(reasons):
                tag = " (strongest factor)" if n == 0 and len(reasons) > 1 else ""
                lines.append(f"• {r['label']}{tag}")
        else:
            lines.append("• None of your inputs stand out as a strong risk factor.")
        lines.append("")

    # advice: factors that drive the prediction first, then any other unhealthy habits
    tips = _tips(d)
    ordered = [f for f, _ in drivers if f in tips] + [f for f in tips if f not in {x for x, _ in drivers}]
    lines.append("Personalised advice:")
    for f in ordered:
        lines.append(f"• {tips[f]}")

    if not ordered:
        lines.append("• Your habits and health indicators look good. Keep it up and check your blood pressure periodically.")
    elif drivers and all(f not in {"Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI", "Smoking_Status", "Exercise_Level"} for f, _ in drivers):
        lines.append("• Your main risk factors cannot be changed, so regular blood pressure checks matter most - "
                     "healthy habits still help keep your risk down.")

    if risk == "High":
        lines.append("• Please consult a doctor for a proper blood pressure check and evaluation.")
    elif risk == "Medium":
        lines.append("• Consider getting your blood pressure checked by a health professional.")
    lines.append("• This is a screening estimate, not a medical diagnosis.")

    return "\n".join(lines), reasons


# ---------- ROUTES ----------
@app.post("/register")
def register(user: Register):
    conn = get_db()
    cursor = conn.cursor()

    if user.role.lower() == "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin registration is not allowed from frontend"
        )

    try:
        cursor.execute(
            "INSERT INTO users (email, password, role) VALUES (?, ?, ?)",
            (user.email, hash_password(user.password), "user")
        )
        conn.commit()

    except sqlite3.IntegrityError:
        # THIS is the real duplicate email error
        raise HTTPException(status_code=400, detail="Email already exists")

    except Exception as e:
        # THIS shows real error
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()

    return {"message": "Registered successfully"}


@app.post("/login")
def login(user: Login):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (user.email,))
    db_user = cursor.fetchone()

    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "message": "Login successful",
        "role": db_user["role"],
        "user_id": db_user["id"]
    }

@app.post("/predict/{user_id}")
def predict(user_id: int, data: HealthInput):
    df = encode_input(data)
    X_scaled = scaler.transform(df)

    prob = model.predict_proba(X_scaled)[0][1]

    if prob < 0.33:
        risk = "Low"
    elif prob < 0.66:
        risk = "Medium"
    else:
        risk = "High"

    # personalised "why" + advice (falls back gracefully so /predict never fails because of it)
    try:
        phi = explain_prediction(df)
    except Exception:
        phi = None
    advice, reasons = build_explanation_and_advice(data, risk, prob, phi)

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO health_data 
        (user_id, Age, Salt_Intake, Stress_Score, BP_History, 
         Sleep_Duration, BMI, Medication, Family_History, 
         Exercise_Level, Smoking_Status, risk, advice)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        data.Age,
        data.Salt_Intake,
        data.Stress_Score,
        encode_input(data)["BP_History"].iloc[0],
        data.Sleep_Duration,
        data.BMI,
        1 if data.Medication.lower() != "none" else 0,
        1 if data.Family_History.lower() == "yes" else 0,
        {"low":0, "moderate":1, "high":2}.get(data.Exercise_Level.lower(), 1),
        1 if data.Smoking_Status.lower() != "non-smoker" else 0,
        risk,
        advice
    ))

    conn.commit()
    conn.close()

    return {"risk": risk, "advice": advice,
            "probability": round(float(prob), 3), "reasons": reasons}

# ---------- USER HISTORY (FIXED: NO MORE 1970) ----------
@app.get("/user/{user_id}/history")
def get_user_history(user_id: int):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, risk, advice, created_at
        FROM health_data
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))

    rows = cursor.fetchall()

    return [
        {
            "prediction_id": r["id"],
            "risk": r["risk"],
            "advice": r["advice"] or "No advice stored",
            "time": r["created_at"] or "No timestamp recorded"
        }
        for r in rows
    ]

# ---------- ADMIN ENDPOINTS ----------
@app.get("/admin/users")
def get_users():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, role FROM users")
    users = cursor.fetchall()
    return [dict(u) for u in users]

@app.get("/admin/stats")
def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT risk, COUNT(*) as count FROM health_data GROUP BY risk")
    stats = cursor.fetchall()
    return [dict(s) for s in stats]

@app.get("/admin/predictions")
def get_predictions_by_date(start: str = None, end: str = None):
    conn = get_db()
    cursor = conn.cursor()

    if start and end:
        cursor.execute("""
            SELECT user_id, risk, created_at
            FROM health_data
            WHERE date(created_at) BETWEEN date(?) AND date(?)
            ORDER BY created_at DESC
        """, (start, end))
    else:
        cursor.execute("""
            SELECT user_id, risk, created_at
            FROM health_data
            ORDER BY created_at DESC
        """)

    rows = cursor.fetchall()
    return [dict(r) for r in rows]

# ---------- DELETE ONE PREDICTION ----------
@app.delete("/user/history/{prediction_id}")
def delete_single_prediction(prediction_id: int):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM health_data WHERE id = ?", (prediction_id,))
    conn.commit()
    conn.close()

    return {"message": f"Prediction {prediction_id} deleted"}

# ---------- CLEAR ALL HISTORY FOR A USER ----------
@app.delete("/user/{user_id}/history")
def delete_user_history(user_id: int):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM health_data WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    return {"message": "All history cleared"}
@app.get("/admin/latest-risks")
def get_latest_risks():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.email, h.risk
        FROM users u
        JOIN health_data h ON h.id = (
            SELECT id FROM health_data
            WHERE user_id = u.id
            ORDER BY created_at DESC
            LIMIT 1
        )
        WHERE u.role = 'user'
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
