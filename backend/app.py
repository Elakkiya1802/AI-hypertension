from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import pickle
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

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
        advice = (
            "• Maintain current diet and exercise routine.\n"
            "• Keep salt intake under 6g/day.\n"
            "• Sleep at least 7 hours daily."
        )
    elif prob < 0.66:
        risk = "Medium"
        advice = (
            "• Reduce salt intake.\n"
            "• Add 30 minutes of exercise daily.\n"
            "• Monitor blood pressure weekly."
        )
    else:
        risk = "High"
        advice = (
            "• Strictly limit salt intake.\n"
            "• Avoid smoking and alcohol.\n"
            "• Improve sleep schedule.\n"
            "• Consult a doctor immediately."
        )

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

    return {"risk": risk, "advice": advice}

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
