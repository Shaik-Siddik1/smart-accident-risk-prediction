from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

app = Flask(__name__)

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

DB_FILE = "risk.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            hour INTEGER,
            month INTEGER,
            temperature REAL,
            humidity REAL,
            visibility REAL,
            wind_speed REAL,
            precipitation REAL,
            weather TEXT,
            junction INTEGER,
            crossing INTEGER,
            traffic_signal INTEGER,
            stop INTEGER,
            railway INTEGER,
            risk TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


init_db()


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json()

    hour = int(d["hour"])
    row = {
        "hour": hour,
        "month": int(d["month"]),
        "temperature": float(d["temperature"]),
        "humidity": float(d["humidity"]),
        "visibility": float(d["visibility"]),
        "wind_speed": float(d["wind_speed"]),
        "precipitation": float(d["precipitation"]),
        "weather": d["weather"],
        "is_night": 1 if (hour >= 19 or hour < 6) else 0,
        "junction": int(d["junction"]),
        "crossing": int(d["crossing"]),
        "traffic_signal": int(d["traffic_signal"]),
        "stop": int(d["stop"]),
        "railway": int(d["railway"]),
    }

    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=["weather"])
    df = df.reindex(columns=columns, fill_value=0)

    risk = model.predict(df)[0]
    confidence = round(float(max(model.predict_proba(df)[0])) * 100, 2)

    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """INSERT INTO predictions
           (created_at, hour, month, temperature, humidity, visibility,
            wind_speed, precipitation, weather, junction, crossing,
            traffic_signal, stop, railway, risk, confidence)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), hour, row["month"],
         row["temperature"], row["humidity"], row["visibility"],
         row["wind_speed"], row["precipitation"], row["weather"],
         row["junction"], row["crossing"], row["traffic_signal"],
         row["stop"], row["railway"], risk, confidence)
    )
    conn.commit()
    conn.close()

    return jsonify({"risk": risk, "confidence": confidence})


@app.route("/history", methods=["GET"])
def history():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    app.run(debug=True)