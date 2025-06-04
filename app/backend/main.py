#File: app/backend/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
from pydantic import BaseModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import zipfile
import os
from fpdf import FPDF
import re
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# --- ML Model Initialization ---
def create_mock_model():
    np.random.seed(42)
    X = pd.DataFrame({
        "load_mean": np.random.uniform(100, 400, 100),
        "load_std": np.random.uniform(10, 100, 100),
        "load_max": np.random.uniform(200, 800, 100),
        "load_min": np.random.uniform(0, 100, 100),
        "position_range": np.random.uniform(50, 120, 100),
        "diff_mean": np.random.uniform(5, 80, 100)
    })
    y = np.random.choice([
        "Gas Interference", "Tubing Leak", "Vibration Interference",
        "Insufficient Inflow", "Flowing with Pumping", "Gas Locking",
        "Traveling Valve Leaking", "Heavy Oil Interference", "Sand Interference"
    ], 100)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

ml_model = create_mock_model()

# --- Column Mapping Logic ---
def normalize_dynocard_df(df):
    col_map = {
        "Surface rod load Position": "Position",
        "Surface rod load": "Load",
        "Downhole pump load position": "Position",
        "Downhole pump load": "Load"
    }
    df = df.rename(columns=col_map)
    if "Position" in df.columns and "Load" in df.columns:
        return df[["Position", "Load"]].dropna()
    else:
        raise ValueError("Dynacard must contain 'Position' and 'Load' columns")

# --- Diagnostics Rules ---
def detect_issue_patterns(df: pd.DataFrame):
    issues = []
    if df["Load"].max() < 200:
        issues.append("Gas Interference")
    if df["Load"].min() > 50:
        issues.append("Tubing Leak")
    if df["Load"].diff().abs().mean() > 100:
        issues.append("Vibration Interference")
    if df["Position"].iloc[-1] < df["Position"].max() * 0.75:
        issues.append("Insufficient Inflow")
    if df["Load"].std() < 20:
        issues.append("Flowing with Pumping")
    if df["Load"].iloc[0] < 50 and df["Load"].iloc[10] > 200:
        issues.append("Gas Locking")
    if df["Load"].rolling(5).mean().diff().abs().max() < 10:
        issues.append("Traveling Valve Leaking")
    if df["Load"].apply(lambda x: x > 500).sum() > 50:
        issues.append("Heavy Oil Interference")
    if df["Load"].diff().min() < -200:
        issues.append("Sand Interference")
    return issues

# --- ML Diagnostics ---
def extract_features(df: pd.DataFrame):
    return pd.DataFrame.from_dict({
        "load_mean": [df["Load"].mean()],
        "load_std": [df["Load"].std()],
        "load_max": [df["Load"].max()],
        "load_min": [df["Load"].min()],
        "position_range": [df["Position"].max() - df["Position"].min()],
        "diff_mean": [df["Load"].diff().abs().mean()],
    })

def ml_predict_issue(df: pd.DataFrame):
    features = extract_features(df)
    preds = ml_model.predict(features)
    probs = ml_model.predict_proba(features)
    return preds[0], float(np.max(probs))

# --- Efficiency & Rod String Analysis ---
def calculate_efficiency_metrics(fillage, stroke_length, spm):
    volumetric_eff = fillage / 100
    system_eff = volumetric_eff * (stroke_length * spm) / 100
    return {"volumetric_eff": volumetric_eff, "system_eff": system_eff}

def parse_rod_string(rod_string):
    rods = re.findall(r"(\d+\.\d+)x(\d+)", rod_string)
    parsed = [(float(d), int(l)) for d, l in rods]
    total_weight = sum(d**2 * l * 0.1 for d, l in parsed)
    return {"rod_config": parsed, "rod_total_weight": total_weight}

# --- Optimizer ---
def suggest_optimization(stroke_length, spm, rod_weight, fillage):
    recommendations = []
    if fillage < 80:
        recommendations.append("Reduce stroke length or spm to avoid fluid pound")
    if rod_weight > 10000:
        recommendations.append("Consider rod changeout to reduce load")
    if stroke_length > 120:
        recommendations.append("Stroke length may be excessive, evaluate shorter stroke")
    return recommendations

# --- Dyno Sim & Metrics ---
def calculate_metrics(stroke_length, spm, rod_weight, pump_depth, fluid_level, rod_string):
    prhp = (rod_weight * stroke_length * spm) / 33000
    fillage = 85.0
    return {"prhp": prhp, "fillage": fillage}

def generate_dyno_card(stroke_length, rod_weight):
    position = np.linspace(0, stroke_length, 100)
    load = rod_weight * np.sin(np.linspace(0, np.pi, 100))
    return pd.DataFrame({"Position": position, "Load": load})

# --- File I/O ---
def parse_excel(file: UploadFile):
    df = pd.read_excel(file.file)
    return normalize_dynocard_df(df)

def generate_csv(data: dict, filename: str):
    df = pd.DataFrame([data])
    path = os.path.join(EXPORT_DIR, filename)
    df.to_csv(path, index=False)
    return path

def generate_dyno_chart(df: pd.DataFrame, filename: str):
    fig, ax = plt.subplots()
    ax.plot(df["Position"], df["Load"])
    ax.set_title("Dyno Card")
    ax.set_xlabel("Position")
    ax.set_ylabel("Load")
    path = os.path.join(EXPORT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path

def generate_pdf(metrics: dict, chart_path: str, issues: list, suggestions: list, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Pump Metrics", ln=True)
    for k, v in metrics.items():
        pdf.cell(200, 10, txt=f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}", ln=True)
    pdf.cell(200, 10, txt="\nDetected Issues:", ln=True)
    for issue in issues:
        pdf.cell(200, 10, txt=f"- {issue}", ln=True)
    pdf.cell(200, 10, txt="\nOptimization Suggestions:", ln=True)
    for s in suggestions:
        pdf.cell(200, 10, txt=f"- {s}", ln=True)
    pdf.image(chart_path, x=10, y=120, w=180)
    path = os.path.join(EXPORT_DIR, filename)
    pdf.output(path)
    return path

# --- API ---
@app.post("/api/calculate")
async def calculate(
    stroke_length: float = Form(...),
    spm: float = Form(...),
    rod_weight: float = Form(...),
    pump_depth: float = Form(...),
    fluid_level: Optional[float] = Form(None),
    rod_string: str = Form(...),
    surface_card_file: UploadFile = File(...),
    downhole_card_file: UploadFile = File(...)
):
    surface_df = parse_excel(surface_card_file)
    downhole_df = parse_excel(downhole_card_file)

    base_metrics = calculate_metrics(stroke_length, spm, rod_weight, pump_depth, fluid_level, rod_string)
    efficiency = calculate_efficiency_metrics(base_metrics["fillage"], stroke_length, spm)
    rod_info = parse_rod_string(rod_string)
    all_metrics = {**base_metrics, **efficiency, **rod_info}

    sim_dyno_df = downhole_df
    issues = detect_issue_patterns(downhole_df)
    ml_issue, ml_conf = ml_predict_issue(downhole_df)
    issues.append(f"ML Suggests: {ml_issue} ({ml_conf * 100:.1f}% confidence)")

    suggestions = suggest_optimization(stroke_length, spm, rod_weight, base_metrics["fillage"])

    generate_csv(all_metrics, "metrics.csv")
    chart_path = generate_dyno_chart(sim_dyno_df, "dyno_chart.png")
    generate_pdf(all_metrics, chart_path, issues, suggestions, "report.pdf")

    return {"metrics": all_metrics, "issues": issues, "suggestions": suggestions}

@app.get("/api/export")
def export():
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zipf:
        for fname in ["metrics.csv", "report.pdf", "dyno_chart.png"]:
            path = os.path.join(EXPORT_DIR, fname)
            zipf.write(path, arcname=fname)
    zip_buf.seek(0)
    zip_buf.seek(0)
    return StreamingResponse(zip_buf, media_type='application/zip', headers={"Content-Disposition": "attachment; filename=report.zip"})
