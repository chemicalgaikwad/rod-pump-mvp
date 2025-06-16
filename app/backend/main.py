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
from math import pi
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

# --- Dyno Metrics ---
def calculate_metrics(spm, rod_weight, pump_depth, fluid_level, rod_string, surface_df, downhole_df, plunger_diameter=1.5, fluid_sg=0.85):
    normalized_range = surface_df["Position"].max() - surface_df["Position"].min()
    if 0.9 <= normalized_range <= 1.1:
        stroke_length = 75
    elif 99 <= normalized_range <= 101:
        stroke_length = 75
        surface_df["Position"] /= 100
        downhole_df["Position"] /= 100
    else:
        stroke_length = normalized_range

    surface_df["Position"] *= stroke_length / normalized_range
    downhole_df["Position"] *= stroke_length / normalized_range
    prhp = (rod_weight * stroke_length * spm) / 33000
    load_range = downhole_df["Load"].max() - downhole_df["Load"].min()
    fillage = min((load_range / rod_weight) * 100, 100) if rod_weight else 0
    fluid_load = downhole_df["Load"].mean()
    max_fluid_load = downhole_df["Load"].max()
    plunger_area = pi * (plunger_diameter / 2) ** 2
    effective_stroke = stroke_length * fillage / 100
    pump_displacement = plunger_area * effective_stroke * spm
    fluid_load_calc = plunger_area * fluid_sg * 62.4 * pump_depth
    return {"stroke_length": stroke_length, "prhp": prhp, "fillage": fillage, "fluid_load": fluid_load, "max_fluid_load": max_fluid_load, "pump_displacement": pump_displacement, "fluid_load_calc": fluid_load_calc}

# --- Efficiency & Rod String Analysis ---
def calculate_efficiency_metrics(fillage):
    return {"volumetric_eff": fillage}

def parse_rod_string(rod_string):
    rods = re.findall(r"(\d+\.\d+)x(\d+)", rod_string)
    parsed = [(float(d), int(l)) for d, l in rods]
    total_weight = sum(d**2 * l * 0.1 for d, l in parsed)
    return {"rod_config": parsed, "rod_total_weight": total_weight}

# --- Optimization ---
def suggest_optimization(stroke_length, spm, rod_weight, fillage):
    recommendations = []
    if fillage < 80:
        recommendations.append("Reduce stroke length or spm to avoid fluid pound")
    if rod_weight > 10000:
        recommendations.append("Consider rod changeout to reduce load")
    if stroke_length > 120:
        recommendations.append("Stroke length may be excessive, evaluate shorter stroke")
    recommended_stroke = max(min(120, stroke_length * (fillage / 100)), 60)
    return recommendations, recommended_stroke

# --- File I/O ---
def parse_excel(file: UploadFile):
    df = pd.read_excel(file.file)
    return normalize_dynocard_df(df)

def generate_csv(data: dict, filename: str):
    df = pd.DataFrame([data])
    path = os.path.join(EXPORT_DIR, filename)
    df.to_csv(path, index=False)
    return path

def generate_dyno_chart_combined(surface_df: pd.DataFrame, downhole_df: pd.DataFrame, filename: str):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    axs[0].plot(surface_df["Position"], surface_df["Load"], color='blue')
    axs[0].set_title("Surface Dyno Card")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Load")
    axs[0].grid(True)
    axs[1].plot(downhole_df["Position"], downhole_df["Load"], color='orange')
    axs[1].set_title("Downhole Dyno Card")
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Load")
    axs[1].grid(True)
    plt.tight_layout()
    path = os.path.join(EXPORT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path

def generate_pdf(metrics: dict, chart_paths: list, issues: list, suggestions: list, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for k, v in metrics.items():
        pdf.cell(200, 10, txt=f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}", ln=True)
    pdf.ln(5)
    for issue in issues:
        pdf.cell(200, 10, txt=f"Issue: {issue}", ln=True)
    pdf.ln(5)
    for s in suggestions:
        pdf.cell(200, 10, txt=f"Suggestion: {s}", ln=True)
    for path in chart_paths:
        pdf.add_page()
        pdf.image(path, x=10, y=30, w=180)
    pdf.output(os.path.join(EXPORT_DIR, filename))

@app.post("/api/calculate")
async def calculate(
    spm: float = Form(...),
    pump_depth: float = Form(...),
    fluid_level: Optional[float] = Form(None),
    rod_string: str = Form(...),
    plunger_diameter: float = Form(1.5),
    fluid_sg: float = Form(0.85),
    surface_card_file: UploadFile = File(...),
    downhole_card_file: UploadFile = File(...)
):
    surface_df = parse_excel(surface_card_file)
    downhole_df = parse_excel(downhole_card_file)
    rod_info = parse_rod_string(rod_string)
    rod_weight = rod_info["rod_total_weight"]
    base_metrics = calculate_metrics(spm, rod_weight, pump_depth, fluid_level, rod_string, surface_df, downhole_df, plunger_diameter, fluid_sg)
    efficiency = calculate_efficiency_metrics(base_metrics["fillage"])
    suggestions, recommended_stroke = suggest_optimization(base_metrics["stroke_length"], spm, rod_weight, base_metrics["fillage"])
    all_metrics = {**base_metrics, **efficiency, **rod_info, "recommended_stroke_length": recommended_stroke}
    issues = detect_issue_patterns(downhole_df)
    ml_issue, ml_conf = ml_predict_issue(downhole_df)
    issues.append(f"ML Suggests: {ml_issue} ({ml_conf * 100:.1f}% confidence)")
    generate_csv(all_metrics, "metrics.csv")
    combined_chart_path = generate_dyno_chart_combined(surface_df, downhole_df, "combined_dyno_chart.png")
    generate_pdf(all_metrics, [combined_chart_path], issues, suggestions, "report.pdf")
    return {"metrics": all_metrics, "issues": issues, "suggestions": suggestions}

@app.get("/api/export")
def export():
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zipf:
        for fname in ["metrics.csv", "report.pdf", "combined_dyno_chart.png"]:
            path = os.path.join(EXPORT_DIR, fname)
            zipf.write(path, arcname=fname)
    zip_buf.seek(0)
    return StreamingResponse(zip_buf, media_type='application/zip', headers={"Content-Disposition": "attachment; filename=report.zip"})
