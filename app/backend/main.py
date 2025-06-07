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

# --- Dyno Metrics ---
def calculate_metrics(spm, rod_weight, pump_depth, fluid_level, rod_string, surface_df, downhole_df):
    normalized_range = surface_df["Position"].max() - surface_df["Position"].min()
    # Infer actual stroke length dynamically from metadata (assuming Position is normalized 0–1 or 0–100%)
    # Common convention: normalized range of 1 → 100% stroke
    # We'll scale it to 75 inches if normalized range is close to 1
    if 0.9 <= normalized_range <= 1.1:
        stroke_length = 75
    elif 99 <= normalized_range <= 101:
        stroke_length = 75
        surface_df["Position"] /= 100
        downhole_df["Position"] /= 100
    else:
        stroke_length = normalized_range  # Assume position was in inches already

    surface_df["Position"] *= stroke_length / normalized_range
    downhole_df["Position"] *= stroke_length / normalized_range
    prhp = (rod_weight * stroke_length * spm) / 33000
    load_range = downhole_df["Load"].max() - downhole_df["Load"].min()
    fillage = min((load_range / rod_weight) * 100, 100) if rod_weight else 0
    fluid_load = downhole_df["Load"].mean()
    max_fluid_load = downhole_df["Load"].max()
    return {"stroke_length": stroke_length, "prhp": prhp, "fillage": fillage, "fluid_load": fluid_load, "max_fluid_load": max_fluid_load}

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

    # Surface chart
    axs[0].plot(surface_df["Position"], surface_df["Load"], color='blue')
    axs[0].set_title("Surface Dyno Card")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Load")
    axs[0].grid(True)

    # Downhole chart with fluid load lines
    axs[1].plot(downhole_df["Position"], downhole_df["Load"], color='orange')
    axs[1].set_title("Downhole Dyno Card")
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Load")
    axs[1].grid(True)

    # Calculate and annotate fluid load
    fluid_load = downhole_df["Load"].mean()
    max_fluid_load = downhole_df["Load"].max()
    axs[1].axhline(fluid_load, color='green', linestyle='--', label=f"Fluid Load: {fluid_load:.1f}")
    axs[1].axhline(max_fluid_load, color='red', linestyle='--', label=f"Max Fluid Load: {max_fluid_load:.1f}")
    axs[1].legend()

    plt.tight_layout()
    path = os.path.join(EXPORT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path

def generate_pdf(metrics: dict, chart_paths: list, issues: list, suggestions: list, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Section: Pump Metrics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Pump Metrics", ln=True)
    pdf.set_font("Arial", size=12)
    for k, v in metrics.items():
        pdf.cell(200, 10, txt=f"{k}: {v:.2f}%" if k in ['volumetric_eff'] else (f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}"), ln=True)

    # Section: Detected Issues
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Detected Pump Issues", ln=True)
    pdf.set_font("Arial", size=12)
    if issues:
        for issue in issues:
            pdf.cell(200, 10, txt=f"- {issue}", ln=True)
    else:
        pdf.cell(200, 10, txt="None detected", ln=True)

    # Section: Optimization Suggestions
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Optimization Suggestions", ln=True)
    pdf.set_font("Arial", size=12)
    if suggestions:
        for s in suggestions:
            pdf.cell(200, 10, txt=f"- {s}", ln=True)
    else:
        pdf.cell(200, 10, txt="None recommended", ln=True)

    # Section: Diagnosis Reasoning
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Diagnosis Reasoning", ln=True)
    pdf.set_font("Arial", size=12)
    for issue in issues:
        if "Insufficient Inflow" in issue:
            pdf.cell(200, 10, txt="- Insufficient Inflow: Pump stroke not fully utilized, likely due to low reservoir inflow.", ln=True)
        elif "Gas Locking" in issue:
            pdf.cell(200, 10, txt="- Gas Locking: Load rise pattern suggests trapped gas preventing full fluid entry.", ln=True)
        elif "Heavy Oil Interference" in issue:
            pdf.cell(200, 10, txt="- Heavy Oil Interference: High sustained load implies viscous resistance.", ln=True)
        elif "Sand Interference" in issue:
            pdf.cell(200, 10, txt="- Sand Interference: Sudden sharp downstroke drops indicate plunger drag or sand.", ln=True)
        elif "Gas Interference" in issue:
            pdf.cell(200, 10, txt="- Gas Interference: Load never rises high enough, may be mostly gas pumped.", ln=True)
        elif "Traveling Valve Leaking" in issue:
            pdf.cell(200, 10, txt="- Traveling Valve Leaking: Minimal change in load during cycle.", ln=True)
        elif "Tubing Leak" in issue:
            pdf.cell(200, 10, txt="- Tubing Leak: Low stroke efficiency, high minimum load.", ln=True)
        elif "Vibration Interference" in issue:
            pdf.cell(200, 10, txt="- Vibration Interference: Load variation unusually high, may indicate rod vibration.", ln=True)
        elif "Flowing with Pumping" in issue:
            pdf.cell(200, 10, txt="- Flowing with Pumping: Load profile flat and consistent, flow through pump.", ln=True)
        elif "ML Suggests" in issue:
            pdf.cell(200, 10, txt=f"- {issue}", ln=True)
    for i, chart_path in enumerate(chart_paths):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        title = os.path.splitext(os.path.basename(chart_path))[0].replace('_', ' ').title()
        pdf.cell(200, 10, txt=title, ln=True)
        pdf.image(chart_path, x=10, y=30, w=180)

    path = os.path.join(EXPORT_DIR, filename)
    pdf.output(path)
    return path

# --- API ---
@app.post("/api/calculate")
async def calculate(
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

    base_metrics = calculate_metrics(spm, rod_weight, pump_depth, fluid_level, rod_string, surface_df, downhole_df)
    efficiency = calculate_efficiency_metrics(base_metrics["fillage"])
    rod_info = parse_rod_string(rod_string)
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
