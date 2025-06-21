import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import uuid
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
REPORTS_DIR = "reports"
IMAGES_DIR = "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

ml_model = joblib.load("app/backend/ml_model.pkl")

def read_dynacard(file: UploadFile):
    df = pd.read_excel(file.file)
    df.columns = df.columns.str.strip()
    return df

def calculate_metrics(surf_df, down_df, pump_depth):
    stroke_len = surf_df["Position"].max() - surf_df["Position"].min()
    max_load = surf_df["Load"].max()
    min_load = surf_df["Load"].min()
    fillage = down_df["Load"].idxmin() / len(down_df)
    pump_eff = fillage * 100
    return stroke_len, max_load, min_load, fillage, pump_eff

def ml_predict(stroke_len, max_load, min_load, fillage, pump_depth):
    df = pd.DataFrame([{
        "max_load": max_load,
        "min_load": min_load,
        "stroke_length": stroke_len,
        "pump_fillage": fillage,
        "pump_depth": pump_depth
    }])
    prediction = ml_model.predict(df)[0]
    confidence = max(ml_model.predict_proba(df)[0])
    return prediction, confidence

def generate_plot(df, title, file_path):
    plt.figure()
    plt.plot(df["Position"], df["Load"])
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Load")
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

def create_pdf(report_path, metrics, prediction, confidence, image_paths):
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height - 40, "Rod Pump Dynacard Report")

    c.setFont("Helvetica", 12)
    y = height - 70
    for label, value in metrics.items():
        c.drawString(30, y, f"{label}: {value}")
        y -= 20

    c.drawString(30, y - 10, f"ML Predicted Issue: {prediction} ({confidence*100:.1f}% confidence)")

    for path in image_paths:
        y -= 240
        c.drawImage(path, 30, y, width=500, preserveAspectRatio=True)

    c.save()

@app.post("/api/calculate")
async def calculate(
    spm: int = Form(...),
    rod_weight: float = Form(...),
    pump_depth: float = Form(...),
    fluid_level: float = Form(...),
    rod_string: str = Form(...),
    surface_card_file: UploadFile = File(...),
    downhole_card_file: UploadFile = File(...),
):
    surf_df = read_dynacard(surface_card_file)
    down_df = read_dynacard(downhole_card_file)

    stroke_len, max_load, min_load, fillage, pump_eff = calculate_metrics(surf_df, down_df, pump_depth)
    prediction, confidence = ml_predict(stroke_len, max_load, min_load, fillage, pump_depth)

    uid = str(uuid.uuid4())
    surface_plot_path = os.path.join(IMAGES_DIR, f"surface_{uid}.png")
    downhole_plot_path = os.path.join(IMAGES_DIR, f"downhole_{uid}.png")
    generate_plot(surf_df, "Surface Dynacard", surface_plot_path)
    generate_plot(down_df, "Downhole Dynacard", downhole_plot_path)

    report_path = os.path.join(REPORTS_DIR, f"report_{uid}.pdf")
    metrics = {
        "SPM": spm,
        "Rod Weight (lbs)": rod_weight,
        "Pump Depth (ft)": pump_depth,
        "Fluid Level (ft)": fluid_level,
        "Rod String": rod_string,
        "Effective Stroke Length (in)": round(stroke_len, 2),
        "Pump Fillage": f"{fillage:.2f}",
        "Pump Efficiency (%)": f"{pump_eff:.2f}"
    }
    create_pdf(report_path, metrics, prediction, confidence, [surface_plot_path, downhole_plot_path])

    return {
        "stroke_length": stroke_len,
        "max_load": max_load,
        "min_load": min_load,
        "fillage": fillage,
        "pump_efficiency": pump_eff,
        "ml_prediction": prediction,
        "confidence": confidence,
        "report_path": f"/reports/report_{uid}.pdf",
        "dynocard_image": f"/images/surface_{uid}.png"
    }

@app.get("/reports/{file_name}")
def get_report(file_name: str):
    return FileResponse(path=os.path.join(REPORTS_DIR, file_name))

@app.get("/images/{file_name}")
def get_image(file_name: str):
    return FileResponse(path=os.path.join(IMAGES_DIR, file_name))
