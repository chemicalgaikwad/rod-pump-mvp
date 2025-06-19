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
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Additional calculations
def calculate_effective_stroke(stroke_length, fillage):
    return stroke_length * fillage

def recommend_optimized_stroke(stroke_length, fillage):
    if fillage < 0.85:
        return stroke_length * fillage  # Reduce stroke if fillage low
    return stroke_length

def recommend_optimized_spm(spm, fillage, target_fillage=0.85):
    return spm * (target_fillage / fillage) if fillage != 0 else spm

def calculate_statistical_metrics(df):
    load = df["Load"]
    return {
        "load_skewness": skew(load),
        "load_kurtosis": kurtosis(load),
        "load_stddev": load.std(),
        "load_mean": load.mean()
    }

# Dummy realistic function implementations
async def parse_excel(file):
    contents = await file.read()
    return pd.read_excel(io.BytesIO(contents))

def parse_rod_string(rod_string):
    pattern = r"(\d+\.\d+)x(\d+)"
    matches = re.findall(pattern, rod_string)
    total_weight = 0
    for diameter, count in matches:
        diameter = float(diameter)
        count = int(count)
        weight_per_ft = (diameter ** 2) * 0.785 * 490  # Approximate
        total_weight += weight_per_ft * count * 30  # 30ft assumed length
    return {"rod_total_weight": total_weight}

def calculate_metrics(spm, rod_weight, pump_depth, fluid_level, rod_string, surface_df, downhole_df, plunger_diameter=1.5, fluid_sg=0.8):
    stroke_length = surface_df["Position"].max()
    fillage = min(fluid_level / pump_depth, 1.0) if fluid_level else 1.0
    prhp = 144 * 0.433 * fluid_sg * fluid_level  # psi
    volumetric_eff = fillage
    system_eff = volumetric_eff * 0.8
    return {
        "stroke_length": stroke_length,
        "fillage": fillage,
        "prhp": prhp,
        "volumetric_eff": volumetric_eff,
        "system_eff": system_eff,
    }

def calculate_efficiency_metrics(fillage):
    pump_efficiency = fillage * 100
    return {"pump_efficiency": pump_efficiency}

def generate_csv(metrics, filename):
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(EXPORT_DIR, filename), index=False)

def generate_dyno_chart_combined(surface_df, downhole_df, path, fluid_load, max_fluid_load):
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(surface_df["Position"], surface_df["Load"], label="Surface")
    plt.axhline(y=fluid_load, color='r', linestyle='--', label='Fluid Load')
    plt.axhline(y=max_fluid_load, color='g', linestyle='--', label='Max Fluid Load')
    plt.title("Surface Dynacard")
    plt.grid(True)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(downhole_df["Position"], downhole_df["Load"], label="Downhole", color='orange')
    plt.axhline(y=fluid_load, color='r', linestyle='--', label='Fluid Load')
    plt.axhline(y=max_fluid_load, color='g', linestyle='--', label='Max Fluid Load')
    plt.title("Downhole Dynacard")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_DIR, path))
    return os.path.join(EXPORT_DIR, path)

def detect_issue_patterns(df):
    load = df["Load"]
    issues = []
    if load.max() < 5000: issues.append("Insufficient Inflow")
    if load.min() < -2000: issues.append("Gas Locking")
    if load.std() > 3000: issues.append("Sand Interference")
    return issues

def ml_predict_issue(df):
    model = joblib.load("app/backend/ml_model.pkl")
    features = [df["Load"].mean(), df["Load"].std()]
    pred = model.predict([features])[0]
    prob = max(model.predict_proba([features])[0])
    return str(pred), float(prob)

def suggest_optimization(stroke_length, spm, rod_weight, fillage):
    suggestions = []
    if fillage < 0.8:
        suggestions.append("Increase SPM or reduce stroke.")
    elif fillage > 0.95:
        suggestions.append("Possible energy waste. Reduce SPM.")
    return suggestions, stroke_length * fillage

# The rest remains unchanged as per your code including generate_pdf etc.

# Your FastAPI route and pdf generation already handle this output.
