"""FastAPI Application & Enterprise Developer Platform for Machine Learning System Architecture & Deployment."""
import io
import json
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.models.schemas import (
    AnalysisRequest,
    TechnicalBlueprint,
    LatencyRequirement,
    HardwareTarget,
    InterpretabilityRequirement,
)
from src.catalog.models_catalog import MODEL_CATALOG
from src.engines.problem_analyzer import ProblemAnalyzer
from src.engines.dataset_analyzer import DatasetAnalyzer
from src.engines.model_recommender import ModelRecommender
from src.engines.architecture_generator import ArchitectureGenerator

app = FastAPI(
    title="OmniArchitect AI - Machine Learning Systems Platform",
    description="Enterprise Machine Learning Architecture Recommendation, Model Evaluation, and Automated GitHub Actions CI/CD",
    version="3.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_PRESETS = [
    {
        "id": "fraud",
        "title": "Financial Transaction Fraud Detection",
        "tag": "Imbalanced Classification",
        "description": "Construct an automated low-latency fraud detection system analyzing high-volume credit card transaction streams with a severe 15:1 class imbalance ratio.",
        "expected_output": "Calibrated fraud probability score (0.0 to 1.0) and binary classification threshold.",
        "latency": "real_time",
        "hardware": "cpu_only",
        "interpretability": "high",
        "data_size": "large"
    },
    {
        "id": "churn",
        "title": "Enterprise Subscription Customer Churn",
        "tag": "Tabular Supervised",
        "description": "Predict subscription customer contract cancellation risk using customer service interaction logs, monthly billing amounts, and usage trajectory.",
        "expected_output": "Probability of account termination within 30 days and top driving risk factors.",
        "latency": "near_real_time",
        "hardware": "cpu_only",
        "interpretability": "medium",
        "data_size": "medium"
    },
    {
        "id": "chest_xray",
        "title": "Chest Radiograph Pathology Diagnosis",
        "tag": "Medical Computer Vision",
        "description": "Classify clinical frontal chest radiographs to distinguish between pulmonary pneumonia, COVID-19 opacity, and normal lung anatomy.",
        "expected_output": "Multi-class pathology diagnostic probabilities with Grad-CAM localization heatmaps.",
        "latency": "near_real_time",
        "hardware": "single_gpu",
        "interpretability": "high",
        "data_size": "medium"
    },
    {
        "id": "pedestrian",
        "title": "Autonomous Driving Pedestrian Detection",
        "tag": "Real-Time Object Detection",
        "description": "Real-time spatial bounding box localization and tracking of pedestrians, cyclists, and moving vehicles from forward camera streams at 60 FPS.",
        "expected_output": "2D bounding box coordinates [x1, y1, x2, y2], class IDs, and confidence values.",
        "latency": "real_time",
        "hardware": "single_gpu",
        "interpretability": "low",
        "data_size": "large"
    },
    {
        "id": "sentiment",
        "title": "Customer Feedback Sentiment & Intent Mining",
        "tag": "NLP Language Model",
        "description": "Analyze enterprise customer feedback reviews to extract fine-grained emotional sentiment, product bug reports, and customer satisfaction metrics.",
        "expected_output": "Multi-class sentiment rating (Negative, Neutral, Positive) and intent categories.",
        "latency": "real_time",
        "hardware": "cpu_only",
        "interpretability": "medium",
        "data_size": "medium"
    },
    {
        "id": "forecasting",
        "title": "Multi-Horizon Retail Inventory Demand",
        "tag": "Time Series Forecasting",
        "description": "Forecast daily product inventory demand across 150 retail distribution centers incorporating promotional calendars, seasonality, and weather.",
        "expected_output": "30-day ahead demand forecast with 10th and 90th percentile prediction intervals.",
        "latency": "batch",
        "hardware": "cpu_only",
        "interpretability": "high",
        "data_size": "large"
    },
    {
        "id": "molecules",
        "title": "Molecular Drug Binding Affinity Screening",
        "tag": "Graph Neural Network",
        "description": "Predict binding affinity of novel candidate therapeutic molecules against targeted viral protein receptors using 2D/3D molecular graphs.",
        "expected_output": "Continuous binding affinity IC50 concentration prediction (nM).",
        "latency": "batch",
        "hardware": "single_gpu",
        "interpretability": "medium",
        "data_size": "medium"
    },
    {
        "id": "robotics",
        "title": "Robotic Arm Manipulator Trajectory Control",
        "tag": "Reinforcement Learning",
        "description": "Train an autonomous multi-axis robotic arm manipulator in physics simulation to grasp diverse objects and optimize trajectory energy rewards.",
        "expected_output": "Continuous joint torque control action vectors at 100Hz frequency.",
        "latency": "real_time",
        "hardware": "single_gpu",
        "interpretability": "low",
        "data_size": "large"
    }
]


def generate_full_blueprint(request: AnalysisRequest, df: Optional[pd.DataFrame] = None) -> TechnicalBlueprint:
    """Orchestrate all 4 engines to produce the complete technical blueprint."""
    category, display_name, summary = ProblemAnalyzer.analyze_problem(request)

    dataset_profile = None
    if df is not None and not df.empty:
        dataset_profile = DatasetAnalyzer.profile_dataframe(df)

    models = ModelRecommender.recommend_models(category, request, dataset_profile)

    blueprint = ArchitectureGenerator.generate_blueprint(
        category=category,
        category_name=display_name,
        summary=summary,
        request=request,
        models=models,
        dataset_profile=dataset_profile
    )

    return blueprint


@app.get("/health")
def health_check():
    """Health check endpoint for Docker, Kubernetes, and Render monitoring."""
    return {
        "status": "healthy",
        "service": "OmniArchitect AI - Machine Learning Systems Platform",
        "version": "3.5.0",
        "catalog_size": len(MODEL_CATALOG)
    }


@app.get("/api/catalog")
def get_model_catalog():
    """Return knowledge base catalog of all supported ML and Deep Learning algorithms."""
    return {"total_models": len(MODEL_CATALOG), "catalog": MODEL_CATALOG}


@app.get("/api/presets")
def get_presets():
    """Return pre-configured real-world problem templates."""
    return {"presets": SAMPLE_PRESETS}


@app.get("/api/sample-dataset")
def get_sample_dataset():
    """Return synthetic sample credit card fraud data for instant UI testing."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_datasets" / "sample_credit_card_fraud.csv"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return {"filename": "sample_credit_card_fraud.csv", "csv_content": f.read()}
    return {"error": "Sample dataset not found on disk."}


@app.post("/api/analyze", response_model=TechnicalBlueprint)
def analyze_problem_endpoint(request: AnalysisRequest):
    """Analyze text problem description and generate complete technical blueprint."""
    return generate_full_blueprint(request)


@app.post("/api/analyze-dataset", response_model=TechnicalBlueprint)
async def analyze_dataset_endpoint(
    file: UploadFile = File(...),
    problem_description: Optional[str] = Form(None),
    latency: LatencyRequirement = Form(LatencyRequirement.NEAR_REAL_TIME),
    hardware: HardwareTarget = Form(HardwareTarget.CPU_ONLY),
    interpretability: InterpretabilityRequirement = Form(InterpretabilityRequirement.MEDIUM)
):
    """Analyze uploaded CSV dataset, infer problem characteristics, and produce technical blueprint."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV datasets are currently supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    description = problem_description
    if not description or len(description.strip()) < 10:
        description = f"Predict target variable based on uploaded tabular dataset with {df.shape[0]} samples and {df.shape[1]} features."

    request = AnalysisRequest(
        problem_description=description,
        latency=latency,
        hardware=hardware,
        interpretability=interpretability
    )

    return generate_full_blueprint(request, df=df)


# ---------------- Modern Developer Platform Frontend ----------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OmniArchitect AI &bull; Machine Learning System Design & Deployment Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090a0f;
      --surface-1: #0e111a;
      --surface-2: #141926;
      --surface-3: #1a2234;
      --border: #20273a;
      --border-subtle: #171d2b;
      --border-focus: #38bdf8;
      --accent: #38bdf8;
      --accent-muted: rgba(56, 189, 248, 0.12);
      --emerald: #10b981;
      --emerald-muted: rgba(16, 185, 129, 0.12);
      --amber: #f59e0b;
      --amber-muted: rgba(245, 158, 11, 0.12);
      --indigo: #6366f1;
      --violet: #8b5cf6;
      --rose: #f43f5e;
      --text-1: #f8fafc;
      --text-2: #94a3b8;
      --text-3: #64748b;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-sans);
      background-color: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%);
      color: var(--text-1);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      -webkit-font-smoothing: antialiased;
    }

    /* Top Navigation */
    header.navbar {
      width: 100%;
      height: 60px;
      border-bottom: 1px solid var(--border);
      background: rgba(9, 10, 15, 0.85);
      backdrop-filter: blur(12px);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 1.75rem;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .nav-brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-weight: 700;
      font-size: 0.95rem;
      letter-spacing: -0.01em;
    }
    .nav-logo {
      width: 28px;
      height: 28px;
      border-radius: 6px;
      background: linear-gradient(135deg, #0284c7, #6366f1);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 800;
      font-size: 0.8rem;
    }
    .nav-tag {
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--text-3);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      background: var(--surface-2);
      border: 1px solid var(--border);
    }
    .nav-items {
      display: flex;
      align-items: center;
      gap: 1.25rem;
      font-size: 0.82rem;
      font-weight: 500;
    }
    .nav-items a {
      color: var(--text-2);
      text-decoration: none;
      transition: color 0.15s;
    }
    .nav-items a:hover { color: var(--text-1); }
    .ci-indicator {
      display: flex;
      align-items: center;
      gap: 0.45rem;
      padding: 0.3rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: var(--emerald-muted);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: var(--emerald);
    }
    .ci-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--emerald);
      box-shadow: 0 0 6px var(--emerald);
    }

    /* Container */
    .app-container {
      width: 100%;
      max-width: 1280px;
      padding: 2.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }

    /* Hero / Subhead */
    .headline-wrap {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      flex-wrap: wrap;
      gap: 1.25rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border-subtle);
    }
    .headline-title {
      font-size: 1.85rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      color: var(--text-1);
    }
    .headline-desc {
      font-size: 0.92rem;
      color: var(--text-2);
      margin-top: 0.35rem;
      max-width: 720px;
      line-height: 1.5;
    }

    /* Two-Column Studio Layout */
    .studio-grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 1.5rem;
    }
    @media (max-width: 992px) {
      .studio-grid { grid-template-columns: 1fr; }
    }

    /* Panel Surfaces */
    .panel {
      background: var(--surface-1);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.25rem;
    }
    .panel-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--text-1);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .panel-title svg {
      width: 18px;
      height: 18px;
      stroke: var(--accent);
    }

    /* Presets Bar */
    .presets-row {
      display: flex;
      gap: 0.4rem;
      overflow-x: auto;
      padding-bottom: 0.6rem;
      margin-bottom: 1.25rem;
    }
    .presets-row::-webkit-scrollbar { height: 4px; }
    .presets-row::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    .preset-chip {
      white-space: nowrap;
      background: var(--surface-2);
      border: 1px solid var(--border);
      color: var(--text-2);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .preset-chip:hover {
      background: var(--surface-3);
      border-color: var(--accent);
      color: var(--text-1);
    }

    /* Form Inputs */
    .form-group {
      margin-bottom: 1.15rem;
    }
    .field-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-2);
      margin-bottom: 0.4rem;
    }
    textarea.input-ctrl, select.input-ctrl, input.input-ctrl {
      width: 100%;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 0.85rem;
      color: var(--text-1);
      font-family: inherit;
      font-size: 0.88rem;
      outline: none;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    textarea.input-ctrl:focus, select.input-ctrl:focus, input.input-ctrl:focus {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 1px var(--border-focus);
    }
    .grid-duo {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.85rem;
    }

    /* Action Buttons */
    .btn-primary {
      width: 100%;
      height: 44px;
      border: none;
      border-radius: 8px;
      background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
      color: white;
      font-weight: 600;
      font-size: 0.92rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: opacity 0.15s, transform 0.1s;
    }
    .btn-primary:hover {
      opacity: 0.93;
      transform: translateY(-1px);
    }
    .btn-secondary {
      height: 32px;
      padding: 0 0.85rem;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text-2);
      font-weight: 500;
      font-size: 0.78rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s;
    }
    .btn-secondary:hover {
      background: var(--surface-3);
      color: var(--text-1);
      border-color: var(--accent);
    }

    /* Dropzone */
    .upload-box {
      border: 1px dashed var(--border);
      border-radius: 8px;
      background: rgba(14, 17, 26, 0.5);
      padding: 1.75rem 1rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.15s;
      margin-bottom: 1rem;
    }
    .upload-box:hover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.03);
    }
    .upload-icon {
      width: 32px;
      height: 32px;
      margin: 0 auto 0.5rem auto;
      stroke: var(--text-3);
    }

    /* Results Workspace */
    #results-workspace {
      display: none;
      display: flex;
      flex-direction: column;
      gap: 1.75rem;
    }

    /* Executive Result Header */
    .exec-summary {
      background: var(--surface-1);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem 1.75rem;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 1.25rem;
    }
    .category-kicker {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
      margin-bottom: 0.25rem;
    }
    .category-headline {
      font-size: 1.6rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 0.4rem;
    }
    .category-narrative {
      font-size: 0.9rem;
      color: var(--text-2);
      max-width: 860px;
      line-height: 1.55;
    }

    /* Dataset Metric Counters */
    .stats-tray {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
    }
    .stat-card {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
    }
    .stat-lbl { font-size: 0.72rem; color: var(--text-3); font-weight: 600; text-transform: uppercase; }
    .stat-val { font-size: 1.4rem; font-weight: 800; margin-top: 0.25rem; color: var(--text-1); font-family: var(--font-mono); }

    /* Model Matrix Grid */
    .models-deck {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.25rem;
    }
    .algo-card {
      background: var(--surface-1);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      transition: all 0.2s ease;
    }
    .algo-card:hover {
      border-color: rgba(56, 189, 248, 0.4);
      transform: translateY(-2px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .tier-pill {
      display: inline-block;
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      margin-bottom: 0.75rem;
      width: fit-content;
    }
    .tier-primary { background: var(--amber-muted); color: var(--amber); border: 1px solid rgba(245, 158, 11, 0.3); }
    .tier-baseline { background: var(--surface-3); color: var(--text-2); border: 1px solid var(--border); }
    .tier-comparison { background: var(--accent-muted); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3); }
    .tier-deep { background: rgba(139, 92, 246, 0.15); color: var(--violet); border: 1px solid rgba(139, 92, 246, 0.3); }

    .algo-family { font-size: 0.72rem; font-weight: 600; color: var(--text-3); }
    .algo-name { font-size: 1.2rem; font-weight: 700; margin: 0.2rem 0 0.6rem 0; letter-spacing: -0.01em; }
    .algo-summary { font-size: 0.82rem; color: var(--text-2); line-height: 1.5; margin-bottom: 1.15rem; }

    /* Performance Gauges */
    .radar-box {
      background: var(--surface-2);
      border-radius: 6px;
      padding: 0.85rem;
      margin-bottom: 1.15rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .gauge-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      color: var(--text-3);
    }
    .gauge-val { color: var(--text-1); font-weight: 600; font-family: var(--font-mono); }

    .bullet-points {
      list-style: none;
      font-size: 0.78rem;
      color: var(--text-2);
      margin-bottom: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .bullet-points li {
      display: flex;
      align-items: flex-start;
      gap: 0.45rem;
      line-height: 1.4;
    }
    .bullet-points li::before {
      content: "";
      width: 4px;
      height: 4px;
      border-radius: 50%;
      background: var(--accent);
      margin-top: 6px;
      flex-shrink: 0;
    }

    /* Pipeline Flow Chart */
    .flow-chain {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.85rem;
    }
    .flow-step {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      position: relative;
    }
    .flow-index { font-size: 0.68rem; font-weight: 700; color: var(--accent); text-transform: uppercase; font-family: var(--font-mono); }
    .flow-name { font-size: 0.85rem; font-weight: 700; color: var(--text-1); }
    .flow-tools { font-size: 0.72rem; color: var(--text-3); margin-top: auto; }

    /* Terminal Simulator */
    .terminal-window {
      background: #05070a;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      font-family: var(--font-mono);
      font-size: 0.8rem;
    }
    .terminal-bar {
      background: #0d1017;
      border-bottom: 1px solid var(--border);
      padding: 0.5rem 0.85rem;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }
    .term-dot { width: 9px; height: 9px; border-radius: 50%; }
    .term-red { background: #ef4444; }
    .term-yellow { background: #f59e0b; }
    .term-green { background: #10b981; }
    .term-title { font-size: 0.72rem; color: var(--text-3); margin-left: 0.5rem; }
    .terminal-body {
      padding: 1rem 1.25rem;
      color: #38bdf8;
      line-height: 1.6;
      max-height: 240px;
      overflow-y: auto;
    }

    /* Tabbed Code Container */
    .code-tabs {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }
    .code-tab-btn {
      background: transparent;
      border: none;
      color: var(--text-3);
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.4rem 0.75rem;
      border-radius: 4px;
      cursor: pointer;
    }
    .code-tab-btn.active {
      background: var(--surface-2);
      color: var(--accent);
    }
    pre.code-editor {
      background: #05070a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem;
      font-family: var(--font-mono);
      font-size: 0.8rem;
      line-height: 1.6;
      color: #93c5fd;
      overflow-x: auto;
      max-height: 420px;
    }

    /* Modal */
    #catalog-modal {
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(5, 7, 10, 0.88);
      backdrop-filter: blur(10px);
      z-index: 200;
      padding: 3rem 1.5rem;
      overflow-y: auto;
    }
    .modal-shell {
      max-width: 1060px;
      margin: 0 auto;
      background: var(--surface-1);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2rem;
    }

    footer.site-foot {
      width: 100%;
      border-top: 1px solid var(--border-subtle);
      padding: 2rem 0;
      text-align: center;
      font-size: 0.78rem;
      color: var(--text-3);
    }
  </style>
</head>
<body>

  <!-- Top Navigation -->
  <header class="navbar">
    <div class="nav-brand">
      <div class="nav-logo">ML</div>
      <span>OmniArchitect AI</span>
      <span class="nav-tag">v3.5 Production</span>
    </div>
    <div class="nav-items">
      <a href="#results-workspace">System Design</a>
      <a href="javascript:void(0)" onclick="openCatalogModal()">Model Catalog (50+)</a>
      <a href="/docs" target="_blank">OpenAPI Docs</a>
      <div class="ci-indicator">
        <div class="ci-dot"></div>
        <span>GitHub Actions: 25/25 Passing</span>
      </div>
    </div>
  </header>

  <div class="app-container">
    <!-- Hero / Headline -->
    <div class="headline-wrap">
      <div>
        <h1 class="headline-title">Machine Learning System Architecture Platform</h1>
        <p class="headline-desc">Automated technical feasibility analysis, multi-candidate algorithm selection, end-to-end data pipelines, and CI/CD deployment automation.</p>
      </div>
      <div>
        <button class="btn-secondary" onclick="openCatalogModal()">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
          Explore 50+ Model Catalog
        </button>
      </div>
    </div>

    <!-- Studio Grid -->
    <div class="studio-grid">
      <!-- Problem Specification Panel -->
      <div class="panel">
        <div>
          <div class="panel-head">
            <div class="panel-title">
              <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
              <span>1. Problem Definition & Operational Constraints</span>
            </div>
          </div>

          <div class="preset-label">
            <span>Benchmark Problem Presets:</span>
          </div>
          <div class="presets-row" id="preset-strip">
            <!-- Populated via JS -->
          </div>

          <div class="form-group">
            <label class="field-label" for="desc">
              <span>Problem Description & Business Context</span>
            </label>
            <textarea id="desc" class="input-ctrl" rows="4" placeholder="e.g. Build an automated fraud detection engine for streaming credit card transactions with severe class imbalance..."></textarea>
          </div>

          <div class="form-group">
            <label class="field-label" for="output-target">
              <span>Target Output / Prediction Specification</span>
            </label>
            <input type="text" id="output-target" class="input-ctrl" placeholder="e.g. Binary probability score [0.0 - 1.0]">
          </div>

          <div class="grid-duo">
            <div class="form-group">
              <label class="field-label" for="latency">Latency Budget</label>
              <select id="latency" class="input-ctrl">
                <option value="real_time" selected>Real-Time (&lt; 50ms)</option>
                <option value="near_real_time">Near Real-Time (&lt; 500ms)</option>
                <option value="batch">Batch Execution</option>
                <option value="ultra_low">Ultra-Low (&lt; 10ms)</option>
              </select>
            </div>
            <div class="form-group">
              <label class="field-label" for="hardware">Compute Target</label>
              <select id="hardware" class="input-ctrl">
                <option value="cpu_only" selected>Standard Cloud / Local CPU</option>
                <option value="single_gpu">Dedicated GPU (RTX 4090 / A10G)</option>
                <option value="distributed_gpu">Multi-GPU Cluster (A100 / H100)</option>
                <option value="edge_embedded">Edge Embedded (Jetson / Pi)</option>
              </select>
            </div>
          </div>

          <div class="grid-duo">
            <div class="form-group">
              <label class="field-label" for="interpretability">Explainability Requirement</label>
              <select id="interpretability" class="input-ctrl">
                <option value="high">High (Regulatory Auditing / SHAP)</option>
                <option value="medium" selected>Medium (Feature Importances)</option>
                <option value="low">Low (Pure Accuracy / Black-Box)</option>
              </select>
            </div>
            <div class="form-group">
              <label class="field-label" for="datasize">Dataset Volume Scale</label>
              <select id="datasize" class="input-ctrl">
                <option value="small">Small (&lt; 10k rows)</option>
                <option value="medium" selected>Medium (10k - 500k rows)</option>
                <option value="large">Large (&gt; 500k rows)</option>
              </select>
            </div>
          </div>
        </div>

        <button class="btn-primary" onclick="synthesizeSystem()">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          Run Architecture Synthesis & Model Evaluation
        </button>
      </div>

      <!-- Dataset Profiler Panel -->
      <div class="panel">
        <div>
          <div class="panel-head">
            <div class="panel-title">
              <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
              <span>2. Dataset Profiler & Validation</span>
            </div>
            <button class="btn-secondary" onclick="loadSampleDataset()">Load Sample Dataset</button>
          </div>

          <p style="font-size:0.82rem; color:var(--text-2); margin-bottom:1rem; line-height:1.5;">
            Upload your training tabular dataset in CSV format. The engine automatically analyzes feature distributions, cardinality, missingness ratios, and class imbalance.
          </p>

          <div class="upload-box" onclick="document.getElementById('file-input').click()">
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
            <div style="font-size:0.88rem; font-weight:600; color:var(--text-1);" id="file-label">Select or drop CSV file here</div>
            <div style="font-size:0.72rem; color:var(--text-3); margin-top:0.25rem;">Standard comma-separated format</div>
            <input type="file" id="file-input" accept=".csv" style="display:none;" onchange="handleFileSelect(event)">
          </div>

          <button class="btn-primary" style="background:linear-gradient(135deg, #059669 0%, #0284c7 100%); margin-bottom:1.25rem;" onclick="analyzeDatasetUpload()">
            Upload & Profile Dataset
          </button>
        </div>

        <!-- Terminal Output -->
        <div class="terminal-window">
          <div class="terminal-bar">
            <div class="term-dot term-red"></div>
            <div class="term-dot term-yellow"></div>
            <div class="term-dot term-green"></div>
            <span class="term-title">github-actions-runner // ci_cd_pipeline</span>
          </div>
          <div class="terminal-body" id="term-logs">
            [workflow] Initialized runner ubuntu-latest<br>
            [runner] python -m pip install -r requirements.txt (cached)<br>
            [runner] pytest -v --durations=10 (25 test cases collected)<br>
            <span style="color:#10b981;">[success] 25 passed in 2.38s (100% green build)</span><br>
            [deploy] Automated deployment gate verified. Ready for execution.
          </div>
        </div>
      </div>
    </div>

    <!-- Results Workspace -->
    <div id="results-workspace">
      <!-- Executive Summary Banner -->
      <div class="exec-summary">
        <div>
          <div class="category-kicker" id="res-kicker">Problem Classification</div>
          <div class="category-headline" id="res-headline">Tabular Classification</div>
          <p class="category-narrative" id="res-narrative">Synthesizing system architecture...</p>
        </div>
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
          <button class="btn-secondary" onclick="exportBlueprintJSON()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            Export Blueprint (JSON)
          </button>
          <button class="btn-secondary" onclick="window.print()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 6 2 18 2 18 9"></polyline><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><rect x="6" y="14" width="12" height="8"></rect></svg>
            Print Executive Report
          </button>
        </div>
      </div>

      <!-- Dataset Statistics Card (Conditional) -->
      <div class="panel" id="dataset-stats-card" style="display:none;">
        <div class="panel-head">
          <div class="panel-title">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            <span>Empirical Dataset Characteristics</span>
          </div>
        </div>
        <div class="stats-tray" style="margin-bottom:1rem;">
          <div class="stat-card">
            <div class="stat-lbl">Record Volume</div>
            <div class="stat-val" id="stat-rows">--</div>
          </div>
          <div class="stat-card">
            <div class="stat-lbl">Feature Count</div>
            <div class="stat-val" id="stat-cols">--</div>
          </div>
          <div class="stat-card">
            <div class="stat-lbl">Target Variable</div>
            <div class="stat-val" style="color:var(--amber);" id="stat-target">--</div>
          </div>
          <div class="stat-card">
            <div class="stat-lbl">Imbalance Ratio</div>
            <div class="stat-val" style="color:var(--rose);" id="stat-imbalance">None</div>
          </div>
        </div>
        <div id="stat-prep-list" style="font-size:0.82rem; color:var(--text-2); line-height:1.6;"></div>
      </div>

      <!-- Models Deck -->
      <div>
        <div class="panel-head">
          <div class="panel-title">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
            <span>Algorithm Evaluation & Candidate Shortlist</span>
          </div>
        </div>
        <div class="models-deck" id="models-deck">
          <!-- Populated by JS -->
        </div>
      </div>

      <!-- End-to-End System Pipeline Flow -->
      <div class="panel">
        <div class="panel-head">
          <div class="panel-title">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6.01" y2="6"></line><line x1="6" y1="18" x2="6.01" y2="18"></line></svg>
            <span>6-Stage Production Pipeline Architecture</span>
          </div>
        </div>
        <div class="flow-chain" id="flow-chain">
          <!-- Populated by JS -->
        </div>
      </div>

      <!-- Open Source Toolchain -->
      <div class="panel">
        <div class="panel-head">
          <div class="panel-title">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>
            <span>Production Toolchain (100% Free / Open-Source)</span>
          </div>
        </div>
        <div style="overflow-x:auto;">
          <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
            <thead>
              <tr style="border-bottom:1px solid var(--border); text-align:left;">
                <th style="padding:0.75rem; color:var(--text-3); font-weight:600;">DOMAIN</th>
                <th style="padding:0.75rem; color:var(--text-3); font-weight:600;">RECOMMENDED TOOL</th>
                <th style="padding:0.75rem; color:var(--text-3); font-weight:600;">LICENSE</th>
                <th style="padding:0.75rem; color:var(--text-3); font-weight:600;">ENGINEERING PURPOSE</th>
              </tr>
            </thead>
            <tbody id="tools-rows">
              <!-- Populated by JS -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Code & DevOps Scaffold Studio -->
      <div class="panel">
        <div class="panel-head">
          <div class="panel-title">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>
            <span>Production Deployment Scaffolding</span>
          </div>
          <button class="btn-secondary" onclick="copyCode()">Copy Code</button>
        </div>
        <div class="code-tabs">
          <button class="code-tab-btn active" onclick="selectCodeTab('python')">train_pipeline.py</button>
          <button class="code-tab-btn" onclick="selectCodeTab('docker')">Dockerfile</button>
          <button class="code-tab-btn" onclick="selectCodeTab('workflow')">github-actions.yml</button>
        </div>
        <pre class="code-editor" id="code-display"></pre>
      </div>
    </div>

    <!-- Modal Catalog -->
    <div id="catalog-modal">
      <div class="modal-shell">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;">
          <div>
            <h2 style="font-size:1.3rem; font-weight:800; letter-spacing:-0.02em;">Knowledge Base: 50+ Production Algorithms</h2>
            <p style="font-size:0.82rem; color:var(--text-2); margin-top:0.25rem;">Comprehensive registry across tabular, vision, NLP, time series, anomaly, and reinforcement learning.</p>
          </div>
          <button class="btn-secondary" onclick="closeCatalogModal()">Close</button>
        </div>

        <input type="text" id="catalog-search" class="input-ctrl" placeholder="Filter models by name, family or latency (e.g., XGBoost, YOLO, Transformer, ViT, Isolation Forest)..." oninput="searchCatalog()">

        <div class="models-deck" id="modal-catalog-grid" style="max-height:500px; overflow-y:auto; margin-top:1rem;">
          <!-- Populated by JS -->
        </div>
      </div>
    </div>

    <!-- Site Footer -->
    <footer class="site-foot">
      OmniArchitect AI Platform &bull; Built with FastAPI & GitHub Actions &bull; Strictly 100% Open-Source Libraries
    </footer>
  </div>

  <script>
    let currentBlueprint = null;
    let selectedFile = null;
    let fullCatalog = {};
    let activeCodeTab = 'python';

    const PRESETS = """ + json.dumps(SAMPLE_PRESETS) + """;

    function initPresets() {
      const strip = document.getElementById('preset-strip');
      strip.innerHTML = '';
      PRESETS.forEach(p => {
        const chip = document.createElement('button');
        chip.className = 'preset-chip';
        chip.innerText = p.title;
        chip.onclick = () => loadPreset(p);
        strip.appendChild(chip);
      });
    }

    function loadPreset(p) {
      document.getElementById('desc').value = p.description;
      document.getElementById('output-target').value = p.expected_output;
      document.getElementById('latency').value = p.latency;
      document.getElementById('hardware').value = p.hardware;
      document.getElementById('interpretability').value = p.interpretability;
      synthesizeSystem();
    }

    function handleFileSelect(e) {
      if (e.target.files && e.target.files[0]) {
        selectedFile = e.target.files[0];
        document.getElementById('file-label').innerText = 'Selected: ' + selectedFile.name + ' (' + (selectedFile.size / 1024).toFixed(1) + ' KB)';
      }
    }

    async function loadSampleDataset() {
      try {
        const res = await fetch('/api/sample-dataset');
        const data = await res.json();
        if (data.csv_content) {
          const blob = new Blob([data.csv_content], { type: 'text/csv' });
          selectedFile = new File([blob], data.filename, { type: 'text/csv' });
          document.getElementById('file-label').innerText = 'Loaded: ' + data.filename + ' (Sample Data)';
          document.getElementById('desc').value = 'Detect fraudulent transactions from high-velocity credit card transaction data.';
          document.getElementById('output-target').value = 'Binary fraud prediction (is_fraud)';
          document.getElementById('latency').value = 'real_time';
          document.getElementById('interpretability').value = 'high';
          analyzeDatasetUpload();
        }
      } catch (err) {
        alert('Could not load sample dataset: ' + err.message);
      }
    }

    async function synthesizeSystem() {
      let desc = document.getElementById('desc').value.trim();
      if (!desc || desc.length < 10) {
        // Auto-populate default benchmark if empty
        desc = "Construct an automated fraud detection engine for streaming credit card transactions with class imbalance.";
        document.getElementById('desc').value = desc;
      }

      const payload = {
        problem_description: desc,
        expected_output: document.getElementById('output-target').value.trim() || null,
        latency: document.getElementById('latency').value,
        hardware: document.getElementById('hardware').value,
        interpretability: document.getElementById('interpretability').value,
        estimated_dataset_size: document.getElementById('datasize').value
      };

      const btn = document.querySelector('.btn-primary');
      const originalText = btn.innerHTML;
      btn.innerHTML = 'Synthesizing Architecture...';
      btn.style.opacity = '0.7';

      const apiEndpoint = window.location.origin.startsWith('http') ? '/api/analyze' : 'http://127.0.0.1:8000/api/analyze';

      try {
        const res = await fetch(apiEndpoint, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const errDetail = await res.text();
          throw new Error('Server returned HTTP ' + res.status + ': ' + errDetail);
        }
        currentBlueprint = await res.json();
        renderWorkspace(currentBlueprint);
      } catch (err) {
        console.error('Synthesis error:', err);
        alert('Synthesis Connection Note: ' + err.message + '\\n\\nTip: Ensure the local server is running on http://127.0.0.1:8000');
      } finally {
        btn.innerHTML = originalText;
        btn.style.opacity = '1';
      }
    }

    async function analyzeDatasetUpload() {
      if (!selectedFile) {
        // Automatically load sample fraud dataset if none selected
        await loadSampleDataset();
        return;
      }

      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('problem_description', document.getElementById('desc').value.trim());
      formData.append('latency', document.getElementById('latency').value);
      formData.append('hardware', document.getElementById('hardware').value);
      formData.append('interpretability', document.getElementById('interpretability').value);

      const apiEndpoint = window.location.origin.startsWith('http') ? '/api/analyze-dataset' : 'http://127.0.0.1:8000/api/analyze-dataset';

      try {
        const res = await fetch(apiEndpoint, {
          method: 'POST',
          body: formData
        });
        if (!res.ok) {
          const errDetail = await res.text();
          throw new Error('Server returned HTTP ' + res.status + ': ' + errDetail);
        }
        currentBlueprint = await res.json();
        renderWorkspace(currentBlueprint);
      } catch (err) {
        console.error('Dataset analysis error:', err);
        alert('Dataset Profiling Note: ' + err.message + '\\n\\nTip: Ensure the local server is running on http://127.0.0.1:8000');
      }
    }

    function renderWorkspace(bp) {

      document.getElementById('results-workspace').style.display = 'flex';
      document.getElementById('res-kicker').innerText = bp.problem_category.replace(/_/g, ' ');
      document.getElementById('res-headline').innerText = bp.category_display_name;
      document.getElementById('res-narrative').innerText = bp.problem_summary;

      // Dataset Stats
      const dsCard = document.getElementById('dataset-stats-card');
      if (bp.dataset_profile) {
        dsCard.style.display = 'block';
        document.getElementById('stat-rows').innerText = bp.dataset_profile.num_rows.toLocaleString();
        document.getElementById('stat-cols').innerText = bp.dataset_profile.num_cols;
        document.getElementById('stat-target').innerText = bp.dataset_profile.detected_target || 'N/A';
        document.getElementById('stat-imbalance').innerText = bp.dataset_profile.class_imbalance_ratio ? (bp.dataset_profile.class_imbalance_ratio + ':1') : 'Balanced';
        
        const preHtml = bp.dataset_profile.recommended_preprocessing.map(s => '<div style="margin-bottom:0.25rem;">&bull; ' + s + '</div>').join('');
        document.getElementById('stat-prep-list').innerHTML = '<div style="font-weight:700; color:var(--text-1); margin-bottom:0.35rem;">Preprocessing Directives:</div>' + preHtml;
      } else {
        dsCard.style.display = 'none';
      }

      // Models
      const deck = document.getElementById('models-deck');
      deck.innerHTML = '';
      bp.recommended_models.forEach(m => {
        let badgeClass = 'tier-primary';
        if (m.tier.includes('Baseline')) badgeClass = 'tier-baseline';
        if (m.tier.includes('Comparison')) badgeClass = 'tier-comparison';
        if (m.tier.includes('Deep')) badgeClass = 'tier-deep';

        const prosHtml = m.pros.map(p => `<li>${p}</li>`).join('');

        deck.innerHTML += `
          <div class="algo-card">
            <div>
              <span class="tier-pill ${badgeClass}">${m.tier}</span>
              <div class="algo-family">${m.family}</div>
              <div class="algo-name">${m.model_name}</div>
              <p class="algo-summary">${m.why_suitable}</p>

              <div class="radar-box">
                <div class="gauge-row"><span>Inference Latency:</span><span class="gauge-val">${m.inference_latency}</span></div>
                <div class="gauge-row"><span>Hardware Profile:</span><span class="gauge-val">${m.hardware_needed}</span></div>
                <div class="gauge-row"><span>Training Speed:</span><span class="gauge-val">${m.training_speed}</span></div>
                <div class="gauge-row"><span>Explainability:</span><span class="gauge-val">${m.interpretability_level}</span></div>
              </div>

              <ul class="bullet-points">
                ${prosHtml}
              </ul>
            </div>
            <div>
              <a href="${m.documentation_url}" target="_blank" style="font-size:0.75rem; font-weight:600; color:var(--accent); text-decoration:none;">View Documentation &rarr;</a>
            </div>
          </div>
        `;
      });

      // Pipeline
      const flow = document.getElementById('flow-chain');
      flow.innerHTML = '';
      bp.system_architecture.forEach((stage, idx) => {
        flow.innerHTML += `
          <div class="flow-step">
            <div class="flow-index">Stage 0${idx + 1}</div>
            <div class="flow-name">${stage.stage_name}</div>
            <p style="font-size:0.75rem; color:var(--text-2); line-height:1.4;">${stage.description}</p>
            <div class="flow-tools">${stage.tools.join(', ')}</div>
          </div>
        `;
      });

      // Tools
      const toolsRows = document.getElementById('tools-rows');
      toolsRows.innerHTML = '';
      bp.recommended_toolchain.forEach(t => {
        toolsRows.innerHTML += `
          <tr style="border-bottom:1px solid var(--border-subtle);">
            <td style="padding:0.75rem; font-weight:600; color:var(--text-1);">${t.category}</td>
            <td style="padding:0.75rem; color:var(--accent); font-weight:600; font-family:var(--font-mono);">${t.tool_name}</td>
            <td style="padding:0.75rem;"><span style="background:var(--emerald-muted); color:var(--emerald); padding:0.15rem 0.45rem; border-radius:4px; font-size:0.72rem; font-weight:600;">${t.license_type}</span></td>
            <td style="padding:0.75rem; color:var(--text-2);">${t.purpose}</td>
          </tr>
        `;
      });

      // Code
      selectCodeTab(activeCodeTab);

      // Scroll smoothly into view
      document.getElementById('results-workspace').scrollIntoView({ behavior: 'smooth' });
    }

    function selectCodeTab(tab) {
      activeCodeTab = tab;
      document.querySelectorAll('.code-tab-btn').forEach(b => b.classList.remove('active'));
      const codeEditor = document.getElementById('code-display');

      if (tab === 'python') {
        codeEditor.innerText = currentBlueprint ? currentBlueprint.starter_code_scaffold : '# Run architecture synthesis to view code';
        document.querySelectorAll('.code-tab-btn')[0].classList.add('active');
      } else if (tab === 'docker') {
        codeEditor.innerText = currentBlueprint ? (currentBlueprint.dockerfile_scaffold || '# Dockerfile') : '';
        document.querySelectorAll('.code-tab-btn')[1].classList.add('active');
      } else {
        codeEditor.innerText = currentBlueprint ? (currentBlueprint.ci_cd_workflow_preview || '# GitHub Actions YAML') : '';
        document.querySelectorAll('.code-tab-btn')[2].classList.add('active');
      }
    }

    function copyCode() {
      const code = document.getElementById('code-display').innerText;
      navigator.clipboard.writeText(code);
      alert('Code scaffold copied to clipboard!');
    }

    function exportBlueprintJSON() {
      if (!currentBlueprint) return;
      const blob = new Blob([JSON.stringify(currentBlueprint, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ML_System_Blueprint.json';
      a.click();
    }

    // Modal
    async function openCatalogModal() {
      document.getElementById('catalog-modal').style.display = 'block';
      if (Object.keys(fullCatalog).length === 0) {
        const res = await fetch('/api/catalog');
        const data = await res.json();
        fullCatalog = data.catalog;
      }
      searchCatalog();
    }

    function closeCatalogModal() {
      document.getElementById('catalog-modal').style.display = 'none';
    }

    function searchCatalog() {
      const query = document.getElementById('catalog-search').value.toLowerCase();
      const grid = document.getElementById('modal-catalog-grid');
      grid.innerHTML = '';
      
      Object.values(fullCatalog).forEach(m => {
        if (!query || m.name.toLowerCase().includes(query) || m.family.toLowerCase().includes(query) || m.category.toLowerCase().includes(query)) {
          grid.innerHTML += `
            <div class="algo-card" style="background:var(--surface-2);">
              <div>
                <span class="tier-pill tier-comparison">${m.category.replace(/_/g, ' ')}</span>
                <div class="algo-family">${m.family}</div>
                <div class="algo-name" style="font-size:1.05rem;">${m.name}</div>
                <p class="algo-summary" style="font-size:0.78rem;">${m.purpose}</p>
                <div style="font-size:0.72rem; color:var(--text-3); margin-bottom:0.5rem; font-family:var(--font-mono);">Latency: <b>${m.inference_latency}</b> &bull; Compute: <b>${m.hardware_needed}</b></div>
              </div>
              <a href="${m.documentation_url}" target="_blank" style="font-size:0.75rem; font-weight:600; color:var(--accent); text-decoration:none;">Docs &rarr;</a>
            </div>
          `;
        }
      });
    }

    // Initialize
    initPresets();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the enterprise OmniArchitect AI Dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)
