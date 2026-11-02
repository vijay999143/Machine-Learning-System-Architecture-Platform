"""Streamlit Dashboard for Flower Classification ML Model.

To run:
    pip install streamlit
    streamlit run streamlit_app.py
"""
from pathlib import Path
import json
import joblib
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "src" / "model.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "src" / "metrics.json"

st.set_page_config(
    page_title="Flower Classification ML Dashboard",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Flower Classification ML Dashboard")
st.markdown("Automated ML Model Deployment using **Scikit-learn**, **FastAPI**, and **GitHub Actions**.")

# Ensure model exists
if not MODEL_PATH.exists():
    from src.train import train_model
    train_model()

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
target_names = saved_data["target_names"]

# Layout columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ Flower Measurements (cm)")
    
    # Presets
    preset = st.selectbox(
        "Choose a sample preset or customize below:",
        ["Custom", "Iris Setosa (Sample)", "Iris Versicolor (Sample)", "Iris Virginica (Sample)"]
    )
    
    defaults = {
        "Custom": (5.1, 3.5, 1.4, 0.2),
        "Iris Setosa (Sample)": (5.1, 3.5, 1.4, 0.2),
        "Iris Versicolor (Sample)": (6.0, 2.9, 4.5, 1.5),
        "Iris Virginica (Sample)": (6.7, 3.0, 5.2, 2.3),
    }[preset]
    
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, defaults[0], 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, defaults[1], 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, defaults[2], 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.6, defaults[3], 0.1)

with col2:
    st.subheader("🎯 Model Prediction")
    
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred_idx = int(model.predict(features)[0])
    pred_species = target_names[pred_idx]
    probs = model.predict_proba(features)[0]
    
    st.success(f"**Predicted Species:** Iris {pred_species.capitalize()}")
    st.metric("Top Confidence", f"{probs[pred_idx] * 100:.1f}%")
    
    st.write("### Prediction Probabilities:")
    for name, prob in zip(target_names, probs):
        st.write(f"**{name.capitalize()}**: {prob * 100:.1f}%")
        st.progress(float(prob))

# Metrics Section
if METRICS_PATH.exists():
    st.divider()
    st.subheader("📊 MLOps Pipeline Metrics")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Model Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
    mcol2.metric("Precision", f"{metrics.get('precision', 0) * 100:.1f}%")
    mcol3.metric("Recall", f"{metrics.get('recall', 0) * 100:.1f}%")
    mcol4.metric("F1-Score", f"{metrics.get('f1_score', 0) * 100:.1f}%")
