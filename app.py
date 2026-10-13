from pathlib import Path
import joblib
import streamlit as st
from src.data_utils import ROOT, FEATURES, load_dataset
from src.train_model import main as train_model

MODEL_PATH = ROOT / "models" / "crop_recommendation_model.joblib"

st.set_page_config(page_title="Smart Agriculture | Crop Prediction", page_icon="🌱", layout="wide")
st.title("🌱 Smart Agriculture: Data Analysis & Crop Prediction")
st.write("Explore agricultural data and predict a suitable crop from soil nutrient and climate conditions.")
st.info("This demo uses a public crop-recommendation dataset. It does not receive live sensor data; soil moisture is not a feature in the standard dataset.")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Make sure the Kaggle CSV is in data/Crop_recommendation.csv.")
    if st.button("Train model now"):
        with st.spinner("Training and evaluating models..."):
            try:
                train_model()
                st.success("Training complete. Refresh the page to make a prediction.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
with st.sidebar:
    st.header("Enter field conditions")
    n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0)
    p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0)
    k = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=43.0)
    temp = st.number_input("Temperature (°C)", min_value=-5.0, max_value=60.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0)
    predict = st.button("Predict suitable crop", type="primary", use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("Selected model", bundle.get("model_name", "Trained classifier"))
with col2:
    st.metric("Held-out test accuracy", f"{bundle.get('test_accuracy', 0):.1%}")
if predict:
    import pandas as pd
    values = pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]], columns=FEATURES)
    crop = model.predict(values)[0]
    st.success(f"Recommended crop: **{str(crop).title()}**")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(values)[0]
        ranked = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:5]
        st.subheader("Top predictions")
        st.bar_chart({"Model score": {str(c).title(): float(s) for c, s in ranked}})
        st.caption("Scores are model probabilities, not guarantees of crop success.")

st.subheader("Dataset overview")
try:
    df = load_dataset()
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Crop categories", df["label"].nunique())
    c.metric("Input features", len(FEATURES))
    st.dataframe(df.head(10), use_container_width=True)
except Exception as e:
    st.warning(str(e))

for title, filename in [("Crop distribution", "crop_distribution.png"), ("Feature correlation", "feature_correlation.png"), ("Confusion matrix", "confusion_matrix.png"), ("Feature importance", "feature_importance.png")]:
    path = ROOT / "outputs" / filename
    if path.exists():
        st.subheader(title)
        st.image(str(path), use_container_width=True)
