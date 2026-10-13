from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_CANDIDATES = ["label", "crop", "Crop"]


def load_dataset(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download Crop_recommendation.csv from "
            "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset "
            "and place it in the project's data/ folder."
        )
    df = pd.read_csv(path)
    # Handle common target-column naming variants.
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        raise ValueError(f"Could not find crop target column. Columns found: {list(df.columns)}")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")
    df = df[FEATURES + [target]].copy().rename(columns={target: "label"})
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + ["label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if df.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")
    return df
