from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data_utils import ROOT, FEATURES, load_dataset

MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def main():
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print("Missing values after cleaning:\n", df.isna().sum())
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Crop classes:", df["label"].nunique())
    df.describe().to_csv(OUTPUTS / "descriptive_statistics.csv")
    df["label"].value_counts().sort_values().plot(kind="barh", figsize=(9, 8))
    plt.title("Number of Samples per Crop")
    plt.xlabel("Samples")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "crop_distribution.png", dpi=160)
    plt.close()
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[FEATURES].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Input Features")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_correlation.png", dpi=160)
    plt.close()

    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, random_state=42, class_weight="balanced"),
    }
    results = {}
    best_name, best_model, best_score = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        results[name] = {"accuracy": float(score)}
        print(f"{name} accuracy: {score:.4f}")
        if score > best_score:
            best_name, best_model, best_score = name, model, score
    best_pred = best_model.predict(X_test)
    report = classification_report(y_test, best_pred, zero_division=0)
    (OUTPUTS / "classification_report.txt").write_text(
        f"Best model: {best_name}\nTest accuracy: {best_score:.4f}\n\n{report}\n",
        encoding="utf-8"
    )
    with open(OUTPUTS / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "test_accuracy": best_score, "models": results}, f, indent=2)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, xticks_rotation=90, ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=160)
    plt.close()
    joblib.dump({"model": best_model, "features": FEATURES, "model_name": best_name,
                 "test_accuracy": float(best_score), "classes": sorted(y.unique())},
                MODELS / "crop_recommendation_model.joblib")
    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
        importance.plot(kind="barh", figsize=(8, 5), title="Feature Importance")
        plt.tight_layout()
        plt.savefig(OUTPUTS / "feature_importance.png", dpi=160)
        plt.close()
    print(f"\nBest model: {best_name}; accuracy: {best_score:.4f}")
    print("Saved model to models/crop_recommendation_model.joblib")
    print("Saved charts and evaluation files to outputs/")


if __name__ == "__main__":
    main()
