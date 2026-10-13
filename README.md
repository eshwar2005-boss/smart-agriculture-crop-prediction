# Smart Agriculture Sensor Data Analysis and Crop Prediction

A data-science project aligned with the submitted abstract: explore agricultural soil/climate data, visualize patterns, train a crop recommendation model, and use an interactive app to predict a suitable crop.

## Dataset

Use the Kaggle **Crop Recommendation Dataset** by Atharva Ingle:
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Download `Crop_recommendation.csv` and place it at `data/Crop_recommendation.csv`.

**Important project-scope note:** this commonly used dataset contains N, P, K, temperature, humidity, pH, rainfall and crop label. It does **not** contain soil-moisture readings or raw IoT sensor time-series. Therefore, this version predicts crops from soil nutrients and climate features; do not describe it as real-time sensor prediction unless sensor data is separately collected and integrated.

## Features
- Dataset quality checks and exploratory data analysis
- Charts saved under `outputs/`
- Train/test split with stratification
- Compare Logistic Regression, Random Forest, and Extra Trees
- Accuracy, classification report, confusion matrix, and feature importance
- Saved best model and interactive Streamlit prediction interface

## Setup (Windows)
```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the project
1. Put the Kaggle CSV in `data/Crop_recommendation.csv`.
2. Run analysis and train models:
```bash
python src/train_model.py
```
3. Launch the interface:
```bash
streamlit run app.py
```

The app can also train the model automatically when the saved model is missing.

## Project structure
```text
smart_agriculture_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/                 # Place Kaggle CSV here (not committed by default)
├── models/               # Trained model saved here
├── outputs/              # EDA charts and evaluation output
└── src/
    ├── __init__.py
    ├── data_utils.py
    └── train_model.py
```

## Dataset citation
Atharva Ingle, *Crop Recommendation Dataset*, Kaggle. Follow the license and attribution terms shown on the dataset page.

## Limitations
This is a recommendation model trained on a public tabular dataset, not a substitute for agronomist advice. Results depend on dataset quality and should be validated locally before agricultural use. Dataset values are not live sensor readings.
