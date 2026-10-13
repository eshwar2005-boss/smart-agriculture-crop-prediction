# Smart Agriculture Sensor Data Analysis and Crop Prediction

A data science project that analyzes agricultural soil and climate data, visualizes patterns, trains machine learning models, and predicts suitable crops through an interactive application.

## Dataset

Use the Kaggle **Crop Recommendation Dataset** by Atharva Ingle.

Dataset link: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Download `Crop_recommendation.csv` and place it in `data/Crop_recommendation.csv`.

**Important:** This dataset contains nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, rainfall, and crop labels. It does not contain soil moisture readings or real-time IoT sensor data.

## Features

- Dataset quality checks and exploratory data analysis
- Agricultural data visualization and charts
- Train/test split with stratification
- Comparison of Logistic Regression, Random Forest, and Extra Trees
- Model accuracy and classification report
- Confusion matrix and feature importance
- Saved machine learning model
- Interactive Streamlit crop prediction interface

## Setup on Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the Project

1. Place the dataset in `data/Crop_recommendation.csv`.
2. Train the models and generate analysis outputs:

```bash
python src/train_model.py
```

3. Start the application:

```bash
python -m streamlit run app.py
```

## Project Structure

```text
smart_agriculture_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
├── outputs/
└── src/
    ├── __init__.py
    ├── data_utils.py
    └── train_model.py
```

## Limitations

This project uses a public tabular dataset and does not perform live sensor monitoring. Predictions should be validated against local agricultural conditions before practical use.

## Dataset Citation

Atharva Ingle, Crop Recommendation Dataset, Kaggle. Refer to the dataset page for licensing and attribution terms.