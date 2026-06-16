# House Price Prediction MLOps

End-to-end Python MLOps project for predicting house prices from property size, room counts, amenities, furnishing status, and location-related features.

## Stack

- Python, Pandas, NumPy, Scikit-learn, XGBoost
- Flask web app
- MLflow experiment tracking
- DVC pipeline and data versioning
- Docker containerization
- GitHub Actions CI

## Project Structure

```text
house-price-prediction-mlops/
├── data/
├── notebooks/
├── src/
├── pipeline/
├── app.py
├── templates/
├── static/
├── config/
├── artifacts/
├── tests/
├── requirements.txt
├── Dockerfile
├── .github/workflows/main.yml
├── dvc.yaml
├── params.yaml
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Run The Pipeline

```bash
python -m pipeline.run_pipeline
```

This creates a sample raw dataset when one is not present, validates the data, builds preprocessing artifacts, trains Linear Regression, Random Forest, and XGBoost, then saves the best model by RMSE.

## MLflow

```bash
mlflow ui
```

Open `http://127.0.0.1:5000` for the MLflow UI if that is the port shown by MLflow.

## DVC

```bash
dvc init
dvc repro
dvc dag
```

Use `dvc add data/raw/house_prices.csv` to version a real dataset. Configure remote storage with `dvc remote add -d storage <remote-url>`.

## Flask App

Train the model first, then run:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Docker

```bash
docker build -t house-price-prediction-mlops .
docker run -p 5000:5000 house-price-prediction-mlops
```

## Tests

```bash
pytest -q
```

## Screenshots

Add screenshots of the Flask form, prediction result, MLflow experiment page, and CI workflow run here.

## File Guide

- `src/data_ingestion.py`: creates or loads raw house price data.
- `src/data_preprocessing.py`: validates schema, engineers features, handles missing values, encodes categories, scales numeric fields, and writes train/test splits.
- `src/model_training.py`: trains candidate models, tunes hyperparameters, logs runs to MLflow, and persists the best model.
- `src/model_evaluation.py`: calculates MAE, MSE, RMSE, and R2.
- `src/prediction.py`: loads the trained model and returns predictions for app/user input.
- `pipeline/run_pipeline.py`: orchestrates ingestion, preprocessing, and training.
- `app.py`: Flask entrypoint for browser predictions.
- `dvc.yaml`: reproducible DVC stages.
- `params.yaml`: central configuration for data, features, and training.
- `.github/workflows/main.yml`: CI pipeline for tests, training, and Docker build.
