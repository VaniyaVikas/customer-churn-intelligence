
# Customer Churn Intelligence

An end-to-end machine learning project that predicts customer churn
and suggests possible retention actions.

## Features

- Data cleaning and exploratory data analysis
- Numeric and categorical feature preprocessing
- Logistic Regression and Random Forest comparison
- Model evaluation using Accuracy, Precision, Recall, F1 and ROC-AUC
- Churn probability prediction
- FastAPI prediction endpoint
- Streamlit dashboard
- Automated tests
- Docker support

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn,
FastAPI, Streamlit, Pydantic, Pytest and Docker.

## Project Structure

- `data/raw/`: Original dataset
- `data/processed/`: Processed dataset outputs
- `notebooks/`: Data exploration scripts
- `src/`: ML pipeline and prediction logic
- `api/`: FastAPI application
- `dashboard/`: Streamlit user interface
- `tests/`: Automated tests
- `models/`: Saved model artifact
- `reports/`: Metrics and generated charts

## Setup

Create and activate a virtual environment.

Windows PowerShell:

    python -m venv venv
    .\venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

## Dataset

Place the Telco Customer Churn CSV file at:

    data/raw/customer_churn.csv

The target column must be named `Churn` and contain `Yes`/`No`
or binary `1`/`0` labels.

Use a dataset you are permitted to use. Check its license and
column names before training.

## Train the Model

Run these commands from the project root:

    python -m src.train

View saved evaluation metrics:

    python -m src.evaluate

View feature importance:

    python -m src.explain

## Run the API

    uvicorn api.main:app --reload

API documentation:

    http://127.0.0.1:8000/docs

## Run the Dashboard

In another terminal:

    streamlit run dashboard/app.py

## Run Tests

    python -m pytest -q

## Important Notes

Model predictions are estimates and should be evaluated on
representative data before being used for real customer decisions.

The default prediction threshold is 0.50. Business teams should
choose a threshold based on validation results and retention costs.
