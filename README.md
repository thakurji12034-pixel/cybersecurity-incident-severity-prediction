# Cybersecurity Incident Severity Prediction System

A machine learning project that predicts the triage grade (`IncidentGrade`) of a security incident using Microsoft's GUIDE dataset. The trained model is served through a FastAPI backend and explored through a Streamlit dashboard.

## Problem

Security Operations Centers receive a very large number of alerts, and many of them turn out to be false positives. This project learns from historical, labelled incidents and predicts the grade of a new incident, so that analysts can prioritise their time.

**Target variable:** `IncidentGrade`, with three classes:

| Class | Meaning |
|---|---|
| BenignPositive | The alert was correct, but the activity was not harmful |
| FalsePositive | The alert was wrong, there is no real threat |
| TruePositive | A real security threat |

`IncidentGrade` is an analyst triage verdict. Reading it as "severity" (TruePositive = high, BenignPositive = medium, FalsePositive = low) is a design interpretation of this project.

## Architecture

```
 Streamlit dashboard (app.py)        FastAPI backend (api.py)          Saved model
 frontend, port 8501        --HTTP/JSON-->   port 8000          -->   models/random_forest.joblib
 Overview, Model Analytics,         GET  /health
 Live Prediction, API Monitor,      GET  /metrics
 Incident Explorer, About Project   POST /predict
                                    GET  /logs
                      ^
                      |  main.py trains the models and writes outputs/ and models/
```

- `main.py` is the training and evaluation pipeline.
- `api.py` loads the saved Random Forest and exposes it as a REST API.
- `app.py` is the frontend. It calls the API for the model metrics and for live predictions. If the API is offline, the dashboard says so in the sidebar and reads the metrics from the local CSV instead.
- `predict.py` is a small command-line demo of a single prediction.

## Project structure

```
Cybersecurity-Incident-Severity-Prediction/
├── .streamlit/config.toml     dashboard theme
├── dataset/                   GUIDE_Train.csv, GUIDE_Test.csv (not in the repository)
├── models/                    saved models and feature list (.joblib)
├── outputs/                   metrics, reports and graphs
├── screenshots/               screenshots of the working system
├── api.py                     FastAPI backend
├── app.py                     Streamlit frontend
├── main.py                    training and evaluation
├── predict.py                 single-record prediction demo
├── requirements.txt
├── README.md
├── PROGRESS_REPORT.md
└── TESTING_REPORT.md
```

## Dataset

Microsoft Security Incident Prediction (GUIDE) from Kaggle:
https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction

Download it and place `GUIDE_Train.csv` and `GUIDE_Test.csv` inside the `dataset/` folder. The training file has 9,516,837 rows. A random sample of 50,000 labelled rows is used for training and evaluation.

## Method

1. Load the training file and remove rows with a missing `IncidentGrade`.
2. Remove duplicate rows (22,541 were removed).
3. Take a random sample of 50,000 rows.
4. Exclude 23 columns that could cause memorisation or are not available at triage time: row, incident and alert IDs, device, account, IP, hash, URL, file and registry fields, email and network identifiers, and the action columns.
5. From `Timestamp`, create `Hour`, `DayOfWeek` and `Month`.
6. Treat code-like columns (for example `OrgId`, `DetectorId`, `AlertTitle`, `CountryCode`) as categories, not as numbers.
7. Split 80/20 **by `IncidentId`** (`GroupShuffleSplit`), so no incident appears in both training and test data. The code checks this and prints the number of shared incidents (0).
8. Preprocess inside a scikit-learn `Pipeline`: median imputation and scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical columns (rare values are grouped).
9. Train two models with `class_weight="balanced"`: Logistic Regression (baseline) and Random Forest (main model).
10. Evaluate with accuracy, precision, recall, weighted F1 and macro F1, and save confusion matrices.

The final feature set has 3 numeric and 21 categorical columns.

## Results

Evaluation on 9,850 test rows (40,150 training rows):

| Model | Train accuracy | Test accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest | 89.34% | 84.77% | 0.848 | 0.839 |
| Logistic Regression | 86.88% | 83.50% | 0.838 | 0.827 |

Random Forest is the selected model, with a small lead over the baseline. These numbers come from a single group-wise split on a 50,000-row sample, so they are an estimate and not a final figure for the full dataset. Per-class results are in `outputs/model_reports.txt`, and the full discussion is in `TESTING_REPORT.md`.

## Installation

Python 3.14 was used for development.

```powershell
python -m venv cyber_env
.\cyber_env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this first in the same window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## How to run

**1. Train and evaluate** (creates `models/` and `outputs/`):

```powershell
python main.py
```

**2. Optional single-record demo:**

```powershell
python predict.py
```

**3. Start the backend API** (keep this terminal open):

```powershell
python -m uvicorn api:app --port 8000
```

**4. Start the dashboard** (in a second terminal, with the environment activated):

```powershell
streamlit run app.py
```

Open `http://localhost:8501` for the dashboard and `http://127.0.0.1:8000/docs` for the interactive API documentation.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Shows whether the API is running and the model is loaded |
| GET | `/metrics` | Returns the model comparison table as JSON |
| POST | `/predict` | Takes one incident record and returns the predicted grade and class probabilities |
| GET | `/logs` | Returns recorded requests as JSON, newest first |

Example request body for `/predict`. Fields that are left out are treated as missing values:

```json
{
  "record": {
    "OrgId": 0,
    "DetectorId": 0,
    "Timestamp": "2024-06-04T12:00:00.000Z"
  }
}
```

Example response shape:

```json
{
  "prediction": "BenignPositive",
  "probabilities": {
    "BenignPositive": 0.0,
    "FalsePositive": 0.0,
    "TruePositive": 0.0
  }
}
```

The values above only show the format, they are not real output. The dashboard's **Live Prediction** page sends real records from `GUIDE_Test.csv` to this endpoint.

## Dashboard pages

- **Overview:** key numbers, model comparison table and saved charts
- **Model Analytics:** metric chart, confusion matrices, feature importance and the classification report
- **Live Prediction:** sends a test record to the API and shows the predicted grade and probabilities
- **API Monitor:** inspects API status, endpoints table, Swagger link, interactive test calls, dashboard call logs, and backend logs
- **Incident Explorer:** preview of rows from the training file
- **About Project:** short description and architecture

## Screenshots

Screenshots of the running system are in the `screenshots/` folder.

## Limitations

- Training and evaluation use a 50,000-row random sample, not the full 9.5 million rows.
- A single group-wise split is used. Cross-validation and hyperparameter tuning have not been done.
- Organisation, detector and alert-type codes are strong signals. Performance on a completely new organisation may be lower.
- `predict.py` and the Live Prediction page show single records and are demonstrations, not performance evaluations.
- This is an offline academic prototype. It has no real-time SIEM integration, authentication, database or monitoring.

## Future scope

Cross-validation and hyperparameter tuning, training on more data, containerisation with Docker, automated tests in CI, and prediction logging and monitoring.
