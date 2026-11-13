# Project Progress Report (Phase 3-4)

## Cybersecurity Incident Severity Prediction System

**Student:** ______________________  **UID:** ______________________
**Type:** Individual project

## 1. Project objective

To build a machine learning system that predicts the grade of a security incident (`IncidentGrade`: BenignPositive, FalsePositive or TruePositive) from the Microsoft GUIDE dataset, and to make the trained model available through a REST API and a dashboard.

## 2. Progress against the Phase 1-2 proposal

| Item in the proposal | Status | Notes |
|---|---|---|
| Data ingestion and basic data checks | Done | Missing targets and duplicates are removed. Only a basic check was done, there is no formal schema validation |
| Exploratory analysis of the target classes | Partly done | Class distribution chart was created. A full EDA was not done |
| Drop high-cardinality IDs and identifiers | Done | 23 identifier, device, account, network and action columns are excluded |
| Time features from `Timestamp` | Done | `Hour`, `DayOfWeek`, `Month` |
| Group-wise train/test split by `IncidentId` | Done | Zero incidents are shared between train and test |
| Preprocessing in a scikit-learn Pipeline | Done | Imputation, scaling and one-hot encoding inside the pipeline |
| Baseline model: Logistic Regression | Done | Test accuracy 83.50%, macro F1 0.827 |
| Advanced model: Random Forest | Done | Test accuracy 84.77%, macro F1 0.839 |
| Evaluation: precision, recall, macro F1, confusion matrix | Done | Also accuracy, weighted F1 and train accuracy |
| Class imbalance handling | Done | `class_weight="balanced"` and macro F1 as the main metric |
| Select the better model and save it | Done | Random Forest saved with joblib |
| Offline prediction | Done | `predict.py` and the API |
| Cross-validation | Not done | A single group-wise split is used |
| Hyperparameter tuning | Not done | Only a small regularisation setting (`min_samples_leaf=3`) was set |
| Aggregation features such as MITRE technique frequency | Not done | |

## 3. Implemented modules

1. **Training pipeline (`main.py`):** loading, cleaning, feature preparation, group-wise split, training of both models, evaluation, charts and saving of models and reports.
2. **Prediction script (`predict.py`):** loads the saved Random Forest and predicts one record from the test file.
3. **Backend API (`api.py`, Flask):** endpoints `/health`, `/metrics`, `/predict`, and `/logs`.
4. **Frontend dashboard (`app.py`, Streamlit):** Overview, Model Analytics, Live Prediction, Incident Explorer and About pages. The dashboard calls the API for metrics and predictions.
5. **Documentation:** README, testing report and this progress report.

## 4. Results

Evaluation on 9,850 unseen test rows from a 50,000-row sample:

| Model | Test accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| Random Forest | 84.77% | 0.848 | 0.839 |
| Logistic Regression | 83.50% | 0.838 | 0.827 |

Random Forest is the selected model, with a small lead. These are results from one split of a sample and not final performance on the full dataset. More detail is in `TESTING_REPORT.md`.

## 5. Deviations from the original proposal

| Change | Reason |
|---|---|
| Sample of 50,000 rows instead of a 3% sample (about 285,000 rows) | To keep memory use and training time manageable on a laptop |
| A REST API (Flask) and a Streamlit dashboard were added | They were not in the proposal. They make the model usable by a front end and show how deployment works |
| Model saved as `.joblib` instead of `.pkl` | joblib is the usual way to save scikit-learn models |
| `StandardScaler` instead of a robust scaler | Simpler, and sufficient for this prototype |
| One group-wise split instead of cross-validation | To keep the project simple. Cross-validation is planned as future work |
| Code-like columns such as `OrgId` and `DetectorId` are used as categories | They carry useful information, but they are not individual identifiers. The proposal excluded identifiers that allow memorisation |

## 6. Problems found and how they were solved

The main ones are listed here. The full table is in `TESTING_REPORT.md`.

- The first experiment used only the first 1,000 rows and a random split, which allowed rows of the same incident in both train and test data. It was replaced by a larger random sample and a split by `IncidentId`.
- Identifier columns such as account, IP and device fields were still part of the model input, which contradicted the proposal. They are now excluded.
- Code-like columns were treated as numbers, which hurt the linear model badly. They are now treated as categories, and the Logistic Regression accuracy rose from 61.47% to 83.50% in the final setup.
- The dashboard read files that were no longer created and showed a hardcoded accuracy. It now reads the saved outputs and gets its numbers from the API.

## 7. Pending work

- Push the final code to the GitHub repository and add the repository link to the submission.
- Add the dashboard and API screenshots to the `screenshots/` folder.
- Optional improvements: cross-validation, hyperparameter tuning, automated tests, Docker, and prediction logging.
- Prepare the final report, presentation and viva for Phase 5.

## 8. Limitations

- Results come from a 50,000-row sample and a single split.
- Performance on an organisation that is not in the training data is unknown.
- The system is an offline prototype. It has no real-time data feed, authentication, database or monitoring.
- Predictions are decision support for an analyst and are not proof that an incident is malicious or harmless.

## 9. Individual contribution

This is an individual project. All work was done by the student:

- Studied the dataset and defined the problem and the target
- Wrote the preprocessing and the training pipeline
- Found and fixed the data leakage problem and the identifier column problem
- Trained and evaluated the Logistic Regression and Random Forest models
- Built the Flask REST API backend and the Streamlit dashboard and connected them
- Tested the system and wrote the documentation

## 10. Conclusion

The core system works end to end. A trained Random Forest model is served by a REST API, and a dashboard uses that API to show results and live predictions. The main technical risks, data leakage and identifier columns, were found and fixed. The remaining work is mostly submission material and optional improvements, and results should be treated as preliminary until they are checked with cross-validation.
