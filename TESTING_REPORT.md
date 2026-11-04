# Testing Report

## Cybersecurity Incident Severity Prediction System

## 1. Objective

To check that the training pipeline, the prediction API and the dashboard work correctly, to measure the performance of the two models on unseen incidents, and to record the bugs found and how they were fixed.

## 2. Test environment

| Item | Details |
|---|---|
| Language | Python 3.14 |
| ML libraries | pandas, NumPy, scikit-learn, joblib, matplotlib, seaborn |
| Backend | FastAPI, served with Uvicorn on port 8000 |
| Frontend | Streamlit on port 8501, charts with Plotly |
| Dataset | Microsoft GUIDE Security Incident Prediction |
| Training file size | 9,516,837 rows |
| Rows used | 50,000 random rows, after removing rows without a label and 22,541 duplicates |
| Train / test rows | 40,150 / 9,850 |
| Operating system | Windows, VS Code, PowerShell |

## 3. Testing methodology

- **Evaluation split:** an 80/20 split grouped by `IncidentId` (`GroupShuffleSplit`, `random_state=42`). All rows of an incident go to either training or testing. The script prints the number of incidents present in both sets, and this must be 0.
- **Metrics:** accuracy, weighted precision, weighted recall, weighted F1 and macro F1. Macro F1 is the main metric because the classes are not balanced. Train accuracy is also recorded to check for overfitting.
- **Class imbalance:** both models use `class_weight="balanced"`.
- **Functional testing:** the training script, prediction script, API endpoints and dashboard pages were run manually and their output was compared with the expected behaviour.
- **Integration testing:** the dashboard's Live Prediction page sends a test record to the API, and the result was compared with the output of `predict.py` and with the actual grade in the test file.

## 4. Model performance

Test set: 9,850 rows.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Train accuracy | 86.88% | 89.34% |
| Test accuracy | 83.50% | 84.77% |
| Weighted precision | 0.8493 | 0.8528 |
| Weighted recall | 0.8350 | 0.8477 |
| Weighted F1 | 0.8382 | 0.8482 |
| Macro F1 | 0.8269 | 0.8394 |

### Per-class results

| Class (test rows) | Model | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| BenignPositive (4,093) | Logistic Regression | 0.87 | 0.85 | 0.86 |
| BenignPositive (4,093) | Random Forest | 0.85 | 0.90 | 0.87 |
| FalsePositive (2,235) | Logistic Regression | 0.68 | 0.85 | 0.76 |
| FalsePositive (2,235) | Random Forest | 0.75 | 0.82 | 0.79 |
| TruePositive (3,522) | Logistic Regression | 0.93 | 0.81 | 0.86 |
| TruePositive (3,522) | Random Forest | 0.92 | 0.80 | 0.86 |

The confusion matrices are in `outputs/random_forest_confusion_matrix.png` and `outputs/logistic_regression_confusion_matrix.png`, and the feature importance chart is in `outputs/rf_feature_importance.png`.

### Observations

- Random Forest is ahead of Logistic Regression on every summary metric, but the gap is small (1.3 points of accuracy, 0.012 macro F1). It was chosen as the final model for this reason, with the caveat that the result comes from one split.
- The gap between train accuracy (89.34%) and test accuracy (84.77%) is about 4.6 points, so there is no sign of severe overfitting.
- FalsePositive is the hardest class for both models: it has the lowest precision (0.68 for Logistic Regression, 0.75 for Random Forest). Logistic Regression catches many FalsePositives (recall 0.85) but also labels many other incidents as FalsePositive.
- TruePositive has the highest precision (0.92 to 0.93) but the lowest recall (0.80 to 0.81), so about one in five real threats is predicted as another class.

### Results across development stages

| Stage | Setup | Random Forest accuracy | Logistic Regression accuracy |
|---|---|---:|---:|
| 1 | First 1,000 rows, random stratified split | 72.36% | 63.82% |
| 2 | 50,000 random rows, split by `IncidentId`, identifier columns still included | 81.74% | 61.47% |
| 3 (final) | Identifier columns removed, duplicates removed, code columns treated as categories | 84.77% | 83.50% |

The stage 1 score is not comparable with the later ones, because the random split let rows of the same incident appear in both training and test data. Between stages 2 and 3, Logistic Regression improved by about 22 points. Code-like columns such as `OrgId` and `DetectorId` were being scaled as numbers in stage 2, and in stage 3 they are one-hot encoded categories. This was the main change for that model, though the identifier removal and de-duplication were made at the same time, so the individual effect of each change was not measured separately.

## 5. Test cases

| ID | Test case | Expected result | Result |
|---|---|---|---|
| T1 | Load the training file and drop rows without `IncidentGrade` | Only labelled rows remain | Passed |
| T2 | Remove duplicate rows | Duplicates are counted and removed | Passed (22,541 removed) |
| T3 | Exclude identifier columns | All 23 listed columns found in the data are dropped | Passed |
| T4 | Check for leakage between train and test | 0 incidents shared | Passed (0) |
| T5 | Create time features from `Timestamp` | `Hour`, `DayOfWeek`, `Month` are created | Passed |
| T6 | Train both models | Both train and report metrics | Passed |
| T7 | Save outputs | Comparison CSV, reports, confusion matrices and charts are written to `outputs/` | Passed |
| T8 | Save models | `.joblib` files and the feature list are written to `models/` | Passed |
| T9 | `predict.py` on a test record | Prints a grade and class probabilities | Passed |
| T10 | `GET /` | Returns a running message, status 200 | Passed |
| T11 | `GET /health` | Returns `model_loaded: true`, status 200 | Passed |
| T12 | `GET /metrics` | Returns the comparison table as JSON, status 200 | Passed |
| T13 | `/docs` page | The interactive API documentation opens | Passed |
| T14 | Dashboard sidebar with the API running | Shows "Backend API online" | Passed |
| T15 | Dashboard sidebar with the API stopped | Shows "Backend API offline" and uses the local CSV | Passed |
| T16 | Live Prediction, test record 0 | Predicted grade is returned and matches `predict.py` (FalsePositive) | Passed |
| T17 | Live Prediction, test record 2 | Predicted grade is returned (BenignPositive), same as the actual grade | Passed |
| T18 | Dashboard pages | Overview, Model Analytics, Incident Explorer and About load without errors | Passed |

Only a few records were checked in T9, T16 and T17. They show that the pipeline works end to end, and they are not a measure of accuracy.

## 6. Bugs found and corrective actions

| # | Bug or issue | Cause | Corrective action |
|---|---|---|---|
| 1 | Results based on too little data | Only the first 1,000 rows were used, which may not be representative | Use a random sample of 50,000 rows |
| 2 | Data leakage | A random split put rows of the same incident in both training and test sets, which inflates the score | Split by `IncidentId` with `GroupShuffleSplit`, and print the overlap (0) |
| 3 | `Timestamp` created many useless columns | It was encoded as a text category | Extract `Hour`, `DayOfWeek` and `Month`, and drop the raw timestamp |
| 4 | Identifier columns were used as model input, against the proposal | Only a few ID columns were excluded | Exclude 23 identifier, device, account, network and action columns |
| 5 | Duplicate rows | Same record appeared many times | Remove duplicates before sampling |
| 6 | Code columns treated as numbers | `OrgId`, `DetectorId`, `AlertTitle`, country, state, city and similar were stored as numbers, which a linear model cannot use meaningfully | Convert them to text categories and one-hot encode them |
| 7 | Same code could be read differently in training and in the API (`123` and `123.0`) | Missing values change the column type | A shared conversion rule turns both into the same text |
| 8 | `requirements.txt` was incomplete | `streamlit`, `plotly` and the API libraries were missing | Added them and pinned all versions |
| 9 | Messy training script | More than 500 lines of old commented code and duplicate print lines | Removed |
| 10 | Dashboard looked for files that were no longer created | `classification_report.txt` and `class_distribution.png` came from an older script | `main.py` now creates the chart, and the dashboard reads `model_reports.txt`. Old files were deleted |
| 11 | Hardcoded accuracy on the dashboard | A fixed value (72.36%) was typed in the code | The value is now read from the API or the comparison CSV |
| 12 | Library deprecation warnings | `dayofweek` in pandas and `use_container_width` in Streamlit | Replaced with `day_of_week` and `width="stretch"` |
| 13 | Dashboard theme was inconsistent | Light default theme mixed with custom dark styling | Added `.streamlit/config.toml` and adjusted the CSS |
| 14 | Documentation errors | PowerShell commands were pasted at the end of the old Markdown files, and the progress report said that no user interface existed | Documents were rewritten |

## 7. Limitations of this testing

- All results come from one 50,000-row sample and one group-wise split. They may change with another sample or split.
- Cross-validation was not performed, so there is no estimate of the spread of the scores.
- No hyperparameter tuning was done. The models use mostly default settings.
- The test cases were run manually. There are no automated unit tests yet.
- The Live Prediction check used a few records, and the API was not load tested.
- The model has not been tested on an organisation that is absent from the training data.

## 8. Conclusion

The full workflow runs successfully: training, evaluation, model saving, the REST API and the dashboard that calls the API. After removing identifier columns, removing duplicates and splitting by incident, Random Forest reached 84.77% test accuracy and 0.839 macro F1, slightly ahead of Logistic Regression at 83.50% and 0.827. The testing also found and fixed a number of problems, including data leakage and identifier columns in the input. Further work is needed on cross-validation, tuning and automated tests before the performance can be treated as final.
