import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Project directory paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Backend Flask API address
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CyberShield | Incident Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark cybersecurity theme
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1728 100%);
    color: #e6f1ff;
}
header[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background-color: #0a1422;
    border-right: 1px solid #1b344b;
}
[data-testid="stSidebar"] * {
    color: #d5e4f3;
}
h1, h2, h3, h4 {
    color: #e6f1ff !important;
}
div[data-testid="stMetric"] {
    background: #101f32;
    border: 1px solid #21415b;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #38d9ff;
}
div[data-testid="stMetricLabel"] p {
    color: #c3d4e5 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #b8cce0;
}
.small-note {
    color: #93a9bf;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


def call_api(method: str, path: str, json_body: dict = None) -> dict:
    """
    Send an HTTP request to the backend Flask API, measure response time,
    and log every call to st.session_state['api_calls'].
    Never crashes when the API is offline.
    """
    if "api_calls" not in st.session_state:
        st.session_state["api_calls"] = []

    method_upper = method.upper()
    url = f"{API_URL}{path}"
    start_time = time.perf_counter()

    ok = False
    status_code = None
    data = None
    error = None

    try:
        if method_upper == "GET":
            response = requests.get(url, timeout=10)
        elif method_upper == "POST":
            response = requests.post(url, json=json_body, timeout=30)
        else:
            response = requests.request(method_upper, url, json=json_body, timeout=10)

        status_code = response.status_code
        ok = response.ok
        try:
            data = response.json()
        except Exception:
            data = response.text

        if not ok:
            error = f"HTTP {status_code}: {response.text}"

    except Exception as exc:
        ok = False
        status_code = None
        data = None
        error = str(exc)

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Append call details to session state log
    st.session_state["api_calls"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "method": method_upper,
        "endpoint": path,
        "status": status_code if status_code is not None else "Offline",
        "ms": duration_ms
    })

    # Keep the latest 100 calls in memory
    if len(st.session_state["api_calls"]) > 100:
        st.session_state["api_calls"] = st.session_state["api_calls"][-100:]

    return {
        "ok": ok,
        "status_code": status_code,
        "ms": duration_ms,
        "data": data,
        "error": error
    }


@st.cache_data
def load_test_rows(path: str, n: int) -> pd.DataFrame:
    """Load preview rows from the test CSV file with caching."""
    return pd.read_csv(path, nrows=n, low_memory=False)


# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Sidebar Header & Navigation
st.sidebar.markdown("# 🛡️ CyberShield")
st.sidebar.caption("INCIDENT INTELLIGENCE PLATFORM")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Model Analytics", "Live Prediction",
     "API Monitor", "Incident Explorer", "About Project"]
)
st.sidebar.divider()
st.sidebar.markdown("**System status**")
st.sidebar.success("Dashboard interface online")

# Check backend API health via call_api
health_res = call_api("GET", "/health")
health_data = health_res["data"] if health_res["ok"] and isinstance(health_res["data"], dict) else {}
api_online = bool(health_res["ok"] and health_data.get("model_loaded"))

if api_online:
    st.sidebar.success("Backend API online (Flask :8000)")
else:
    st.sidebar.error("Backend API offline")
    st.sidebar.caption("Start it with: python api.py")

# Main Header
st.markdown(
    "<p style='color:#38d9ff; letter-spacing:3px;'>CYBERSECURITY / MACHINE LEARNING</p>",
    unsafe_allow_html=True
)
st.title("Cyber Incident Intelligence")
st.markdown(
    "Explore incident patterns and evaluate machine-learning predictions."
)

# Load metrics: from backend API first, fallback to local CSV
comparison = None
data_source = "local file"

metrics_res = call_api("GET", "/metrics")
if metrics_res["ok"] and isinstance(metrics_res["data"], list):
    comparison = pd.DataFrame(metrics_res["data"])
    data_source = "backend API"
else:
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    if comparison_path.exists():
        try:
            comparison = pd.read_csv(comparison_path)
        except Exception:
            comparison = None

# ==============================================================================
# Page 1: Overview
# ==============================================================================
if page == "Overview":
    st.markdown("### Security Operations Overview")
    st.caption("Prototype dashboard • Microsoft GUIDE dataset project")

    if (comparison is not None and not comparison.empty
            and "Accuracy" in comparison.columns):
        best_acc = f"{comparison['Accuracy'].max():.2%}"
    else:
        best_acc = "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ML Models", "02")
    c2.metric("Best Recorded Accuracy", best_acc)
    c3.metric("Incident Classes", "03")
    c4.metric("Dataset", "GUIDE")

    st.markdown("### Incident classification")
    st.info(
        "The model predicts one of the dataset's original labels: "
        "BenignPositive, FalsePositive, or TruePositive."
    )

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Model performance comparison")
        if comparison is not None and not comparison.empty:
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.caption(f"Data source: {data_source}")
        else:
            st.warning("Model comparison was not found. Run main.py first.")

    with right:
        st.markdown("#### Saved visualizations")
        chart_files = [
            ("Model comparison", "model_comparison.png"),
            ("Incident class distribution", "class_distribution.png"),
        ]
        for label, filename in chart_files:
            image_path = OUTPUT_DIR / filename
            if image_path.exists():
                st.caption(label)
                st.image(str(image_path), use_container_width=True)
            else:
                st.caption(f"{label}: image not available")

# ==============================================================================
# Page 2: Model Analytics
# ==============================================================================
elif page == "Model Analytics":
    st.markdown("### Machine Learning Analytics")
    st.write("Review the saved evaluation results from the model training stage.")

    if comparison is not None and not comparison.empty:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.caption(f"Data source: {data_source}")

        numeric_cols = comparison.select_dtypes(include="number").columns.tolist()
        model_col = next(
            (col for col in comparison.columns
             if col.lower() in ["model", "model_name", "algorithm"]),
            None
        )

        if model_col and numeric_cols:
            metric = st.selectbox("Choose metric to compare", numeric_cols)
            fig = px.bar(
                comparison,
                x=model_col,
                y=metric,
                color=model_col,
                title=f"Model comparison: {metric}",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="#0b1728",
                plot_bgcolor="#0b1728",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(
                "The saved comparison file does not have the expected model-name "
                "and numeric metric columns for an interactive chart."
            )
    else:
        st.warning("No model comparison found. Run main.py first.")

    st.markdown("#### Confusion matrices and reports")
    image_cols = st.columns(2)
    image_files = [
        ("Random Forest confusion matrix", "confusion_matrix.png"),
        ("Logistic Regression confusion matrix", "logistic_regression_confusion_matrix.png"),
        ("Overall model comparison", "model_comparison.png"),
        ("Incident class distribution", "class_distribution.png"),
    ]

    for i, (label, filename) in enumerate(image_files):
        path = OUTPUT_DIR / filename
        if path.exists():
            with image_cols[i % 2]:
                st.caption(label)
                st.image(str(path), use_container_width=True)

    report_path = OUTPUT_DIR / "model_reports.txt"
    if report_path.exists():
        with st.expander("View classification report"):
            st.text(report_path.read_text(encoding="utf-8", errors="replace"))

# ==============================================================================
# Page 3: Live Prediction
# ==============================================================================
elif page == "Live Prediction":
    st.markdown("### Live Prediction (Frontend → Flask API → Model)")
    st.write(
        "Pick a record from the test file. The dashboard sends it to the "
        "Flask backend API, which returns the predicted incident grade."
    )

    test_path = BASE_DIR / "dataset" / "GUIDE_Test.csv"

    if not api_online:
        st.error(
            "Backend API is offline. Open a terminal and run: "
            "python api.py"
        )
    elif not test_path.exists():
        st.error("GUIDE_Test.csv was not found in the dataset folder.")
    else:
        rows_df = load_test_rows(str(test_path), 200)
        index = st.number_input(
            "Record number (0 to 199)", min_value=0,
            max_value=len(rows_df) - 1, value=0, step=1
        )
        record_df = rows_df.iloc[[int(index)]]

        with st.expander("View selected record"):
            st.dataframe(record_df.astype(str).T, use_container_width=True)

        if st.button("Predict using Flask API", type="primary"):
            # Clean record of NaNs for JSON compliance
            raw_record = record_df.iloc[0].to_dict()
            clean_record = {k: v for k, v in raw_record.items() if pd.notna(v)}
            payload = {"record": clean_record}

            res = call_api("POST", "/predict", json_body=payload)

            col_s1, col_s2 = st.columns(2)
            col_s1.metric("Status code", res["status_code"] if res["status_code"] is not None else "Offline")
            col_s2.metric("Response time", f"{res['ms']:.1f} ms")

            with st.expander("Request sent to API"):
                st.json(payload)

            with st.expander("Response received from API"):
                if res["data"] is not None:
                    st.json(res["data"])
                else:
                    st.json({"error": res["error"]})

            if res["ok"] and res["data"]:
                result = res["data"]
                predicted_grade = result.get("prediction", "Unknown")

                st.metric("Predicted incident grade", predicted_grade)

                # Severity mapping
                if predicted_grade == "TruePositive":
                    severity = "HIGH"
                    st.error("🚨 Severity: **HIGH** (Real Attack - Immediate Action Required)")
                elif predicted_grade == "BenignPositive":
                    severity = "MEDIUM"
                    st.warning("⚠️ Severity: **MEDIUM** (Harmless Benign Activity)")
                elif predicted_grade == "FalsePositive":
                    severity = "LOW"
                    st.info("🟢 Severity: **LOW** (False Alarm - Safe to Close)")
                else:
                    severity = "UNKNOWN"

                # Class probabilities chart
                if "probabilities" in result and isinstance(result["probabilities"], dict):
                    probs = pd.DataFrame({
                        "Class": list(result["probabilities"].keys()),
                        "Probability": list(result["probabilities"].values())
                    })
                    fig = px.bar(
                        probs, x="Class", y="Probability", color="Class",
                        title="Class probabilities", template="plotly_dark"
                    )
                    fig.update_layout(
                        paper_bgcolor="#0b1728",
                        plot_bgcolor="#0b1728",
                        showlegend=False,
                        yaxis_range=[0, 1]
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if "IncidentGrade" in record_df.columns:
                    actual = record_df["IncidentGrade"].iloc[0]
                    if pd.notna(actual):
                        st.caption(f"Actual grade in the test file: **{actual}**")

                # Append to session state history
                st.session_state["history"].append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Record #": int(index),
                    "Category": str(clean_record.get("Category", "N/A")),
                    "Predicted Grade": predicted_grade,
                    "Severity": severity,
                    "Latency (ms)": res["ms"]
                })

            else:
                st.error(f"API request failed: {res['error']}")

    # Prediction History Table
    st.divider()
    st.markdown("### 📋 Prediction History")
    if st.session_state.get("history"):
        hist_df = pd.DataFrame(st.session_state["history"][::-1])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear Prediction History"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("No predictions recorded in this session yet.")

# ==============================================================================
# Page 4: API Monitor
# ==============================================================================
elif page == "API Monitor":
    st.markdown("### API Monitor")
    st.caption("Live inspection of Flask REST API endpoints, interactive requests, and call logs.")

    if not api_online:
        st.error(
            "Backend API is offline. Open a terminal and run: "
            "python api.py"
        )

    # Metric cards: API status, Base URL, Model loaded, Number of endpoints
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API status", "online" if api_online else "offline")
    c2.metric("Base URL", API_URL)
    c3.metric("Model loaded", "yes" if (health_data.get("model_loaded")) else "no")
    c4.metric("Number of endpoints", "4")

    # Check API Connection Button
    if st.button("🔄 Check API Connection"):
        st.rerun()

    # Table of all endpoints
    st.markdown("#### Endpoints")
    endpoints_df = pd.DataFrame([
        {"Method": "GET", "Path": "/health", "Purpose": "Check whether Flask API is running and model is loaded"},
        {"Method": "GET", "Path": "/metrics", "Purpose": "Return model comparison and evaluation metrics"},
        {"Method": "POST", "Path": "/predict", "Purpose": "Take an incident record and return predicted grade and probabilities"},
        {"Method": "GET", "Path": "/logs", "Purpose": "Return recorded backend requests as JSON, newest first"},
    ])
    st.dataframe(endpoints_df, use_container_width=True, hide_index=True)

    # Explanation of architecture
    st.markdown("#### Architecture & Data Flow")
    st.markdown("""
    - **Streamlit Frontend:** Sends HTTP requests using Python `requests` library.
    - **Flask Backend API:** Listens on `http://127.0.0.1:8000`, processes input, and runs inference using the saved Scikit-learn model.
    - **Machine Learning Model:** Loaded from `models/random_forest.joblib` to produce incident grade classifications.
    """)

    # Try the API section
    st.markdown("### Try the API")

    # Part 1: GET /health
    st.markdown("#### GET /health")
    if st.button("Call GET /health", key="try_health_btn"):
        st.session_state["try_health_res"] = call_api("GET", "/health")

    if "try_health_res" in st.session_state:
        h_res = st.session_state["try_health_res"]
        col_h1, col_h2 = st.columns(2)
        col_h1.metric("Status code", h_res["status_code"] if h_res["status_code"] is not None else "Offline")
        col_h2.metric("Response time", f"{h_res['ms']:.1f} ms")
        if h_res["data"] is not None:
            st.json(h_res["data"])
        else:
            st.error(f"Error: {h_res['error']}")

    # Part 2: GET /metrics
    st.markdown("#### GET /metrics")
    if st.button("Call GET /metrics", key="try_metrics_btn"):
        st.session_state["try_metrics_res"] = call_api("GET", "/metrics")

    if "try_metrics_res" in st.session_state:
        m_res = st.session_state["try_metrics_res"]
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Status code", m_res["status_code"] if m_res["status_code"] is not None else "Offline")
        col_m2.metric("Response time", f"{m_res['ms']:.1f} ms")
        if m_res["data"] is not None:
            st.json(m_res["data"])
            if isinstance(m_res["data"], list):
                st.dataframe(pd.DataFrame(m_res["data"]), use_container_width=True, hide_index=True)
        else:
            st.error(f"Error: {m_res['error']}")

    # Part 3: POST /predict
    st.markdown("#### POST /predict")
    test_csv_path = BASE_DIR / "dataset" / "GUIDE_Test.csv"
    if test_csv_path.exists():
        test_samples = load_test_rows(str(test_csv_path), 200)
        selected_idx = st.number_input(
            "Pick a record (0 to 199)",
            min_value=0,
            max_value=len(test_samples) - 1,
            value=0,
            step=1,
            key="try_predict_idx"
        )
        sample_record_df = test_samples.iloc[[int(selected_idx)]]

        if st.button("Send to API", key="try_predict_btn", type="primary"):
            raw_rec = sample_record_df.iloc[0].to_dict()
            clean_rec = {k: v for k, v in raw_rec.items() if pd.notna(v)}
            record_payload = {"record": clean_rec}

            p_res = call_api("POST", "/predict", json_body=record_payload)
            st.session_state["try_predict_req_body"] = record_payload
            st.session_state["try_predict_response"] = p_res

        if "try_predict_response" in st.session_state:
            req_body = st.session_state.get("try_predict_req_body", {})
            p_res = st.session_state["try_predict_response"]

            with st.expander("Request body sent to API"):
                st.json(req_body)

            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Status code", p_res["status_code"] if p_res["status_code"] is not None else "Offline")
            col_p2.metric("Response time", f"{p_res['ms']:.1f} ms")

            if p_res["data"] is not None:
                st.json(p_res["data"])
                if isinstance(p_res["data"], dict) and "prediction" in p_res["data"]:
                    st.metric("Predicted grade", p_res["data"]["prediction"])
            else:
                st.error(f"Error: {p_res['error']}")
    else:
        st.error("GUIDE_Test.csv was not found in dataset folder.")

    # Table: Calls made by the dashboard
    st.markdown("### Calls made by the dashboard")
    st.caption("All API calls initiated by this Streamlit session (newest first).")
    calls = st.session_state.get("api_calls", [])
    if calls:
        dashboard_logs_df = pd.DataFrame(calls[::-1])
        st.dataframe(dashboard_logs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No dashboard API calls recorded yet.")

    # Table: Requests received by the backend
    st.markdown("### Requests received by the backend")
    st.caption("Requests recorded in-memory by Flask backend (newest first).")

    if st.button("Refresh Backend Logs", key="btn_refresh_backend_logs"):
        pass

    backend_logs_res = call_api("GET", "/logs")
    if backend_logs_res["ok"] and isinstance(backend_logs_res["data"], list):
        backend_logs = backend_logs_res["data"]
        if backend_logs:
            backend_logs_df = pd.DataFrame(backend_logs)
            st.dataframe(backend_logs_df, use_container_width=True, hide_index=True)
        else:
            st.info("No requests received by the backend yet.")
    else:
        st.warning("Could not fetch backend logs. Backend API is offline.")

# ==============================================================================
# Page 5: Incident Explorer
# ==============================================================================
elif page == "Incident Explorer":
    st.markdown("### Incident Explorer")
    st.write("Preview a small sample of records without loading the entire large CSV.")

    data_path = BASE_DIR / "dataset" / "GUIDE_Train.csv"

    if data_path.exists():
        rows = st.slider("Number of sample rows", 5, 100, 20, step=5)

        if st.button("Load sample incidents", type="primary"):
            try:
                sample = pd.read_csv(data_path, nrows=rows, low_memory=False)
                st.session_state["incident_sample"] = sample
            except Exception as error:
                st.error(f"Could not read the sample: {error}")

        if "incident_sample" in st.session_state:
            sample = st.session_state["incident_sample"]
            st.metric("Rows loaded for preview", len(sample))

            if "IncidentGrade" in sample.columns:
                st.markdown("#### Class distribution in this sample")
                counts = sample["IncidentGrade"].fillna("Missing").value_counts()
                fig = px.pie(
                    names=counts.index,
                    values=counts.values,
                    hole=0.45,
                    title="IncidentGrade distribution",
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor="#0b1728",
                    plot_bgcolor="#0b1728"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(sample, use_container_width=True, hide_index=True)
            st.caption(
                "This is only a small preview of the training file, "
                "not a live security feed."
            )
    else:
        st.error("GUIDE_Train.csv was not found in the dataset folder.")

# ==============================================================================
# Page 6: About Project
# ==============================================================================
elif page == "About Project":
    st.markdown("### About CyberShield")
    st.write(
        "Cybersecurity Incident Severity Prediction System is a student "
        "machine-learning project using Microsoft's GUIDE incident dataset."
    )

    st.markdown("#### Architecture")
    st.markdown("""
    - **Frontend:** Streamlit dashboard (`app.py`) with rich analytics and API monitoring.
    - **Backend:** Flask REST API (`api.py`) with endpoints `/health`, `/metrics`, `/predict`, `/logs`.
    - **ML layer:** Scikit-learn pipelines saved with Joblib (`main.py` trains them).
    - The frontend sends a record as JSON to `/predict`; the backend loads the saved model and returns the predicted grade and class probabilities.
    """)

    st.markdown("#### Models")
    st.markdown("""
    - **Random Forest:** main trained model (84.02% accuracy, 0.841 F1)
    - **Logistic Regression:** baseline comparison model (83.50% accuracy, 0.838 F1)
    """)

    st.markdown("#### Prediction labels")
    st.markdown("""
    - `BenignPositive`: Harmless event (Medium Severity)
    - `FalsePositive`: False alarm (Low Severity)
    - `TruePositive`: Real security threat (High Severity)
    """)

    st.warning(
        "This is an academic prototype. Its predictions are decision-support "
        "outputs, not proof that an incident is malicious or harmless."
    )

# Footer
st.divider()
st.markdown(
    "<p class='small-note'>CyberShield Academic Prototype · "
    "Machine Learning Incident Classification</p>",
    unsafe_allow_html=True
)
