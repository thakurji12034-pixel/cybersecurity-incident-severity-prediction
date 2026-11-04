import json

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Backend API ka address
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


def api_get(path):
    """Backend API se GET request. Fail hone par None."""
    try:
        response = requests.get(f"{API_URL}{path}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


@st.cache_data
def load_test_rows(path, n):
    return pd.read_csv(path, nrows=n, low_memory=False)


# Sidebar
st.sidebar.markdown("# 🛡️ CyberShield")
st.sidebar.caption("INCIDENT INTELLIGENCE PLATFORM")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Model Analytics", "Live Prediction",
     "Incident Explorer", "About Project"]
)
st.sidebar.divider()
st.sidebar.markdown("**System status**")
st.sidebar.success("Dashboard interface online")

health = api_get("/health")
api_online = bool(health and health.get("model_loaded"))
if api_online:
    st.sidebar.success("Backend API online")
else:
    st.sidebar.error("Backend API offline")
    st.sidebar.caption("Start it with: python -m uvicorn api:app --port 8000")

# Header
st.markdown(
    "<p style='color:#38d9ff; letter-spacing:3px;'>CYBERSECURITY / MACHINE LEARNING</p>",
    unsafe_allow_html=True
)
st.title("Cyber Incident Intelligence")
st.markdown(
    "Explore incident patterns and evaluate machine-learning predictions."
)

# Model comparison results: pehle API se, nahi mile to local CSV se
comparison = None
data_source = "local file"

metrics_data = api_get("/metrics")
if metrics_data:
    comparison = pd.DataFrame(metrics_data)
    data_source = "backend API"
else:
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    if comparison_path.exists():
        try:
            comparison = pd.read_csv(comparison_path)
        except Exception:
            comparison = None

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
            st.dataframe(comparison, width="stretch", hide_index=True)
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
                st.image(str(image_path), width="stretch")
            else:
                st.caption(f"{label}: image not available")

elif page == "Model Analytics":
    st.markdown("### Machine Learning Analytics")
    st.write("Review the saved evaluation results from the model training stage.")

    if comparison is not None and not comparison.empty:
        st.dataframe(comparison, width="stretch", hide_index=True)
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
            st.plotly_chart(fig, width="stretch")
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
        ("Random Forest confusion matrix", "random_forest_confusion_matrix.png"),
        ("Logistic Regression confusion matrix", "logistic_regression_confusion_matrix.png"),
        ("Overall model comparison", "model_comparison.png"),
        ("Incident class distribution", "class_distribution.png"),
        ("Random Forest top features", "rf_feature_importance.png"),
    ]

    for i, (label, filename) in enumerate(image_files):
        path = OUTPUT_DIR / filename
        if path.exists():
            with image_cols[i % 2]:
                st.caption(label)
                st.image(str(path), width="stretch")

    report_path = OUTPUT_DIR / "model_reports.txt"
    if report_path.exists():
        with st.expander("View classification report"):
            st.text(report_path.read_text(encoding="utf-8", errors="replace"))

elif page == "Live Prediction":
    st.markdown("### Live Prediction (Frontend → API → Model)")
    st.write(
        "Pick a record from the test file. The dashboard sends it to the "
        "backend API, which returns the predicted incident grade."
    )

    test_path = BASE_DIR / "dataset" / "GUIDE_Test.csv"

    if not api_online:
        st.error(
            "Backend API is offline. Open a new terminal and run: "
            "python -m uvicorn api:app --port 8000"
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
            st.dataframe(record_df.astype(str).T, width="stretch")

        if st.button("Predict using API", type="primary"):
            payload = {"record": json.loads(record_df.to_json(orient="records"))[0]}
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()

                st.metric("Predicted incident grade", result["prediction"])

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
                st.plotly_chart(fig, width="stretch")

                if "IncidentGrade" in record_df.columns:
                    actual = record_df["IncidentGrade"].iloc[0]
                    if pd.notna(actual):
                        st.caption(f"Actual grade in the test file: {actual}")

                st.caption(
                    "This is a demo prediction for one record, "
                    "not a performance evaluation."
                )
            except Exception as error:
                st.error(f"API request failed: {error}")

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
                st.plotly_chart(fig, width="stretch")

            st.dataframe(sample, width="stretch", hide_index=True)
            st.caption(
                "This is only a small preview of the training file, "
                "not a live security feed."
            )
    else:
        st.error("GUIDE_Train.csv was not found in the dataset folder.")

elif page == "About Project":
    st.markdown("### About CyberShield")
    st.write(
        "Cybersecurity Incident Severity Prediction System is a student "
        "machine-learning project using Microsoft's GUIDE incident dataset."
    )

    st.markdown("#### Architecture")
    st.markdown("""
    - **Frontend:** Streamlit dashboard (`app.py`)
    - **Backend:** FastAPI service (`api.py`) with endpoints `/health`, `/metrics`, `/predict`
    - **ML layer:** scikit-learn pipelines saved with joblib (`main.py` trains them)
    - The frontend sends a record as JSON to `/predict`; the backend loads the saved model and returns the predicted grade and class probabilities.
    """)

    st.markdown("#### Models")
    st.markdown("""
    - **Random Forest:** main trained model
    - **Logistic Regression:** baseline comparison model
    """)

    st.markdown("#### Prediction labels")
    st.markdown("""
    - `BenignPositive`
    - `FalsePositive`
    - `TruePositive`
    """)

    st.warning(
        "This is an academic prototype. Its predictions are decision-support "
        "outputs, not proof that an incident is malicious or harmless."
    )

st.divider()
st.markdown(
    "<p class='small-note'>CyberShield Academic Prototype · "
    "Machine Learning Incident Classification</p>",
    unsafe_allow_html=True
)