import json
from html import escape
from pathlib import Path

import streamlit as st

from src.predictor import predict_food_waste

BASE_DIR = Path(__file__).resolve().parent
STYLE_PATH = BASE_DIR / "assets" / "styles.css"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
TREE_IMAGE_PATH = BASE_DIR / "images" / "decision_tree.png"

st.set_page_config(
    page_title="Food Waste Prediction",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    if STYLE_PATH.exists():
        css = STYLE_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def load_metadata():
    if not METADATA_PATH.exists():
        return {}

    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def show_prediction_page():
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Machine Learning Project</div>
          <h1>Food Waste Prediction</h1>
          <p>
            Enter the current inventory condition. The app calculates the remaining
            stock and uses the trained Decision Tree model to classify the input as
            Safe or Potential Waste.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Inventory Input")

        with st.form("prediction_form", clear_on_submit=False):
            product_name = st.text_input(
                "Product name",
                placeholder="Example: Bread",
                help="This is only used to label the result and is not a model feature.",
            )

            col1, col2 = st.columns(2)
            with col1:
                stock = st.number_input(
                    "Available stock",
                    min_value=1,
                    value=100,
                    step=1,
                )
            with col2:
                sold = st.number_input(
                    "Items sold",
                    min_value=0,
                    value=40,
                    step=1,
                )

            col3, col4 = st.columns(2)
            with col3:
                day_index = st.number_input(
                    "Day index",
                    min_value=1,
                    max_value=30,
                    value=15,
                    step=1,
                    help="Matches the hari_ke feature in the supplied dataset (1-30).",
                )
            with col4:
                weather_display = st.selectbox(
                    "Weather code",
                    ["Code 0", "Code 1"],
                    help=(
                        "The supplied project files only document this feature as 0/1. "
                        "The real-world meaning of each code is not defined."
                    ),
                )

            special_day = st.toggle(
                "Holiday or special day",
                value=False,
                help="Converted to 0 or 1 before prediction.",
            )

            submitted = st.form_submit_button("Predict", type="primary")

        st.markdown(
            """
            <div class="note-card">
              <h4>Calculated automatically</h4>
              <div class="small-muted">
                Remaining stock and remaining-stock percentage are calculated from
                available stock and items sold. Users do not need to upload the CSV dataset.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Prediction Result")

        if not submitted:
            st.markdown(
                """
                <div class="info-card">
                  <div class="small-muted">
                    Complete the input form and select <b>Predict</b> to see the result.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        if sold > stock:
            st.error("Items sold cannot be greater than available stock.")
            return

        weather_code = 0 if weather_display == "Code 0" else 1

        try:
            result = predict_food_waste(
                stock=int(stock),
                sold=int(sold),
                day_index=int(day_index),
                weather_code=weather_code,
                special_day=special_day,
            )
        except (FileNotFoundError, ValueError, KeyError) as error:
            st.error(str(error))
            return

        product_display = escape(product_name.strip() or "Current product")
        risk_class = "risk-high" if result["prediction"] == 1 else "risk-low"

        st.markdown(
            f"""
            <div class="result-card">
              <span class="{risk_class}">{result['risk_label']}</span>
              <div class="result-title">{product_display}</div>
              <p class="result-subtitle">Classification: <b>{result['status']}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Stock", result["stock"])
        metric2.metric("Sold", result["sold"])
        metric3.metric("Remaining", result["remaining"])

        st.metric(
            "Remaining stock percentage",
            f"{result['remaining_percentage']:.1f}%",
        )

        if result["confidence"] is not None:
            st.caption(
                f"Model confidence for this classification: {result['confidence']:.0%}. "
                "This value comes from the model and should not be treated as a "
                "guaranteed real-world probability of waste."
            )

        st.markdown(
            f"""
            <div class="recommendation-card">
              <h4>Simple Recommendation</h4>
              <div>{result['recommendation']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_model_page(metadata):
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Project Information</div>
          <h1>Model Information</h1>
          <p>
            This page shows the model used by the application and the main limitations
            of the current dataset.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="model-card">
              <div class="model-label">Model</div>
              <div class="model-value">{metadata.get('model', 'Decision Tree Classifier')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Dataset rows", metadata.get("dataset_rows", "—"))

    accuracy = metadata.get("accuracy")
    with col3:
        st.metric(
            "Test accuracy",
            f"{accuracy:.0%}" if isinstance(accuracy, (int, float)) else "—",
        )

    st.subheader("Features Used")
    st.code(
        "stok, terjual, hari_ke, cuaca, hari_besar, sisa_persen",
        language="text",
    )

    st.subheader("Current Limitations")
    st.markdown(
        """
        - The dataset contains only 80 rows, so the evaluation result is based on a small sample.
        - The original source of the dataset is not documented in the supplied project files.
        - The meaning of weather codes `0` and `1` is not documented, so the app keeps the raw codes.
        - `sisa_persen` is derived from stock and sold items, and the labels in the current dataset follow a very simple remaining-stock pattern.
        - The model output is suitable for an academic project demonstration, not for real supermarket operational decisions.
        """
    )

    if TREE_IMAGE_PATH.exists():
        with st.expander("View Decision Tree"):
            st.image(str(TREE_IMAGE_PATH), use_container_width=True)


load_css()
metadata = load_metadata()

with st.sidebar:
    st.markdown("### Food Waste Prediction")
    page = st.radio(
        "Navigation",
        ["Prediction", "Model Information"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Academic machine learning project using a Decision Tree classifier.")

if page == "Prediction":
    show_prediction_page()
else:
    show_model_page(metadata)
