from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import streamlit as st

from src.predictor import predict_food_waste

BASE_DIR = Path(__file__).resolve().parent
STYLE_PATH = BASE_DIR / "assets" / "styles.css"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

st.set_page_config(
    page_title="AI Food Waste Prediction",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


load_css()
metadata = load_metadata()

with st.sidebar:
    st.markdown("### Food Waste AI")
    page = st.radio("Navigation", ["Prediction", "Model Information"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Manual input only — users do not need to upload a CSV file.")

if page == "Prediction":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">AI-based decision support</div>
          <h1>Food Waste Prediction</h1>
          <p>Enter the current inventory condition. The application calculates the remaining stock automatically and uses the trained model to classify potential food waste.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.08, .92], gap="large")

    with left:
        st.subheader("Inventory condition")
        with st.form("prediction_form", clear_on_submit=False):
            product_name = st.text_input(
                "Product name",
                placeholder="e.g. Bread",
                help="Used only to identify the result; it is not a model feature.",
            )

            c1, c2 = st.columns(2)
            with c1:
                stock = st.number_input("Available stock", min_value=1, value=100, step=1)
            with c2:
                sold = st.number_input("Items sold", min_value=0, value=40, step=1)

            c3, c4 = st.columns(2)
            with c3:
                selected_date = st.date_input("Date", value=date.today())
            with c4:
                weather_display = st.selectbox(
                    "Weather condition",
                    ["Condition 0", "Condition 1"],
                    help=(
                        "The supplied dataset only documents weather as 0/1 and does not "
                        "define the real-world meaning of each code."
                    ),
                )

            special_day = st.toggle(
                "Holiday or special day",
                value=False,
                help="Converted automatically to the 0/1 format used by the model.",
            )

            submitted = st.form_submit_button("Analyze waste risk", type="primary")

        st.markdown(
            """
            <div class="note-card">
              <h4>What the user does not need to enter</h4>
              <div class="small-muted">Remaining stock and remaining-stock percentage are calculated automatically from stock and items sold. No CSV upload or Python editing is required.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Analysis result")

        if not submitted:
            st.markdown(
                """
                <div class="info-card">
                  <div class="small-muted">Your prediction will appear here after you complete the form and select <b>Analyze waste risk</b>.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if sold > stock:
                st.error("Items sold cannot be greater than the available stock.")
            else:
                weather_code = 0 if weather_display == "Condition 0" else 1
                try:
                    result = predict_food_waste(
                        stock=int(stock),
                        sold=int(sold),
                        day_index=int(selected_date.day),
                        weather_code=weather_code,
                        special_day=special_day,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    product_display = product_name.strip() or "Current product"
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
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Stock", f"{result['stock']}")
                    m2.metric("Sold", f"{result['sold']}")
                    m3.metric("Remaining", f"{result['remaining']}")

                    st.metric("Remaining stock percentage", f"{result['remaining_percentage']:.1f}%")

                    if result["confidence"] is not None:
                        st.caption(
                            f"Model confidence for this classification: {result['confidence']:.0%}. "
                            "This is model output, not a guaranteed real-world probability of waste."
                        )

                    st.markdown(
                        f"""
                        <div class="recommendation-card">
                          <h4>Recommendation</h4>
                          <div>{result['recommendation']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

else:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Project transparency</div>
          <h1>Model Information</h1>
          <p>This page explains what is currently behind the web interface and the main limitations of the project.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="model-card">
                <div class="model-label">Model</div>
                <div class="model-value">{metadata.get("model", "Decision Tree Classifier")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.metric("Dataset rows", metadata.get("dataset_rows", "—"))

    accuracy = metadata.get("accuracy")
    with c3:
        st.metric(
            "Test accuracy",
            f"{accuracy:.0%}" if isinstance(accuracy, (float, int)) else "—",
        )

    st.subheader("Features used by the current model")
    st.code("stok, terjual, hari_ke, cuaca, hari_besar, sisa_persen", language="text")

    st.subheader("Important limitations")
    st.markdown(
        """
        - The supplied dataset is simulated and small, so the result should be treated as a project prototype rather than a production decision system.
        - The dataset documents `cuaca` only as numeric codes `0` and `1`; their real-world meanings are not defined in the project files.
        - `sisa_persen` is calculated automatically from stock and sold items in the app, so users do not need to provide it manually.
        - The current model remains the original Decision Tree approach; the web app upgrades how the model is used, not the underlying dataset quality.
        """
    )

    if (BASE_DIR / "images" / "decision_tree.png").exists():
        with st.expander("View Decision Tree visualization"):
            st.image(str(BASE_DIR / "images" / "decision_tree.png"), use_container_width=True)
