
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Maintenance | BusGuard",
    page_icon="B",
    layout="wide",
)

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "full_run.csv"
)

CLASS_ORDER = ["loose", "mid", "tight"]

CLASS_DISPLAY = {
    "loose": "Loose",
    "mid": "Intermediate",
    "tight": "Tight",
}

NAVY = "#17324D"
TEAL = "#16877B"
MUTED = "#63758A"
GRID = "#E3E9EF"
BACKGROUND = "#F4F7FA"
WHITE = "#FFFFFF"

CLASS_COLORS = {
    "loose": "#B65D5D",
    "mid": "#B18A43",
    "tight": "#16877B",
}

ELECTRICAL = ["I", "vdrop_mV", "R_mohm"]
THERMAL = ["t_max", "t_mean", "delta_T"]
ACOUSTIC = ["ac_100", "ac_band"]

FEATURES = ELECTRICAL + THERMAL + ACOUSTIC


# =========================================================
# PAGE STYLING
# =========================================================

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {BACKGROUND};
        }}

        [data-testid="stSidebar"] {{
            background: {WHITE};
            border-right: 1px solid {GRID};
        }}

        [data-testid="stMainBlockContainer"] {{
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }}

        h1, h2, h3 {{
            color: {NAVY};
            letter-spacing: -0.025em;
        }}

        h1 {{
            font-weight: 700;
        }}

        h2, h3 {{
            font-weight: 650;
        }}

        p, li, label {{
            color: {NAVY};
        }}

        [data-testid="stMetric"] {{
            background: {WHITE};
            border: 1px solid {GRID};
            border-radius: 12px;
            padding: 17px 19px;
            min-height: 105px;
        }}

        [data-testid="stMetricLabel"] {{
            color: {MUTED};
            font-size: 0.83rem;
        }}

        [data-testid="stMetricValue"] {{
            color: {NAVY};
            font-weight: 700;
        }}

        [data-testid="stExpander"] {{
            background: {WHITE};
            border: 1px solid {GRID};
            border-radius: 10px;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {GRID};
            border-radius: 10px;
            overflow: hidden;
        }}

        div[data-testid="stDownloadButton"] button {{
            background: {NAVY};
            color: {WHITE};
            border: 1px solid {NAVY};
            border-radius: 7px;
            font-weight: 600;
            min-height: 2.7rem;
        }}

        div[data-testid="stDownloadButton"] button:hover {{
            background: {TEAL};
            border-color: {TEAL};
            color: {WHITE};
        }}

        .eyebrow {{
            color: {TEAL};
            font-size: 0.74rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }}

        .page-subtitle {{
            color: {MUTED};
            font-size: 0.95rem;
            line-height: 1.6;
            max-width: 850px;
        }}

        .section-note {{
            color: {MUTED};
            font-size: 0.84rem;
            line-height: 1.55;
            margin-top: -0.35rem;
            margin-bottom: 1rem;
        }}

        .status-chip {{
            display: inline-block;
            padding: 5px 10px;
            border: 1px solid #CDE5E1;
            border-radius: 5px;
            background: #EDF7F5;
            color: {TEAL};
            font-size: 0.71rem;
            font-weight: 750;
            letter-spacing: 0.07em;
        }}

        .technical-card {{
            background: {WHITE};
            border: 1px solid {GRID};
            border-radius: 12px;
            padding: 18px 20px;
            height: 100%;
        }}

        .technical-card-title {{
            color: {MUTED};
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .technical-card-value {{
            color: {NAVY};
            font-size: 1.14rem;
            font-weight: 700;
            line-height: 1.45;
        }}

        .technical-card-copy {{
            color: {MUTED};
            font-size: 0.82rem;
            line-height: 1.55;
            margin-top: 7px;
        }}

        .status-heading {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {NAVY};
            margin-bottom: 0.4rem;
        }}

        .status-copy {{
            font-size: 0.88rem;
            line-height: 1.6;
            color: {MUTED};
        }}

        .small-label {{
            font-size: 0.75rem;
            color: {MUTED};
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 700;
        }}

        hr {{
            border-color: {GRID};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    required = [
        "t",
        "I",
        "vdrop_mV",
        "R_mohm",
        "t_max",
        "t_mean",
        "t_amb",
        "ac_100",
        "ac_band",
        "label",
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[required].isnull().any().any():
        raise ValueError("Missing values found in required columns.")

    if set(data["label"].unique()) != set(CLASS_ORDER):
        raise ValueError("Expected labels: loose, mid and tight.")

    numeric_columns = [
        "t",
        "I",
        "vdrop_mV",
        "R_mohm",
        "t_max",
        "t_mean",
        "t_amb",
        "ac_100",
        "ac_band",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if not np.isfinite(
        data[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite numeric values found in the dataset.")

    data = data.sort_values("t").reset_index(drop=True)
    data["delta_T"] = data["t_max"] - data["t_amb"]

    return data


# =========================================================
# HELD-OUT PREDICTIONS
# =========================================================

@st.cache_data
def get_oof_predictions(data):
    """
    Generate out-of-fold predictions using grouped cross-validation.

    Groups contain five consecutive observations. Each observation is
    predicted by a model that did not train on its group.
    """
    data = data.reset_index(drop=True)
    groups = np.arange(len(data)) // 5

    if len(np.unique(groups)) < 5:
        raise ValueError("At least five groups are required.")

    cv = GroupKFold(n_splits=5)

    predictions = np.empty(len(data), dtype=object)
    probabilities = np.zeros(
        (len(data), len(CLASS_ORDER)),
        dtype=float,
    )
    fold_numbers = np.zeros(len(data), dtype=int)

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(data, data["label"], groups=groups),
        start=1,
    ):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
        )

        model.fit(
            train_data[FEATURES],
            train_data["label"],
        )

        predictions[test_idx] = model.predict(
            test_data[FEATURES]
        )

        fold_probabilities = model.predict_proba(
            test_data[FEATURES]
        )

        for class_position, class_name in enumerate(model.classes_):
            probabilities[
                test_idx,
                CLASS_ORDER.index(class_name),
            ] = fold_probabilities[:, class_position]

        fold_numbers[test_idx] = fold_number

    return predictions, probabilities, fold_numbers


# =========================================================
# RECOMMENDATION LOGIC
# =========================================================

def get_recommendation(predicted_label):
    if predicted_label == "loose":
        return {
            "title": "Further inspection recommended",
            "summary": (
                "The model classified this observation as loose. "
                "Prioritize review by qualified electrical maintenance personnel."
            ),
            "actions": [
                "Review the recorded resistance and voltage-drop readings.",
                "Compare with the equipment's approved limits and historical baseline.",
                "Arrange an inspection under the site's approved electrical safety procedure.",
                (
                    "Do not touch or retighten an energized joint. Any physical "
                    "inspection must be performed by qualified personnel using "
                    "the required isolation and safety procedures."
                ),
            ],
        }

    if predicted_label == "mid":
        return {
            "title": "Condition requires review",
            "summary": (
                "The model classified this observation as intermediate. "
                "Review the measurements and trend before deciding on maintenance."
            ),
            "actions": [
                "Compare resistance and voltage drop with previous readings.",
                "Review temperature rise above ambient and the acoustic measurements.",
                "Check the manufacturer's limits and the site's maintenance procedure.",
                "Escalate for qualified inspection if readings are abnormal or worsening.",
            ],
        }

    return {
        "title": "Reference-like model classification",
        "summary": (
            "The model classified this observation as tight. "
            "This is a reference-like classification, not proof that the joint is safe."
        ),
        "actions": [
            "Retain the readings as a comparison point for future monitoring.",
            "Continue the established inspection and maintenance schedule.",
            "Investigate any abnormal trend or deviation from approved equipment limits.",
        ],
    }


# =========================================================
# PAGE HEADER
# =========================================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        '<div class="eyebrow">Decision support / 04</div>',
        unsafe_allow_html=True,
    )

    st.title("Maintenance & Recommendations")

    st.markdown(
        """
        <div class="page-subtitle">
            Review a recorded observation, inspect its held-out model
            prediction, and examine the corresponding prototype
            maintenance guidance.
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right; padding-top:12px;">
            <span class="status-chip">PROTOTYPE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =========================================================
# LOAD DATA AND PREDICTIONS
# =========================================================

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))

    (
        oof_predictions,
        oof_probabilities,
        fold_numbers,
    ) = get_oof_predictions(df)

except Exception as exc:
    st.error(f"Unable to prepare assessment: {exc}")
    st.stop()


# =========================================================
# OBSERVATION SELECTOR
# =========================================================

st.subheader("Select recorded observation")

st.markdown(
    """
    <div class="section-note">
        Choose a dataset observation to inspect its measurements,
        held-out prediction, and recommended follow-up.
    </div>
    """,
    unsafe_allow_html=True,
)

selected_index = st.selectbox(
    "Observation",
    options=list(range(len(df))),
    format_func=lambda index: (
        f"Observation {index + 1:02d}  |  "
        f"Time: {df.iloc[index]['t']}  |  "
        f"Reference: {CLASS_DISPLAY[str(df.iloc[index]['label'])]}"
    ),
)

row = df.iloc[selected_index]

predicted_label = str(oof_predictions[selected_index])
reference_label = str(row["label"])
fold_number = int(fold_numbers[selected_index])
class_probabilities = oof_probabilities[selected_index]

jhi = (
    class_probabilities[CLASS_ORDER.index("tight")] * 100
    + class_probabilities[CLASS_ORDER.index("mid")] * 50
)

recommendation = get_recommendation(predicted_label)


# =========================================================
# ASSESSMENT SUMMARY
# =========================================================

st.divider()

st.subheader("Assessment summary")

st.markdown(
    """
    <div class="section-note">
        The predicted condition and recorded reference label are shown
        separately so that model output is not confused with the original label.
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Predicted condition",
    CLASS_DISPLAY[predicted_label].upper(),
    help="Prediction from the model that held this observation out of training.",
)

c2.metric(
    "Recorded reference label",
    CLASS_DISPLAY[reference_label].upper(),
    help="Condition label recorded in the dataset.",
)

c3.metric(
    "Validation fold",
    str(fold_number),
    help="Grouped validation fold that generated this observation's prediction.",
)

matches_reference = predicted_label == reference_label

if matches_reference:
    st.success(
        "The held-out prediction matches the recorded reference label."
    )
else:
    st.warning(
        "The held-out prediction differs from the recorded reference label. "
        "Review the measurements and model limitations before interpreting it."
    )

st.caption(
    "This is an offline assessment of a recorded observation, not a live "
    "sensor connection. A model prediction is not a certified safety determination."
)


# =========================================================
# MEASURED VALUES
# =========================================================

st.divider()

st.subheader("Recorded sensor measurements")

st.markdown(
    '<div class="section-note">'
    "Measured values associated with the selected dataset observation."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("**Electrical measurements**")

e1, e2, e3 = st.columns(3)

e1.metric("Current", f"{row['I']:.3f} A")
e2.metric("Voltage drop", f"{row['vdrop_mV']:.3f} mV")
e3.metric("Joint resistance", f"{row['R_mohm']:.3f} mΩ")

st.write("")

st.markdown("**Thermal measurements**")

t1, t2, t3 = st.columns(3)

t1.metric("Maximum temperature", f"{row['t_max']:.3f} °C")
t2.metric("Mean temperature", f"{row['t_mean']:.3f} °C")
t3.metric("Temperature rise", f"{row['delta_T']:.3f} °C")

st.write("")

st.markdown("**Acoustic features**")

a1, a2 = st.columns(2)

a1.metric("AC feature 100", f"{row['ac_100']:.4f}")
a2.metric("AC band feature", f"{row['ac_band']:.4f}")

st.caption(
    "Acoustic feature units were not specified in the dataset; values are "
    "displayed without assigning physical units."
)


# =========================================================
# MODEL PROBABILITIES
# =========================================================

st.divider()

st.subheader("Model output")

st.markdown(
    """
    <div class="section-note">
        Probability outputs from the fold-specific Random Forest model.
        These describe the model's classification output, not a calibrated
        estimate of real-world failure risk.
    </div>
    """,
    unsafe_allow_html=True,
)

probability_table = pd.DataFrame(
    {
        "Condition class": [
            CLASS_DISPLAY[name]
            for name in CLASS_ORDER
        ],
        "Model probability": [
            float(class_probabilities[i])
            for i in range(len(CLASS_ORDER))
        ],
    }
)

probability_columns = st.columns(3)

for index, column in enumerate(probability_columns):
    label = CLASS_ORDER[index]
    probability = float(class_probabilities[index])

    with column:
        st.markdown(
            f"""
            <div class="technical-card">
                <div class="technical-card-title">
                    {CLASS_DISPLAY[label]} class
                </div>
                <div class="technical-card-value">
                    {probability:.1%}
                </div>
                <div style="
                    margin-top:12px;
                    height:5px;
                    background:{GRID};
                    border-radius:4px;
                    overflow:hidden;
                ">
                    <div style="
                        width:{probability * 100:.2f}%;
                        height:5px;
                        background:{CLASS_COLORS[label]};
                        border-radius:4px;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("View probability table"):
    st.dataframe(
        probability_table.style.format(
            {"Model probability": "{:.1%}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Probabilities are produced by the fold-specific Random Forest. "
    "They are not guaranteed to be calibrated estimates of real-world risk."
)


# =========================================================
# PROTOTYPE JOINT HEALTH INDEX
# =========================================================

st.divider()

st.subheader("Prototype Joint Health Index")

st.markdown(
    """
    <div class="section-note">
        An illustrative score derived from the three model probabilities.
        It is not an industry-standard health or safety index.
    </div>
    """,
    unsafe_allow_html=True,
)

jhi_left, jhi_right = st.columns([1, 2])

with jhi_left:
    st.metric(
        "Prototype JHI",
        f"{jhi:.1f} / 100",
    )

    st.progress(
        float(np.clip(jhi / 100, 0, 1))
    )

with jhi_right:
    st.markdown(
        f"""
        <div class="technical-card">
            <div class="technical-card-title">Score calculation</div>
            <div class="technical-card-value">
                JHI = 100 × P(tight) + 50 × P(mid)
            </div>
            <div class="technical-card-copy">
                The loose class contributes zero points. The resulting score
                ranges from 0 to 100 and summarizes the model's class
                probabilities using a prototype weighting scheme.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "A higher JHI does not establish that a joint is safe. Do not use this "
    "prototype score as a substitute for approved equipment limits, qualified "
    "inspection, or site safety procedures."
)


# =========================================================
# MAINTENANCE GUIDANCE
# =========================================================

st.divider()

st.subheader("Recommended inspection action")

st.markdown(
    """
    <div class="section-note">
        Rule-based prototype guidance associated with the predicted class.
    </div>
    """,
    unsafe_allow_html=True,
)

rec_left, rec_right = st.columns([2, 1])

with rec_left:
    st.markdown(
        f"""
        <div class="technical-card">
            <div class="technical-card-title">Recommended follow-up</div>
            <div class="technical-card-value">
                {recommendation['title']}
            </div>
            <div class="technical-card-copy">
                {recommendation['summary']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with rec_right:
    st.markdown(
        f"""
        <div class="technical-card">
            <div class="technical-card-title">Assessment basis</div>
            <div class="technical-card-value">
                {CLASS_DISPLAY[predicted_label]}
            </div>
            <div class="technical-card-copy">
                Guidance is selected from the predicted condition class.
                It does not establish a verified fault or safety state.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

st.markdown("**Suggested review steps**")

for action in recommendation["actions"]:
    st.markdown(f"- {action}")

st.caption(
    "No manufacturer thresholds or certified maintenance limits are inferred "
    "from this dataset. Follow the equipment manufacturer's instructions and "
    "the site's approved electrical safety procedures."
)


# =========================================================
# DOWNLOADABLE ASSESSMENT REPORT
# =========================================================

st.divider()

st.subheader("Export assessment")

st.markdown(
    """
    <div class="section-note">
        Export the selected observation's measurements, held-out prediction,
        class probabilities, prototype JHI, and recommendation as a CSV file.
    </div>
    """,
    unsafe_allow_html=True,
)

report = pd.DataFrame(
    [
        {
            "Observation_number": selected_index + 1,
            "Recorded_time": row["t"],
            "Predicted_condition": predicted_label,
            "Recorded_reference_label": reference_label,
            "Prediction_matches_reference": matches_reference,
            "Validation_fold": fold_number,
            "Current_A": row["I"],
            "Voltage_drop_mV": row["vdrop_mV"],
            "Resistance_mohm": row["R_mohm"],
            "Maximum_temperature_C": row["t_max"],
            "Mean_temperature_C": row["t_mean"],
            "Temperature_rise_C": row["delta_T"],
            "Acoustic_feature_100": row["ac_100"],
            "Acoustic_band_feature": row["ac_band"],
            "Probability_loose": class_probabilities[0],
            "Probability_mid": class_probabilities[1],
            "Probability_tight": class_probabilities[2],
            "Prototype_JHI": jhi,
            "Recommended_action": recommendation["title"],
            "Assessment_type": (
                "Offline prototype; not a certified safety assessment"
            ),
        }
    ]
)

csv_buffer = StringIO()
report.to_csv(csv_buffer, index=False)

download_col, info_col = st.columns([1, 2])

with download_col:
    st.download_button(
        label="Download assessment report (CSV)",
        data=csv_buffer.getvalue(),
        file_name=(
            f"busguard_assessment_observation_{selected_index + 1}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

with info_col:
    st.markdown(
        """
        <div class="technical-card">
            <div class="technical-card-title">Report contents</div>
            <div class="technical-card-copy">
                Recorded sensor measurements, held-out model prediction,
                reference-label comparison, validation fold, class
                probabilities, prototype JHI and suggested follow-up.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        display:flex;
        justify-content:space-between;
        gap:16px;
        flex-wrap:wrap;
        color:#63758A;
        font-size:0.78rem;
        padding:4px 0 12px 0;
    ">
        <span>BUSGUARD AI · MAINTENANCE SUPPORT</span>
        <span>Offline prototype · Not a certified safety system</span>
    </div>
    """,
    unsafe_allow_html=True,
)

