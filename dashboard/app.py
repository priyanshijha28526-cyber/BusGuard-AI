
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BusGuard | Overview",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# VISUAL DESIGN
# =========================================================

NAVY = "#17324D"
TEAL = "#16877B"
MUTED = "#63758A"
GRID = "#E3E9EF"
BACKGROUND = "#F4F7FA"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND};
    }}

    [data-testid="stHeader"] {{
        background: rgba(244, 247, 250, 0.94);
    }}

    [data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #E3E9EF;
    }}

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}

    .bg-eyebrow {{
        color: {TEAL};
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }}

    .bg-title {{
        color: {NAVY};
        font-size: 2.25rem;
        font-weight: 750;
        letter-spacing: -0.045em;
        line-height: 1.15;
        margin: 0;
    }}

    .bg-subtitle {{
        color: {MUTED};
        font-size: 0.96rem;
        margin-top: 0.6rem;
        line-height: 1.6;
    }}

    .bg-tag {{
        display: inline-block;
        color: {NAVY};
        background: #E8EEF4;
        border: 1px solid #D8E2EC;
        border-radius: 5px;
        padding: 0.35rem 0.55rem;
        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.09em;
    }}

    .bg-section {{
        color: {NAVY};
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        margin-bottom: 0.15rem;
    }}

    .bg-section-note {{
        color: {MUTED};
        font-size: 0.83rem;
        margin-bottom: 1rem;
    }}

    .bg-panel {{
        background: #FFFFFF;
        border: 1px solid #E1E8EF;
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }}

    [data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #E1E8EF;
        border-radius: 10px;
        padding: 1rem 1rem 0.9rem 1rem;
        min-height: 112px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED};
        font-size: 0.82rem;
    }}

    [data-testid="stMetricValue"] {{
        color: {NAVY};
        font-weight: 750;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 0.78rem;
    }}

    div[data-testid="stAlert"] {{
        border-radius: 9px;
    }}

    div[data-testid="stExpander"] {{
        background: #FFFFFF;
        border: 1px solid #E1E8EF;
        border-radius: 10px;
    }}

    div[data-testid="stDownloadButton"] button {{
        border-radius: 7px;
    }}

    h1, h2, h3 {{
        color: {NAVY};
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

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "full_run.csv"
)


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    required = [
        "t", "I", "vdrop_mV", "R_mohm",
        "t_max", "t_mean", "t_amb",
        "ac_100", "ac_band", "label",
    ]

    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[required].isnull().any().any():
        raise ValueError("Missing values found in the dataset.")

    data = data.sort_values("t").reset_index(drop=True)
    data["delta_T"] = data["t_max"] - data["t_amb"]

    if set(data["label"].unique()) != {"loose", "mid", "tight"}:
        raise ValueError("Dataset must contain loose, mid and tight labels.")

    return data


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()


# =========================================================
# MODEL
# =========================================================

features = [
    "I",
    "vdrop_mV",
    "R_mohm",
    "t_max",
    "t_mean",
    "delta_T",
    "ac_100",
    "ac_band",
]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    random_state=42,
)

model.fit(df[features], df["label"])


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.markdown(
    f"""
    <div style="padding:0.4rem 0 0.7rem 0;">
        <div style="font-size:1.35rem;font-weight:800;
                    letter-spacing:0.04em;color:{NAVY};">
            BUSGUARD
        </div>
        <div style="font-size:0.72rem;color:{MUTED};
                    letter-spacing:0.08em;margin-top:0.2rem;">
            PROTOTYPE CONSOLE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

st.sidebar.markdown("#### Observation controls")
st.sidebar.caption("Explore a recorded dataset observation.")

index = st.sidebar.slider(
    "Measurement index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    help="Select a recorded observation from the dataset.",
)

st.sidebar.markdown(
    f"""
    <div style="border:1px solid #E1E8EF;border-radius:9px;
                padding:0.8rem;background:#F7F9FB;margin-top:0.5rem;">
        <div style="font-size:0.7rem;color:{MUTED};
                    text-transform:uppercase;letter-spacing:0.08em;">
            Selected record
        </div>
        <div style="font-size:1.15rem;font-weight:750;
                    color:{NAVY};margin-top:0.25rem;">
            {index + 1:02d} / {len(df):02d}
        </div>
        <div style="font-size:0.8rem;color:{MUTED};margin-top:0.2rem;">
            Reference label: {df.iloc[index]["label"].upper()}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()
st.sidebar.caption(
    "Offline dataset playback. No live equipment connection is active."
)


# =========================================================
# SELECTED OBSERVATION AND PROTOTYPE ESTIMATE
# =========================================================

row = df.iloc[index:index + 1]

probabilities = model.predict_proba(row[features])[0]
class_prob = dict(zip(model.classes_, probabilities))

predicted_condition = model.predict(row[features])[0]

p_loose = float(class_prob.get("loose", 0))
p_mid = float(class_prob.get("mid", 0))
p_tight = float(class_prob.get("tight", 0))

health_index = p_tight * 100 + p_mid * 50

if health_index >= 70:
    status = "REFERENCE-LIKE"
    status_detail = (
        "The prototype score falls in the reference-like range. "
        "This does not establish that the joint is safe."
    )
    status_color = TEAL
elif health_index >= 30:
    status = "INTERMEDIATE"
    status_detail = (
        "The prototype score falls in the intermediate range. "
        "Review the measurements and compare them with approved limits."
    )
    status_color = "#9A6A16"
else:
    status = "DEGRADED"
    status_detail = (
        "The prototype score falls in the degraded range. "
        "Prioritize review by qualified electrical maintenance personnel."
    )
    status_color = "#B34D4D"


# =========================================================
# PAGE HEADER
# =========================================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        '<div class="bg-eyebrow">Condition intelligence / 01</div>'
        '<div class="bg-title">Busbar joint overview</div>'
        '<div class="bg-subtitle">'
        'Multi-sensor analysis of recorded electrical, thermal and acoustic '
        'signatures for prototype joint-condition assessment.'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        f"""
        <div style="text-align:right;padding-top:0.35rem;">
            <span class="bg-tag">EXPERIMENTAL PROTOTYPE</span>
            <div style="font-size:0.78rem;color:{MUTED};
                        margin-top:0.7rem;">
                {len(df)} recorded observations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# PRIMARY CONDITION SUMMARY
# =========================================================

st.markdown(
    '<div class="bg-section">Condition summary</div>'
    '<div class="bg-section-note">'
    'Prototype model output for the selected recorded observation.'
    '</div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns([1.1, 1.1, 0.9])

with m1:
    st.metric(
        "Prototype Joint Health Index",
        f"{health_index:.1f} / 100",
        help=(
            "Illustrative score: 100 × P(tight) + 50 × P(mid). "
            "Not an industry-standard safety index."
        ),
    )

with m2:
    st.metric(
        "Predicted condition",
        predicted_condition.upper(),
        help=(
            "This is an in-sample estimate because this model was trained "
            "on the full dataset, including the selected observation."
        ),
    )

with m3:
    st.metric(
        "Recorded current",
        f"{row['I'].iloc[0]:.2f} A",
    )

st.markdown(
    f"""
    <div style="background:#FFFFFF;border:1px solid #E1E8EF;
                border-left:4px solid {status_color};
                border-radius:9px;padding:1rem 1.1rem;margin-top:0.4rem;">
        <div style="font-size:0.7rem;color:{MUTED};
                    text-transform:uppercase;letter-spacing:0.1em;">
            Prototype score category
        </div>
        <div style="font-size:1.15rem;font-weight:800;
                    color:{status_color};margin:0.25rem 0 0.35rem 0;">
            {status}
        </div>
        <div style="font-size:0.88rem;color:{MUTED};line-height:1.55;">
            {status_detail}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# SENSOR MEASUREMENTS
# =========================================================

st.markdown(
    '<div class="bg-section">Sensor measurements</div>'
    '<div class="bg-section-note">'
    'Recorded values for the selected observation.'
    '</div>',
    unsafe_allow_html=True,
)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric(
        "Joint resistance",
        f"{row['R_mohm'].iloc[0]:.3f} mΩ",
    )

with s2:
    st.metric(
        "Voltage drop",
        f"{row['vdrop_mV'].iloc[0]:.2f} mV",
    )

with s3:
    st.metric(
        "Maximum temperature",
        f"{row['t_max'].iloc[0]:.2f} °C",
    )

with s4:
    st.metric(
        "Acoustic band",
        f"{row['ac_band'].iloc[0]:.3f}",
    )

with st.expander("View all recorded sensor values"):
    all_readings = pd.DataFrame(
        {
            "Measurement": [
                "Current",
                "Joint resistance",
                "Voltage drop",
                "Maximum temperature",
                "Mean temperature",
                "Ambient temperature",
                "Temperature rise",
                "Acoustic feature 100",
                "Acoustic band feature",
            ],
            "Value": [
                f"{row['I'].iloc[0]:.3f} A",
                f"{row['R_mohm'].iloc[0]:.3f} mΩ",
                f"{row['vdrop_mV'].iloc[0]:.3f} mV",
                f"{row['t_max'].iloc[0]:.3f} °C",
                f"{row['t_mean'].iloc[0]:.3f} °C",
                f"{row['t_amb'].iloc[0]:.3f} °C",
                f"{row['delta_T'].iloc[0]:.3f} °C",
                f"{row['ac_100'].iloc[0]:.4f}",
                f"{row['ac_band'].iloc[0]:.4f}",
            ],
        }
    )

    st.dataframe(
        all_readings,
        hide_index=True,
        use_container_width=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# JOINT HEALTH TREND
# =========================================================

st.markdown(
    '<div class="bg-section">Joint health trend</div>'
    '<div class="bg-section-note">'
    'Prototype score across the recorded observations; not a live time-series feed.'
    '</div>',
    unsafe_allow_html=True,
)

health_values = []

for _, observation in df.iterrows():
    observation_X = observation[features].to_frame().T
    probabilities_i = model.predict_proba(observation_X)[0]
    class_prob_i = dict(zip(model.classes_, probabilities_i))

    health_values.append(
        class_prob_i.get("tight", 0) * 100
        + class_prob_i.get("mid", 0) * 50
    )

fig, ax = plt.subplots(figsize=(12, 4.1))

fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

ax.plot(
    df["t"],
    health_values,
    color=TEAL,
    linewidth=2.5,
    solid_capstyle="round",
)

ax.scatter(
    [df.iloc[index]["t"]],
    [health_values[index]],
    color=NAVY,
    s=55,
    zorder=4,
    edgecolor="#FFFFFF",
    linewidth=1.3,
)

ax.axhline(70, color="#A6B4C2", linewidth=1, linestyle="--")
ax.axhline(30, color="#A6B4C2", linewidth=1, linestyle="--")

ax.set_xlabel("Recorded time (s)", color=MUTED, labelpad=10)
ax.set_ylabel("Prototype JHI", color=MUTED, labelpad=10)
ax.set_ylim(-5, 105)

ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.grid(axis="x", visible=False)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.caption(
    "The plotted values use predictions from a model fitted on all observations. "
    "They are descriptive in-sample estimates, not independent validation results. "
    "Dashed lines mark the prototype score categories only."
)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# AI CONDITION PROBABILITIES
# =========================================================

st.markdown(
    '<div class="bg-section">AI condition probabilities</div>'
    '<div class="bg-section-note">'
    'Relative model output for the selected observation.'
    '</div>',
    unsafe_allow_html=True,
)

prob_left, prob_right = st.columns([1.5, 1])

with prob_left:
    probability_df = pd.DataFrame(
        {
            "Condition": ["Loose", "Mid", "Tight"],
            "Probability": [p_loose, p_mid, p_tight],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    bars = ax.barh(
        probability_df["Condition"],
        probability_df["Probability"],
        color=[MUTED, "#9AABB9", TEAL],
        height=0.52,
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Model probability", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_color(GRID)

    for bar, probability in zip(bars, probability_df["Probability"]):
        ax.text(
            min(probability + 0.025, 0.91),
            bar.get_y() + bar.get_height() / 2,
            f"{probability:.1%}",
            va="center",
            ha="left",
            color=NAVY,
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with prob_right:
    st.markdown(
        f"""
        <div class="bg-panel">
            <div style="font-size:0.72rem;color:{MUTED};
                        text-transform:uppercase;letter-spacing:0.08em;">
                Highest probability
            </div>
            <div style="font-size:1.45rem;font-weight:800;color:{NAVY};
                        margin-top:0.4rem;">
                {predicted_condition.upper()}
            </div>
            <div style="font-size:0.9rem;color:{MUTED};margin-top:0.25rem;">
                {max(p_loose, p_mid, p_tight):.1%} model probability
            </div>
            <hr style="border:0;border-top:1px solid {GRID};
                       margin:1rem 0;">
            <div style="font-size:0.8rem;color:{MUTED};line-height:1.6;">
                These probabilities describe the model's output on the
                selected record. They are not calibrated industrial risk
                estimates.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# PROTOTYPE RECOMMENDATION
# =========================================================

st.markdown(
    '<div class="bg-section">Inspection guidance</div>'
    '<div class="bg-section-note">'
    'Rule-based prototype guidance associated with the score category.'
    '</div>',
    unsafe_allow_html=True,
)

if status == "REFERENCE-LIKE":
    recommendation = (
        "The prototype score falls in the reference-like range. "
        "Continue the established inspection schedule and compare future "
        "measurements against approved equipment limits."
    )
elif status == "INTERMEDIATE":
    recommendation = (
        "Review resistance, voltage drop, temperature rise and acoustic "
        "features against historical readings and approved limits. "
        "Arrange qualified inspection if readings are abnormal or worsening."
    )
else:
    recommendation = (
        "Prioritize review by qualified electrical maintenance personnel. "
        "Verify the recorded readings against approved limits and follow "
        "the site's electrical safety and isolation procedures before any "
        "physical inspection."
    )

st.markdown(
    f"""
    <div style="background:#FFFFFF;border:1px solid #E1E8EF;
                border-radius:10px;padding:1.1rem 1.2rem;">
        <div style="font-size:0.72rem;color:{TEAL};font-weight:750;
                    text-transform:uppercase;letter-spacing:0.1em;">
            Recommended next step
        </div>
        <div style="font-size:0.96rem;color:{NAVY};line-height:1.7;
                    margin-top:0.5rem;">
            {recommendation}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Prototype guidance only. The score boundaries are illustrative, not "
    "manufacturer limits, certified safety thresholds, or proof of failure."
)


# =========================================================
# TECHNICAL NOTES
# =========================================================

with st.expander("Model and prototype limitations"):
    st.markdown(
        """
        **Model**
        - Random Forest classifier using electrical, thermal and acoustic features.
        - The model is fitted on all 74 observations for this overview.
        - Therefore, the selected-record prediction and displayed trend are
          in-sample estimates, not held-out validation results.

        **Prototype Joint Health Index**
        - JHI = 100 × P(tight) + 50 × P(mid) + 0 × P(loose).
        - Score categories are prototype rules and are not an industry standard.

        **Limitations**
        - The dataset is small and may contain correlated observations.
        - Model probabilities are not guaranteed to be calibrated.
        - The dashboard does not establish industrial field accuracy, remaining
          useful life, failure prediction, or certified safety.
        - The page displays recorded dataset values; no live sensor connection
          is active.
        """
    )

st.divider()

st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;gap:1rem;
                flex-wrap:wrap;color:{MUTED};font-size:0.75rem;">
        <span>BUSGUARD / CONDITION INTELLIGENCE</span>
        <span>Experimental prototype · Offline dataset</span>
    </div>
    """,
    unsafe_allow_html=True,
)
