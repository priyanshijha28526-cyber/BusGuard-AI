
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sensor Analysis | BusGuard",
    page_icon="B",
    layout="wide",
)

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "full_run.csv"
)

CONDITIONS = ["loose", "mid", "tight"]

# Consistent with the Overview page's industrial visual style.
NAVY = "#17324D"
TEAL = "#16877B"
MUTED = "#63758A"
GRID = "#E3E9EF"
BACKGROUND = "#F4F7FA"

CONDITION_COLORS = {
    "loose": "#B65D5D",
    "mid": "#B18A43",
    "tight": "#16877B",
}


# =========================================================
# VISUAL STYLING
# =========================================================

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
        border-right: 1px solid {GRID};
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
        font-size: 2.2rem;
        font-weight: 750;
        letter-spacing: -0.045em;
        line-height: 1.15;
        margin: 0;
    }}

    .bg-subtitle {{
        color: {MUTED};
        font-size: 0.94rem;
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
        letter-spacing: 0.08em;
    }}

    .bg-section {{
        color: {NAVY};
        font-size: 1.14rem;
        font-weight: 750;
        letter-spacing: -0.015em;
        margin-bottom: 0.15rem;
    }}

    .bg-section-note {{
        color: {MUTED};
        font-size: 0.83rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }}

    [data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #E1E8EF;
        border-radius: 10px;
        padding: 1rem;
        min-height: 108px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED};
        font-size: 0.82rem;
    }}

    [data-testid="stMetricValue"] {{
        color: {NAVY};
        font-weight: 750;
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

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

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

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}"
        )

    if df[required].isnull().any().any():
        raise ValueError(
            "Missing values found in required columns. "
            "Check the source CSV before interpreting results."
        )

    if not set(df["label"].unique()).issubset(set(CONDITIONS)):
        raise ValueError("Unexpected condition labels in the dataset.")

    df = df.sort_values("t").reset_index(drop=True)
    df["delta_T"] = df["t_max"] - df["t_amb"]

    return df


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# =========================================================
# HEADER
# =========================================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        """
        <div class="bg-eyebrow">Signal intelligence / 02</div>
        <div class="bg-title">Sensor analysis</div>
        <div class="bg-subtitle">
            Explore recorded electrical, thermal and acoustic signatures
            across the experimental run.
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        f"""
        <div style="text-align:right;padding-top:0.35rem;">
            <span class="bg-tag">RECORDED DATA</span>
            <div style="font-size:0.78rem;color:{MUTED};
                        margin-top:0.7rem;">
                {len(df)} observations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# FILTERS
# =========================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="font-size:1.2rem;font-weight:800;
                    letter-spacing:0.04em;color:{NAVY};">
            BUSGUARD
        </div>
        <div style="font-size:0.72rem;color:{MUTED};
                    letter-spacing:0.08em;margin-top:0.2rem;">
            SIGNAL EXPLORER
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### Analysis filters")

    selected_conditions = st.multiselect(
        "Joint condition",
        options=CONDITIONS,
        default=CONDITIONS,
        format_func=str.capitalize,
    )

    st.caption(
        "The selected conditions apply to all charts, summaries "
        "and the downloadable table."
    )

filtered = df[df["label"].isin(selected_conditions)].copy()

if filtered.empty:
    st.warning("Select at least one joint condition to view the data.")
    st.stop()


# =========================================================
# DATASET SUMMARY
# =========================================================

st.markdown(
    """
    <div class="bg-section">Dataset snapshot</div>
    <div class="bg-section-note">
        Summary statistics for the currently selected conditions.
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Observations", f"{len(filtered)}")
c2.metric("Mean current", f"{filtered['I'].mean():.2f} A")
c3.metric("Mean resistance", f"{filtered['R_mohm'].mean():.3f} mΩ")
c4.metric("Mean voltage drop", f"{filtered['vdrop_mV'].mean():.2f} mV")

with st.expander("Condition distribution"):
    distribution = (
        filtered["label"]
        .value_counts()
        .reindex(CONDITIONS, fill_value=0)
        .rename_axis("Recorded condition")
        .reset_index(name="Observations")
    )

    st.dataframe(
        distribution,
        hide_index=True,
        use_container_width=True,
    )

st.divider()


# =========================================================
# CHART STYLING HELPER
# =========================================================

def style_axis(ax, title, y_label):
    ax.set_title(
        title,
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=NAVY,
        pad=12,
    )

    ax.set_xlabel("Recorded time (s)", color=MUTED, labelpad=8)
    ax.set_ylabel(y_label, color=MUTED, labelpad=8)

    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)


def plot_signal(data, column, y_label, title):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # A neutral line shows the recorded chronological sequence.
    ax.plot(
        data["t"],
        data[column],
        color="#AAB7C4",
        linewidth=1.3,
        alpha=0.9,
        zorder=1,
    )

    # Points use consistent colours for the recorded labels.
    for condition in CONDITIONS:
        group = data[data["label"] == condition]

        if not group.empty:
            ax.scatter(
                group["t"],
                group[column],
                s=29,
                color=CONDITION_COLORS[condition],
                edgecolors="#FFFFFF",
                linewidths=0.55,
                label=condition.capitalize(),
                zorder=3,
            )

    style_axis(ax, title, y_label)

    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=8,
        loc="best",
        labelcolor=MUTED,
    )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =========================================================
# ELECTRICAL ANALYSIS
# =========================================================

st.markdown(
    """
    <div class="bg-section">01 / Electrical measurements</div>
    <div class="bg-section-note">
        Resistance and voltage drop across the recorded joint conditions.
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    plot_signal(
        filtered,
        "R_mohm",
        "Resistance (mΩ)",
        "Joint resistance",
    )

with right:
    plot_signal(
        filtered,
        "vdrop_mV",
        "Voltage drop (mV)",
        "Joint voltage drop",
    )

st.caption(
    "These are recorded values from the supplied dataset. "
    "Their relationship depends on the experimental setup and current."
)

st.divider()


# =========================================================
# THERMAL ANALYSIS
# =========================================================

st.markdown(
    """
    <div class="bg-section">02 / Thermal measurements</div>
    <div class="bg-section-note">
        Recorded temperature characteristics and rise above ambient.
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    plot_signal(
        filtered,
        "t_max",
        "Temperature (°C)",
        "Maximum recorded temperature",
    )

with right:
    plot_signal(
        filtered,
        "delta_T",
        "Temperature rise (°C)",
        "Temperature rise above ambient",
    )

st.caption(
    "ΔT is calculated as maximum recorded temperature minus ambient "
    "temperature. These measurements do not predict future overheating."
)

st.divider()


# =========================================================
# ACOUSTIC ANALYSIS
# =========================================================

st.markdown(
    """
    <div class="bg-section">03 / Acoustic features</div>
    <div class="bg-section-note">
        Explore the two acoustic features supplied in the dataset.
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    plot_signal(
        filtered,
        "ac_100",
        "Recorded feature value",
        "AC 100 feature",
    )

with right:
    plot_signal(
        filtered,
        "ac_band",
        "Recorded feature value",
        "Acoustic band feature",
    )

st.caption(
    "The acoustic feature units are not specified in the CSV, "
    "so no physical unit is assigned."
)

st.divider()


# =========================================================
# MEASUREMENT EXPLORER
# =========================================================

st.markdown(
    """
    <div class="bg-section">04 / Measurement explorer</div>
    <div class="bg-section-note">
        Focus on an individual signal across the selected observations.
    </div>
    """,
    unsafe_allow_html=True,
)

available_signals = {
    "Current": ("I", "A"),
    "Joint resistance": ("R_mohm", "mΩ"),
    "Voltage drop": ("vdrop_mV", "mV"),
    "Maximum temperature": ("t_max", "°C"),
    "Mean temperature": ("t_mean", "°C"),
    "Temperature rise": ("delta_T", "°C"),
    "AC 100 feature": ("ac_100", "dataset units"),
    "Acoustic band feature": ("ac_band", "dataset units"),
}

choice = st.selectbox(
    "Choose a measurement",
    list(available_signals.keys()),
)

column, unit = available_signals[choice]

fig, ax = plt.subplots(figsize=(11, 3.8))
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

ax.plot(
    filtered["t"],
    filtered[column],
    color="#AAB7C4",
    linewidth=1.3,
    zorder=1,
)

for condition in CONDITIONS:
    group = filtered[filtered["label"] == condition]

    if not group.empty:
        ax.scatter(
            group["t"],
            group[column],
            s=35,
            color=CONDITION_COLORS[condition],
            edgecolors="#FFFFFF",
            linewidths=0.6,
            label=condition.capitalize(),
            zorder=3,
        )

style_axis(ax, choice, f"{choice} ({unit})")

ax.legend(
    frameon=False,
    ncol=3,
    fontsize=8,
    labelcolor=MUTED,
)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.divider()


# =========================================================
# RECORDED DATA TABLE AND DOWNLOAD
# =========================================================

st.markdown(
    """
    <div class="bg-section">05 / Recorded measurements</div>
    <div class="bg-section-note">
        Inspect the filtered records or export them for further analysis.
    </div>
    """,
    unsafe_allow_html=True,
)

table_columns = [
    "t",
    "label",
    "I",
    "vdrop_mV",
    "R_mohm",
    "t_max",
    "t_mean",
    "t_amb",
    "delta_T",
    "ac_100",
    "ac_band",
]

with st.expander("Open recorded data table"):
    st.dataframe(
        filtered[table_columns].round(4),
        use_container_width=True,
        hide_index=True,
    )

download_left, download_right = st.columns([1, 2])

with download_left:
    st.download_button(
        label="Download filtered CSV",
        data=filtered[table_columns].to_csv(index=False).encode("utf-8"),
        file_name="busguard_filtered_measurements.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_right:
    st.caption(
        f"Export includes {len(filtered)} observations matching "
        "the current filter."
    )


# =========================================================
# DATA DEFINITIONS AND LIMITATIONS
# =========================================================

with st.expander("Data definitions and limitations"):
    st.markdown(
        """
        - **t:** recorded time in seconds.
        - **I:** current in amperes.
        - **vdrop_mV:** recorded voltage drop in millivolts.
        - **R_mohm:** recorded joint resistance in milliohms.
        - **t_max / t_mean:** recorded maximum and mean temperatures.
        - **t_amb:** ambient temperature.
        - **ΔT:** calculated as `t_max - t_amb`.
        - **ac_100 / ac_band:** acoustic features supplied in the dataset.
        - **label:** recorded experimental condition (`loose`, `mid`, `tight`).

        The dataset contains 74 observations. Results describe this dataset
        and do not establish industrial field performance or universal
        condition thresholds. The page displays recorded observations,
        not live sensor readings.
        """
    )

st.divider()

st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;gap:1rem;
                flex-wrap:wrap;color:{MUTED};font-size:0.75rem;">
        <span>BUSGUARD / SIGNAL INTELLIGENCE</span>
        <span>Experimental dataset · Offline analysis</span>
    </div>
    """,
    unsafe_allow_html=True,
)
