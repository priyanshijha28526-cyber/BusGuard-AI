
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Assessment | BusGuard",
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

FEATURE_SETS = {
    "Electrical": ELECTRICAL,
    "Thermal": THERMAL,
    "Acoustic": ACOUSTIC,
    "Electrical + Thermal": ELECTRICAL + THERMAL,
    "Full Sensor Fusion": ELECTRICAL + THERMAL + ACOUSTIC,
}


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
            padding: 18px 20px;
            min-height: 112px;
        }}

        [data-testid="stMetricLabel"] {{
            color: {MUTED};
            font-size: 0.84rem;
        }}

        [data-testid="stMetricValue"] {{
            color: {NAVY};
            font-weight: 700;
        }}

        [data-testid="stMetricDelta"] {{
            font-size: 0.8rem;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {GRID};
            border-radius: 10px;
            overflow: hidden;
        }}

        [data-testid="stExpander"] {{
            background: {WHITE};
            border: 1px solid {GRID};
            border-radius: 10px;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 10px;
        }}

        div[data-testid="stVerticalBlock"] > div:has(> .section-marker) {{
            margin-top: 0.4rem;
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
            font-size: 0.96rem;
            line-height: 1.6;
            max-width: 850px;
        }}

        .section-note {{
            color: {MUTED};
            font-size: 0.84rem;
            line-height: 1.55;
            margin-top: -0.4rem;
            margin-bottom: 1rem;
        }}

        .status-chip {{
            display: inline-block;
            padding: 5px 10px;
            border: 1px solid #CDE5E1;
            border-radius: 5px;
            background: #EDF7F5;
            color: {TEAL};
            font-size: 0.72rem;
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
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .technical-card-value {{
            color: {NAVY};
            font-size: 1.18rem;
            font-weight: 700;
        }}

        .technical-card-copy {{
            color: {MUTED};
            font-size: 0.82rem;
            line-height: 1.5;
            margin-top: 6px;
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

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[required].isnull().any().any():
        raise ValueError("Missing values found in required columns.")

    if set(data["label"].unique()) != set(CLASS_ORDER):
        raise ValueError("Expected all three labels: loose, mid and tight.")

    data = data.sort_values("t").reset_index(drop=True)
    data["delta_T"] = data["t_max"] - data["t_amb"]

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
        "delta_T",
    ]

    if not data[numeric_columns].apply(
        lambda column: pd.api.types.is_numeric_dtype(column)
    ).all():
        raise ValueError("One or more sensor columns are not numeric.")

    if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric values found in the dataset.")

    return data


# =========================================================
# GROUPED VALIDATION
# =========================================================

@st.cache_data
def evaluate_models(data):
    data = data.reset_index(drop=True)
    groups = np.arange(len(data)) // 5

    if len(np.unique(groups)) < 5:
        raise ValueError("At least five groups are required.")

    cv = GroupKFold(n_splits=5)

    comparison_rows = []
    fold_details = []

    fusion_actual = np.empty(len(data), dtype=object)
    fusion_predicted = np.empty(len(data), dtype=object)

    for model_name, features in FEATURE_SETS.items():
        fold_accuracies = []
        fold_f1_scores = []

        oof_predictions = np.empty(len(data), dtype=object)

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
                train_data[features],
                train_data["label"],
            )

            predictions = model.predict(test_data[features])
            oof_predictions[test_idx] = predictions

            fold_accuracy = accuracy_score(
                test_data["label"],
                predictions,
            )

            fold_f1 = f1_score(
                test_data["label"],
                predictions,
                labels=CLASS_ORDER,
                average="weighted",
                zero_division=0,
            )

            fold_accuracies.append(fold_accuracy)
            fold_f1_scores.append(fold_f1)

            fold_details.append(
                {
                    "Configuration": model_name,
                    "Fold": fold_number,
                    "Accuracy": fold_accuracy,
                    "Weighted F1": fold_f1,
                    "Test Samples": len(test_idx),
                }
            )

        comparison_rows.append(
            {
                "Configuration": model_name,
                "Mean Accuracy": np.mean(fold_accuracies),
                "Accuracy Std": np.std(fold_accuracies),
                "Mean Weighted F1": np.mean(fold_f1_scores),
                "F1 Std": np.std(fold_f1_scores),
            }
        )

        if model_name == "Full Sensor Fusion":
            fusion_actual[:] = data["label"].to_numpy()
            fusion_predicted[:] = oof_predictions

    comparison = pd.DataFrame(comparison_rows)

    confusion = confusion_matrix(
        fusion_actual,
        fusion_predicted,
        labels=CLASS_ORDER,
    )

    # Descriptive importance from a final model fitted to all observations.
    fusion_features = FEATURE_SETS["Full Sensor Fusion"]

    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    )

    final_model.fit(
        data[fusion_features],
        data["label"],
    )

    importance = pd.DataFrame(
        {
            "Feature": fusion_features,
            "Importance": final_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return (
        comparison,
        confusion,
        importance,
        pd.DataFrame(fold_details),
    )


# =========================================================
# CHART STYLING
# =========================================================

def style_axis(ax, grid_axis="y"):
    ax.set_facecolor(WHITE)

    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(NAVY)

    ax.grid(
        axis=grid_axis,
        color=GRID,
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_axisbelow(True)


# =========================================================
# PAGE HEADER
# =========================================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        '<div class="eyebrow">Model intelligence / 03</div>',
        unsafe_allow_html=True,
    )
    st.title("AI Assessment")
    st.markdown(
        """
        <div class="page-subtitle">
            Evaluate condition-classification performance, compare sensor
            configurations, and inspect the features used by the model.
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        "<div style='text-align:right; padding-top:12px;'>"
        "<span class='status-chip'>OFFLINE EVALUATION</span>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()


# =========================================================
# LOAD AND RUN
# =========================================================

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
    comparison, confusion, importance, fold_details = evaluate_models(df)
except Exception as exc:
    st.error(f"Unable to evaluate models: {exc}")
    st.stop()


# =========================================================
# MODEL OVERVIEW
# =========================================================

st.subheader("Model overview")
st.markdown(
    '<div class="section-note">'
    "Evaluation scope and dataset composition."
    "</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

c1.metric("Classifier", "Random Forest")
c2.metric("Recorded observations", f"{len(df):,}")
c3.metric("Condition classes", str(df["label"].nunique()))

st.markdown(
    """
    <div class="technical-card">
        <div class="technical-card-title">Input architecture</div>
        <div class="technical-card-value">
            Electrical + Thermal + Acoustic
        </div>
        <div class="technical-card-copy">
            Target classes: loose, mid and tight. The model uses eight
            sensor-derived features for the full sensor-fusion configuration.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.divider()


# =========================================================
# VALIDATION METRICS
# =========================================================

st.subheader("Grouped cross-validation")
st.markdown(
    '<div class="section-note">'
    "Mean performance across five grouped folds. "
    "These values summarize this dataset and validation setup."
    "</div>",
    unsafe_allow_html=True,
)

fusion_result = comparison[
    comparison["Configuration"] == "Full Sensor Fusion"
].iloc[0]

m1, m2, m3 = st.columns(3)

m1.metric(
    "Mean fold accuracy",
    f"{fusion_result['Mean Accuracy']:.1%}",
    help="Average accuracy across the five validation folds.",
)

m2.metric(
    "Mean weighted F1",
    f"{fusion_result['Mean Weighted F1']:.3f}",
    help="Average weighted F1 across the five validation folds.",
)

m3.metric(
    "Validation folds",
    "5",
    help="GroupKFold with groups of five consecutive observations.",
)

with st.expander("View all sensor-configuration metrics", expanded=True):
    comparison_display = comparison.rename(
        columns={
            "Mean Accuracy": "Mean accuracy",
            "Accuracy Std": "Accuracy SD",
            "Mean Weighted F1": "Mean weighted F1",
            "F1 Std": "F1 SD",
        }
    )

    st.dataframe(
        comparison_display.round(3),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Standard deviation (SD) describes variation between fold scores. "
    "GroupKFold prevents observations within the same group from appearing "
    "in both training and testing, but groups are not held out in chronological "
    "order. This is not prospective temporal validation."
)

st.divider()


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("Full sensor-fusion confusion matrix")
st.markdown(
    '<div class="section-note">'
    "Out-of-fold predictions: each observation is predicted by a model "
    "that did not train on its group."
    "</div>",
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(figsize=(7, 4.8))
fig.patch.set_facecolor(WHITE)

image = ax.imshow(
    confusion,
    interpolation="nearest",
    cmap="Blues",
)

ax.set(
    xticks=np.arange(len(CLASS_ORDER)),
    yticks=np.arange(len(CLASS_ORDER)),
    xticklabels=[CLASS_DISPLAY[label] for label in CLASS_ORDER],
    yticklabels=[CLASS_DISPLAY[label] for label in CLASS_ORDER],
    xlabel="Predicted condition",
    ylabel="Recorded condition",
)

threshold = confusion.max() / 2 if confusion.size else 0

for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(
            j,
            i,
            str(confusion[i, j]),
            ha="center",
            va="center",
            color=WHITE if confusion[i, j] > threshold else NAVY,
            fontsize=12,
            fontweight="bold",
        )

ax.tick_params(colors=MUTED, labelsize=9)

for spine in ax.spines.values():
    spine.set_color(GRID)

ax.xaxis.label.set_color(MUTED)
ax.yaxis.label.set_color(MUTED)
ax.set_title(
    "Recorded labels vs model predictions",
    loc="left",
    color=NAVY,
    fontsize=12,
    fontweight="bold",
    pad=14,
)

fig.tight_layout()
st.pyplot(fig, use_container_width=False)
plt.close(fig)

st.caption(
    "Rows represent recorded labels; columns represent predictions. "
    "The matrix combines held-out predictions from the five grouped folds."
)

st.divider()


# =========================================================
# SENSOR CONFIGURATION COMPARISON
# =========================================================

st.subheader("Sensor configuration comparison")
st.markdown(
    '<div class="section-note">'
    "Compare classification accuracy when the model receives different "
    "subsets of sensor features."
    "</div>",
    unsafe_allow_html=True,
)

plot_comparison = comparison.sort_values("Mean Accuracy")

fig, ax = plt.subplots(figsize=(10, 4.8))
fig.patch.set_facecolor(WHITE)

bar_colors = [
    TEAL if name == "Full Sensor Fusion" else NAVY
    for name in plot_comparison["Configuration"]
]

bars = ax.barh(
    plot_comparison["Configuration"],
    plot_comparison["Mean Accuracy"],
    xerr=plot_comparison["Accuracy Std"],
    color=bar_colors,
    height=0.56,
    capsize=4,
    error_kw={"ecolor": MUTED, "elinewidth": 1.2},
)

ax.set_xlabel("Mean fold accuracy")
ax.set_xlim(0, 1.12)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda value, _: f"{value:.0%}")
)

style_axis(ax, grid_axis="x")

for bar, value in zip(bars, plot_comparison["Mean Accuracy"]):
    ax.text(
        min(value + 0.015, 1.035),
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1%}",
        va="center",
        ha="left",
        fontsize=9,
        color=NAVY,
        fontweight="bold",
    )

ax.set_title(
    "Accuracy by feature configuration",
    loc="left",
    fontsize=12,
    fontweight="bold",
    pad=14,
)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.caption(
    "Error bars show one standard deviation across fold accuracies. "
    "If the electrical-only model matches full sensor fusion, these results "
    "do not demonstrate an accuracy improvement from adding thermal and "
    "acoustic features under this validation protocol."
)

st.divider()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("Feature importance")
st.markdown(
    '<div class="section-note">'
    "Relative impurity-based importance from a Random Forest fitted "
    "to the complete dataset."
    "</div>",
    unsafe_allow_html=True,
)

importance_plot = importance.sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5.2))
fig.patch.set_facecolor(WHITE)

bar_colors = [
    TEAL if feature in ["R_mohm", "vdrop_mV"] else NAVY
    for feature in importance_plot["Feature"]
]

bars = ax.barh(
    importance_plot["Feature"],
    importance_plot["Importance"],
    color=bar_colors,
    height=0.58,
)

ax.set_xlabel("Relative importance")
style_axis(ax, grid_axis="x")

for bar, value in zip(bars, importance_plot["Importance"]):
    ax.text(
        value + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        ha="left",
        fontsize=9,
        color=NAVY,
    )

ax.set_xlim(
    0,
    max(importance_plot["Importance"].max() * 1.18, 0.1),
)

ax.set_title(
    "Contribution to model splits",
    loc="left",
    fontsize=12,
    fontweight="bold",
    pad=14,
)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

with st.expander("View feature importance values"):
    st.dataframe(
        importance.round(4),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Importance values are descriptive, not causal. Because this model is "
    "fitted on all observations, these values are not an out-of-sample "
    "estimate of each feature's contribution to generalization."
)

st.divider()


# =========================================================
# INTERPRETATION
# =========================================================

st.subheader("Engineering interpretation")

left, right = st.columns(2)

with left:
    st.markdown(
        f"""
        <div class="technical-card">
            <div class="technical-card-title">Electrical indicators</div>
            <div class="technical-card-value">Resistance & voltage drop</div>
            <div class="technical-card-copy">
                These features directly characterize the electrical
                response recorded in the prototype dataset. Their importance
                should be interpreted alongside the dataset's class structure.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="technical-card">
            <div class="technical-card-title">Sensor fusion assessment</div>
            <div class="technical-card-value">Compare, don't assume</div>
            <div class="technical-card-copy">
                Thermal and acoustic signals may provide complementary
                information, but their added value must be demonstrated
                using validation on independent experiments.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

with st.expander("Validation protocol and limitations"):
    st.markdown(
        """
        - **Dataset:** 74 recorded observations.
        - **Target classes:** loose, mid and tight.
        - **Classifier:** Random Forest.
        - **Feature configurations:** electrical, thermal, acoustic,
          electrical plus thermal, and full sensor fusion.
        - **Validation:** five-fold GroupKFold using groups of five
          consecutive observations.
        - **Grouping limitation:** samples from the same group are not
          split across training and testing. However, later groups can
          appear in training while earlier groups are tested.
        - **Dataset limitation:** the dataset is small, and adjacent
          measurements may be correlated.
        - **Feature-importance limitation:** importance does not establish
          causality or prove independent improvement in generalization.
        - **Deployment limitation:** these results do not establish
          industrial field accuracy, failure prediction, remaining useful
          life, or certified safety.
        """
    )

with st.expander("Verify fold-level calculations"):
    st.write("Mean and standard deviation are calculated from the five fold-level scores.")

    st.dataframe(
        comparison[
            [
                "Configuration",
                "Mean Accuracy",
                "Accuracy Std",
                "Mean Weighted F1",
                "F1 Std",
            ]
        ].round(3),
        hide_index=True,
        use_container_width=True,
    )

    st.write("Individual fold results")

    st.dataframe(
        fold_details.round(3),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Weighted F1 is calculated separately for each held-out fold, "
        "then summarized across folds. These values should not be treated "
        "as a guarantee of performance on unseen industrial installations."
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
        <span>BUSGUARD AI · MODEL ASSESSMENT</span>
        <span>Prototype evaluation · Not a certified safety system</span>
    </div>
    """,
    unsafe_allow_html=True,
)

