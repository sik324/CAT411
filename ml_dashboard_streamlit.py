from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from streamlit_plotly_events import plotly_events

st.set_page_config(
    page_title="Bridge Damage Dashboard",
    page_icon="🌉",
    layout="wide",
)

# ============================================================
# PATHS
# ============================================================
DATA_DIR = Path("data")
BRIDGES_CSV = DATA_DIR / "ml_bridges.csv"
FEATURE_CSV = DATA_DIR / "ml_feature_importance.csv"
FRAGILITY_CSV = DATA_DIR / "hazus_bridge_fragility_params.csv"

# ============================================================
# CONSTANTS
# ============================================================
DS_ORDER = ["none", "slight", "moderate", "extensive", "complete"]
DS_INDEX = {ds: i for i, ds in enumerate(DS_ORDER)}

COLOR_MAP = {
    "none": "#4CAF50",
    "slight": "#2196F3",
    "moderate": "#FF9800",
    "extensive": "#F44336",
    "complete": "#7B0000",
}

DAMAGE_RATIO = {
    "none": 0.00,
    "slight": 0.03,
    "moderate": 0.08,
    "extensive": 0.25,
    "complete": 1.00,
}

REPAIR_CATEGORY = {
    "none": "Safe",
    "slight": "Needs Repair",
    "moderate": "Needs Repair",
    "extensive": "Needs Closure",
    "complete": "Needs Replacement",
}

REPAIR_ACTION = {
    "none": "No action needed",
    "slight": "Schedule inspection",
    "moderate": "Plan repair",
    "extensive": "Close bridge and repair",
    "complete": "Close bridge and replace",
}

REPAIR_PRIORITY = {
    "none": 0,
    "slight": 1,
    "moderate": 2,
    "extensive": 3,
    "complete": 4,
}

DEFAULT_BETA = 0.6
DEFAULT_K = 1.0


# ============================================================
# HELPERS
# ============================================================
def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True):
    existing = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in existing:
            return existing[cand.lower()]
    if required:
        raise KeyError(f"Missing column. Tried: {candidates}")
    return None


@st.cache_data
def load_all():
    bridges = pd.read_csv(BRIDGES_CSV)

    feat = pd.DataFrame()
    if FEATURE_CSV.exists():
        feat = pd.read_csv(FEATURE_CSV)

    frag = pd.DataFrame()
    if FRAGILITY_CSV.exists():
        frag = pd.read_csv(FRAGILITY_CSV)

    return bridges, feat, frag


def normalize_bridge_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    colmap = {
        "structure_id": find_col(out, ["Structure_ID", "structure_id", "Structure_Number", "structure_number"]),
        "lat": find_col(out, ["Latitude", "latitude", "lat"]),
        "lon": find_col(out, ["Longitude", "longitude", "lon"]),
        "hwb": find_col(out, ["HWB_Class", "hwb_class"]),
        "year_built": find_col(out, ["Year_Built", "year_built"], required=False),
        "design_era": find_col(out, ["Design_Era", "design_era"], required=False),
        "num_spans": find_col(out, ["Num_Spans", "num_spans"], required=False),
        "sa": find_col(out, ["Sa_1s_g", "sa1s_shakemap", "sa_1s_g"]),
        "pga": find_col(out, ["PGA_g", "pga_shakemap", "pga_g"], required=False),
        "obs": find_col(out, ["Observed_DS", "obs", "Observed_damage"]),
        "rf": find_col(out, ["RF_DS", "pred_rf", "rf_ds"], required=False),
        "gmm_map": find_col(out, ["GMM_MAP_DS", "pred_gmm_map", "gmm_map", "pred_gmm"], required=False),
        "gmm_cost": find_col(out, ["GMM_COST_DS", "pred_gmm_cost", "gmm_cost"], required=False),
        # Optional probability columns
        "rf_p_none": find_col(out, ["RF_P_None", "rf_p_none"], required=False),
        "rf_p_slight": find_col(out, ["RF_P_Slight", "rf_p_slight"], required=False),
        "rf_p_moderate": find_col(out, ["RF_P_Moderate", "rf_p_moderate"], required=False),
        "rf_p_extensive": find_col(out, ["RF_P_Extensive", "rf_p_extensive"], required=False),
        "rf_p_complete": find_col(out, ["RF_P_Complete", "rf_p_complete"], required=False),
        "gmm_p_none": find_col(out, ["GMM_P_None", "gmm_p_none"], required=False),
        "gmm_p_slight": find_col(out, ["GMM_P_Slight", "gmm_p_slight"], required=False),
        "gmm_p_moderate": find_col(out, ["GMM_P_Moderate", "gmm_p_moderate"], required=False),
        "gmm_p_extensive": find_col(out, ["GMM_P_Extensive", "gmm_p_extensive"], required=False),
        "gmm_p_complete": find_col(out, ["GMM_P_Complete", "gmm_p_complete"], required=False),
    }

    rename_dict = {
        colmap["structure_id"]: "Structure_ID",
        colmap["lat"]: "Latitude",
        colmap["lon"]: "Longitude",
        colmap["hwb"]: "HWB_Class",
        colmap["sa"]: "Sa_1s_g",
        colmap["obs"]: "Observed_DS",
    }

    optional_map = {
        "year_built": "Year_Built",
        "design_era": "Design_Era",
        "num_spans": "Num_Spans",
        "pga": "PGA_g",
        "rf": "RF_DS",
        "gmm_map": "GMM_MAP_DS",
        "gmm_cost": "GMM_COST_DS",
        "rf_p_none": "RF_P_None",
        "rf_p_slight": "RF_P_Slight",
        "rf_p_moderate": "RF_P_Moderate",
        "rf_p_extensive": "RF_P_Extensive",
        "rf_p_complete": "RF_P_Complete",
        "gmm_p_none": "GMM_P_None",
        "gmm_p_slight": "GMM_P_Slight",
        "gmm_p_moderate": "GMM_P_Moderate",
        "gmm_p_extensive": "GMM_P_Extensive",
        "gmm_p_complete": "GMM_P_Complete",
    }

    for key, new_name in optional_map.items():
        if colmap[key]:
            rename_dict[colmap[key]] = new_name

    out = out.rename(columns=rename_dict)

    # normalize strings
    for c in ["Observed_DS", "RF_DS", "GMM_MAP_DS", "GMM_COST_DS", "Design_Era", "HWB_Class"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.lower()

    # repair / priority fields
    out["Obs_DR"] = out["Observed_DS"].map(DAMAGE_RATIO)
    out["Obs_Repair_Category"] = out["Observed_DS"].map(REPAIR_CATEGORY)
    out["Obs_Repair_Action"] = out["Observed_DS"].map(REPAIR_ACTION)
    out["Obs_Priority"] = out["Observed_DS"].map(REPAIR_PRIORITY)

    if "RF_DS" in out.columns:
        out["RF_DR"] = out["RF_DS"].map(DAMAGE_RATIO)
        out["RF_Repair_Category"] = out["RF_DS"].map(REPAIR_CATEGORY)
        out["RF_Repair_Action"] = out["RF_DS"].map(REPAIR_ACTION)
        out["RF_Priority"] = out["RF_DS"].map(REPAIR_PRIORITY)

    if "GMM_MAP_DS" in out.columns:
        out["GMM_MAP_DR"] = out["GMM_MAP_DS"].map(DAMAGE_RATIO)
        out["GMM_MAP_Repair_Category"] = out["GMM_MAP_DS"].map(REPAIR_CATEGORY)
        out["GMM_MAP_Repair_Action"] = out["GMM_MAP_DS"].map(REPAIR_ACTION)
        out["GMM_MAP_Priority"] = out["GMM_MAP_DS"].map(REPAIR_PRIORITY)

    if "GMM_COST_DS" in out.columns:
        out["GMM_COST_DR"] = out["GMM_COST_DS"].map(DAMAGE_RATIO)
        out["GMM_COST_Repair_Category"] = out["GMM_COST_DS"].map(REPAIR_CATEGORY)
        out["GMM_COST_Repair_Action"] = out["GMM_COST_DS"].map(REPAIR_ACTION)
        out["GMM_COST_Priority"] = out["GMM_COST_DS"].map(REPAIR_PRIORITY)

    return out


def normalize_fragility_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    colmap = {
        "hwb": find_col(out, ["HWB_Class", "hwb_class"]),
        "slight": find_col(out, ["slight_median"]),
        "moderate": find_col(out, ["moderate_median"]),
        "extensive": find_col(out, ["extensive_median"]),
        "complete": find_col(out, ["complete_median"]),
    }
    out = out.rename(columns={
        colmap["hwb"]: "HWB_Class",
        colmap["slight"]: "slight_median",
        colmap["moderate"]: "moderate_median",
        colmap["extensive"]: "extensive_median",
        colmap["complete"]: "complete_median",
    })
    out["HWB_Class"] = out["HWB_Class"].astype(str).str.strip().str.lower()
    return out


def severity(series: pd.Series) -> pd.Series:
    return series.map(DS_INDEX).fillna(-1)


def metrics_from_prediction(df: pd.DataFrame, pred_col: str) -> dict:
    obs = df["Observed_DS"]
    pred = df[pred_col]
    overall = (obs == pred).mean() * 100
    damaged_mask = obs != "none"
    damaged_acc = (obs[damaged_mask] == pred[damaged_mask]).mean() * 100 if damaged_mask.any() else np.nan
    missed = ((obs != "none") & (pred == "none")).sum()
    false_alarms = ((obs == "none") & (pred != "none")).sum()
    return {
        "overall_accuracy": overall,
        "damaged_accuracy": damaged_acc,
        "missed_damaged": int(missed),
        "false_alarms": int(false_alarms),
    }


def damage_count_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = [ds.capitalize() for ds in DS_ORDER]

    fig.add_bar(
        name="Observed",
        x=x,
        y=[(df["Observed_DS"] == ds).sum() for ds in DS_ORDER],
        marker_color="#4CAF50",
    )

    if "RF_DS" in df.columns:
        fig.add_bar(
            name="RF",
            x=x,
            y=[(df["RF_DS"] == ds).sum() for ds in DS_ORDER],
            marker_color="#2196F3",
        )

    if "GMM_MAP_DS" in df.columns:
        fig.add_bar(
            name="GMM MAP",
            x=x,
            y=[(df["GMM_MAP_DS"] == ds).sum() for ds in DS_ORDER],
            marker_color="#FF9800",
        )

    fig.update_layout(
        barmode="group",
        title="Damage State Counts",
        xaxis_title="Damage State",
        yaxis_title="Number of Bridges",
        height=420,
    )
    return fig


def repair_count_chart(df: pd.DataFrame) -> go.Figure:
    cats = ["Safe", "Needs Repair", "Needs Closure", "Needs Replacement"]
    fig = go.Figure()

    fig.add_bar(
        name="Observed",
        x=cats,
        y=[(df["Obs_Repair_Category"].astype(str).str.lower() == c.lower()).sum() for c in cats],
        marker_color="#4CAF50",
    )

    if "RF_Repair_Category" in df.columns:
        fig.add_bar(
            name="RF",
            x=cats,
            y=[(df["RF_Repair_Category"].astype(str).str.lower() == c.lower()).sum() for c in cats],
            marker_color="#2196F3",
        )

    if "GMM_MAP_Repair_Category" in df.columns:
        fig.add_bar(
            name="GMM MAP",
            x=cats,
            y=[(df["GMM_MAP_Repair_Category"].astype(str).str.lower() == c.lower()).sum() for c in cats],
            marker_color="#FF9800",
        )

    fig.update_layout(
        barmode="group",
        title="Repair Category Counts",
        xaxis_title="Repair Category",
        yaxis_title="Number of Bridges",
        height=420,
    )
    return fig


def per_ds_correct_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = [ds.capitalize() for ds in DS_ORDER]

    if "RF_DS" in df.columns:
        rf = [((df["Observed_DS"] == ds) & (df["RF_DS"] == ds)).sum() for ds in DS_ORDER]
        fig.add_bar(name="RF Correct", x=x, y=rf, marker_color="#2196F3")

    if "GMM_MAP_DS" in df.columns:
        gmm = [((df["Observed_DS"] == ds) & (df["GMM_MAP_DS"] == ds)).sum() for ds in DS_ORDER]
        fig.add_bar(name="GMM MAP Correct", x=x, y=gmm, marker_color="#FF9800")

    obs = [(df["Observed_DS"] == ds).sum() for ds in DS_ORDER]
    fig.add_scatter(name="Observed Total", x=x, y=obs, mode="lines+markers", line=dict(color="green", width=3))

    fig.update_layout(
        barmode="group",
        title="Per Damage-State Correct Predictions",
        xaxis_title="Damage State",
        yaxis_title="Count",
        height=420,
    )
    return fig


def missed_false_alarm_chart(df: pd.DataFrame) -> go.Figure:
    labels, missed, false_alarms = [], [], []

    if "RF_DS" in df.columns:
        m = metrics_from_prediction(df, "RF_DS")
        labels.append("RF")
        missed.append(m["missed_damaged"])
        false_alarms.append(m["false_alarms"])

    if "GMM_MAP_DS" in df.columns:
        m = metrics_from_prediction(df, "GMM_MAP_DS")
        labels.append("GMM MAP")
        missed.append(m["missed_damaged"])
        false_alarms.append(m["false_alarms"])

    fig = go.Figure()
    fig.add_bar(name="Missed Damaged", x=labels, y=missed, marker_color="#F44336")
    fig.add_bar(name="False Alarms", x=labels, y=false_alarms, marker_color="#9C27B0")

    fig.update_layout(
        barmode="group",
        title="Missed Damaged vs False Alarms",
        yaxis_title="Count",
        height=420,
    )
    return fig


def make_feature_chart(feat_df: pd.DataFrame) -> go.Figure:
    if feat_df.empty:
        return go.Figure()

    feat_cols = {c.lower(): c for c in feat_df.columns}
    feature_col = feat_cols.get("feature")
    importance_col = feat_cols.get("importance")
    if feature_col is None or importance_col is None:
        return go.Figure()

    chart_df = feat_df[[feature_col, importance_col]].rename(
        columns={feature_col: "Feature", importance_col: "Importance"}
    ).sort_values("Importance", ascending=True)

    fig = go.Figure()
    fig.add_bar(
        x=chart_df["Importance"],
        y=chart_df["Feature"],
        orientation="h",
        marker_color="#2196F3",
    )
    fig.update_layout(
        title="Random Forest Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=420,
    )
    return fig


def build_map(df: pd.DataFrame, show_rf: bool, show_gmm: bool) -> go.Figure:
    fig = go.Figure()

    # observed damaged = solid markers
    obs_dmg = df[df["Observed_DS"] != "none"].copy()
    for ds in ["slight", "moderate", "extensive", "complete"]:
        part = obs_dmg[obs_dmg["Observed_DS"] == ds]
        if len(part) == 0:
            continue

        fig.add_trace(
            go.Scattermapbox(
                lat=part["Latitude"],
                lon=part["Longitude"],
                mode="markers",
                name=f"Observed {ds}",
                marker=dict(size=10, color=COLOR_MAP[ds], opacity=0.90),
                customdata=np.stack(
                    [
                        part["Structure_ID"].astype(str),
                        part["Observed_DS"],
                        part["Sa_1s_g"].astype(float),
                        part["HWB_Class"].astype(str),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>Structure ID:</b> %{customdata[0]}<br>"
                    "<b>Observed:</b> %{customdata[1]}<br>"
                    "<b>Sa(1.0s):</b> %{customdata[2]:.3f}<br>"
                    "<b>HWB:</b> %{customdata[3]}<extra></extra>"
                ),
            )
        )

    def add_overprediction(pred_col: str, label: str, color: str):
        tmp = df.copy()
        tmp["obs_idx"] = severity(tmp["Observed_DS"])
        tmp["pred_idx"] = severity(tmp[pred_col])
        over = tmp[tmp["pred_idx"] > tmp["obs_idx"]].copy()

        if len(over) == 0:
            return

        fig.add_trace(
            go.Scattermapbox(
                lat=over["Latitude"],
                lon=over["Longitude"],
                mode="markers",
                name=label,
                marker=dict(
                    size=17,
                    color="rgba(0,0,0,0)",
                    symbol="circle-open",
                    line=dict(width=2.5, color=color),
                ),
                customdata=np.stack(
                    [
                        over["Structure_ID"].astype(str),
                        over["Observed_DS"],
                        over[pred_col].astype(str),
                        over["Sa_1s_g"].astype(float),
                        over["HWB_Class"].astype(str),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>Structure ID:</b> %{customdata[0]}<br>"
                    "<b>Observed:</b> %{customdata[1]}<br>"
                    "<b>Predicted:</b> %{customdata[2]}<br>"
                    "<b>Sa(1.0s):</b> %{customdata[3]:.3f}<br>"
                    "<b>HWB:</b> %{customdata[4]}<extra></extra>"
                ),
            )
        )

    if show_rf and "RF_DS" in df.columns:
        add_overprediction("RF_DS", "RF overprediction", "#2196F3")

    if show_gmm and "GMM_MAP_DS" in df.columns:
        add_overprediction("GMM_MAP_DS", "GMM overprediction", "#FF9800")

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=8,
        mapbox_center={"lat": float(df["Latitude"].mean()), "lon": float(df["Longitude"].mean())},
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
        height=700,
        title="Observed damaged bridges (solid) + model overpredictions (open circles)",
        legend=dict(orientation="h"),
    )
    return fig


def bridge_detail_df(row: pd.Series) -> pd.DataFrame:
    fields = {
        "Structure ID": row.get("Structure_ID"),
        "HWB Class": row.get("HWB_Class"),
        "Year Built": row.get("Year_Built", np.nan),
        "Design Era": row.get("Design_Era", "NA"),
        "Num Spans": row.get("Num_Spans", np.nan),
        "Sa(1.0s)": row.get("Sa_1s_g"),
        "PGA": row.get("PGA_g", np.nan),
        "Observed DS": row.get("Observed_DS"),
        "RF DS": row.get("RF_DS", "NA"),
        "GMM MAP DS": row.get("GMM_MAP_DS", "NA"),
        "Observed Repair Action": row.get("Obs_Repair_Action"),
        "RF Repair Action": row.get("RF_Repair_Action", "NA"),
        "GMM MAP Repair Action": row.get("GMM_MAP_Repair_Action", "NA"),
    }
    return pd.DataFrame({"Field": list(fields.keys()), "Value": list(fields.values())})


def probability_df(row: pd.Series) -> pd.DataFrame:
    rows = {"Damage State": DS_ORDER}

    rf_cols = ["RF_P_None", "RF_P_Slight", "RF_P_Moderate", "RF_P_Extensive", "RF_P_Complete"]
    if all(c in row.index for c in rf_cols):
        rows["RF Probability"] = [row[c] for c in rf_cols]

    gmm_cols = ["GMM_P_None", "GMM_P_Slight", "GMM_P_Moderate", "GMM_P_Extensive", "GMM_P_Complete"]
    if all(c in row.index for c in gmm_cols):
        rows["GMM Probability"] = [row[c] for c in gmm_cols]

    return pd.DataFrame(rows)


def lognormal_cdf(sa: np.ndarray, median: float, beta: float, k: float = 1.0) -> np.ndarray:
    if median <= 0 or beta <= 0:
        return np.zeros_like(sa, dtype=float)
    return norm.cdf((np.log(sa) - np.log(median * k)) / beta)


def fragility_curve_figure(row: pd.Series, frag_df: pd.DataFrame, beta: float, k: float) -> go.Figure | None:
    if frag_df.empty or "HWB_Class" not in row.index:
        return None

    hwb = str(row["HWB_Class"]).strip().lower()
    frag_row = frag_df[frag_df["HWB_Class"] == hwb]
    if frag_row.empty:
        return None

    frag_row = frag_row.iloc[0]
    sa_actual = float(row["Sa_1s_g"])
    sa_max = max(2.5, sa_actual * 1.4)
    sa_range = np.linspace(0.01, sa_max, 250)

    fig = go.Figure()
    for ds, color in [("slight", COLOR_MAP["slight"]),
                      ("moderate", COLOR_MAP["moderate"]),
                      ("extensive", COLOR_MAP["extensive"]),
                      ("complete", COLOR_MAP["complete"])]:
        median = float(frag_row[f"{ds}_median"])
        curve = lognormal_cdf(sa_range, median, beta, k)
        fig.add_scatter(
            x=sa_range,
            y=curve,
            mode="lines",
            name=f"P(≥ {ds})",
            line=dict(color=color, width=3),
        )

    fig.add_vline(
        x=sa_actual,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Bridge Sa={sa_actual:.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Fragility Curves — {hwb.upper()}",
        xaxis_title="Sa(1.0s) [g]",
        yaxis_title="Exceedance Probability",
        yaxis_range=[0, 1],
        height=450,
    )
    return fig


# ============================================================
# LOAD DATA
# ============================================================
try:
    bridges_raw, feat_df, frag_raw = load_all()
    df = normalize_bridge_columns(bridges_raw)
    frag_df = normalize_fragility_columns(frag_raw)
except Exception as e:
    st.error("Could not load input files.")
    st.exception(e)
    st.stop()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.title("Filters")

selected_hwb = st.sidebar.multiselect(
    "HWB Class",
    sorted(df["HWB_Class"].dropna().unique().tolist())
)

design_era_col_exists = "Design_Era" in df.columns
selected_era = []
if design_era_col_exists:
    selected_era = st.sidebar.multiselect(
        "Design Era",
        sorted(df["Design_Era"].dropna().unique().tolist())
    )

sa_min = float(df["Sa_1s_g"].min())
sa_max = float(df["Sa_1s_g"].max())
sa_range = st.sidebar.slider(
    "Sa(1.0s) Range",
    min_value=sa_min,
    max_value=sa_max,
    value=(sa_min, sa_max),
)

show_rf = st.sidebar.checkbox("Show RF overprediction", value=True)
show_gmm = st.sidebar.checkbox("Show GMM overprediction", value=True)

beta_val = st.sidebar.number_input("Fragility beta", min_value=0.1, max_value=2.0, value=DEFAULT_BETA, step=0.05)
k_val = st.sidebar.number_input("Fragility k", min_value=0.1, max_value=5.0, value=DEFAULT_K, step=0.1)

filtered = df.copy()
filtered = filtered[(filtered["Sa_1s_g"] >= sa_range[0]) & (filtered["Sa_1s_g"] <= sa_range[1])]
if selected_hwb:
    filtered = filtered[filtered["HWB_Class"].isin(selected_hwb)]
if selected_era and design_era_col_exists:
    filtered = filtered[filtered["Design_Era"].isin(selected_era)]

# persistent clicked bridge
if "selected_bridge_id" not in st.session_state:
    st.session_state["selected_bridge_id"] = None

# ============================================================
# HEADER
# ============================================================
st.title("Bridge Damage Dashboard")
st.caption("Observed vs Random Forest vs Gaussian Mixture Model")

# ============================================================
# TOP TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["Damage View", "Bridge Explorer", "Repair Priority", "Model Insights"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bridges", len(filtered))
    c2.metric("Observed Damaged", int((filtered["Observed_DS"] != "none").sum()))

    if "RF_DS" in filtered.columns:
        rf_m = metrics_from_prediction(filtered, "RF_DS")
        c3.metric("RF Damaged Accuracy", f'{rf_m["damaged_accuracy"]:.1f}%')
    if "GMM_MAP_DS" in filtered.columns:
        gmm_m = metrics_from_prediction(filtered, "GMM_MAP_DS")
        c4.metric("GMM Damaged Accuracy", f'{gmm_m["damaged_accuracy"]:.1f}%')

    map_fig = build_map(filtered, show_rf=show_rf, show_gmm=show_gmm)
    selected_points = plotly_events(
        map_fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=700,
        key="damage_map",
    )

    if selected_points:
        point = selected_points[0]
        curve_no = point["curveNumber"]
        point_no = point["pointIndex"]
        customdata = map_fig.data[curve_no].customdata[point_no]
        st.session_state["selected_bridge_id"] = str(customdata[0])

    st.info("Click a marker on the map to see bridge details and fragility curves below.")

    if st.session_state["selected_bridge_id"] is not None:
        selected_id = str(st.session_state["selected_bridge_id"])
        match = filtered[filtered["Structure_ID"].astype(str) == selected_id]
        if len(match) > 0:
            row = match.iloc[0]

            st.subheader(f"Clicked Bridge: {selected_id}")
            left, right = st.columns([1, 1])

            with left:
                st.dataframe(bridge_detail_df(row), use_container_width=True, hide_index=True)

                probs = probability_df(row)
                if probs.shape[1] > 1:
                    st.dataframe(probs.style.format(precision=3), use_container_width=True, hide_index=True)

            with right:
                frag_fig = fragility_curve_figure(row, frag_df, beta=beta_val, k=k_val)
                if frag_fig is not None:
                    st.plotly_chart(frag_fig, use_container_width=True)
                else:
                    st.warning("Fragility parameters for this bridge class were not found.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(damage_count_chart(filtered), use_container_width=True)
    with right:
        st.plotly_chart(repair_count_chart(filtered), use_container_width=True)

    st.plotly_chart(per_ds_correct_chart(filtered), use_container_width=True)
    st.plotly_chart(missed_false_alarm_chart(filtered), use_container_width=True)

with tab2:
    ids = filtered["Structure_ID"].astype(str).tolist()
    default_index = 0
    if st.session_state["selected_bridge_id"] is not None and st.session_state["selected_bridge_id"] in ids:
        default_index = ids.index(st.session_state["selected_bridge_id"])

    selected_id = st.selectbox("Select Structure ID", ids, index=default_index)
    row = filtered[filtered["Structure_ID"].astype(str) == str(selected_id)].iloc[0]

    left, right = st.columns([1, 1])
    with left:
        st.dataframe(bridge_detail_df(row), use_container_width=True, hide_index=True)
        probs = probability_df(row)
        if probs.shape[1] > 1:
            st.dataframe(probs.style.format(precision=3), use_container_width=True, hide_index=True)

    with right:
        frag_fig = fragility_curve_figure(row, frag_df, beta=beta_val, k=k_val)
        if frag_fig is not None:
            st.plotly_chart(frag_fig, use_container_width=True)
        else:
            st.warning("Fragility parameters for this bridge class were not found.")

with tab3:
    options = []
    if "RF_Priority" in filtered.columns:
        options.append(("Random Forest", "RF_Priority", "RF_DS", "RF_Repair_Action"))
    if "GMM_MAP_Priority" in filtered.columns:
        options.append(("GMM MAP", "GMM_MAP_Priority", "GMM_MAP_DS", "GMM_MAP_Repair_Action"))

    if options:
        model_names = [x[0] for x in options]
        selected_model = st.radio("Priority source", model_names, horizontal=True)
        choice = next(x for x in options if x[0] == selected_model)
        _, pri_col, ds_col, action_col = choice

        min_priority = st.slider("Minimum priority", 0, 4, 2)
        top_df = filtered[filtered[pri_col] >= min_priority].sort_values(pri_col, ascending=False)

        show_cols = ["Structure_ID", "HWB_Class", "Year_Built", "Sa_1s_g", ds_col, action_col, pri_col]
        show_cols = [c for c in show_cols if c in top_df.columns]

        st.dataframe(top_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("Priority columns were not found in ml_bridges.csv.")

with tab4:
    if not feat_df.empty:
        st.plotly_chart(make_feature_chart(feat_df), use_container_width=True)
    else:
        st.warning("Feature importance file not found.")

    rows = []
    if "RF_DS" in filtered.columns:
        m = metrics_from_prediction(filtered, "RF_DS")
        rows.append(["Random Forest", m["overall_accuracy"], m["damaged_accuracy"], m["missed_damaged"], m["false_alarms"]])
    if "GMM_MAP_DS" in filtered.columns:
        m = metrics_from_prediction(filtered, "GMM_MAP_DS")
        rows.append(["GMM MAP", m["overall_accuracy"], m["damaged_accuracy"], m["missed_damaged"], m["false_alarms"]])

    if rows:
        summary_df = pd.DataFrame(
            rows,
            columns=["Model", "Overall Accuracy (%)", "Damaged Accuracy (%)", "Missed Damaged", "False Alarms"]
        )
        st.dataframe(summary_df.round(2), use_container_width=True, hide_index=True)

    st.plotly_chart(per_ds_correct_chart(filtered), use_container_width=True)
    st.plotly_chart(missed_false_alarm_chart(filtered), use_container_width=True)

st.markdown("---")
st.caption("Observed damaged bridges are solid markers. RF and GMM overpredictions are open circles. Click a bridge to inspect details and fragility curves.")
