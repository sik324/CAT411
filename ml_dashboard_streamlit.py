from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Bridge Damage Dashboard",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================
DATA_DIR = Path("data")
BRIDGES_CSV = DATA_DIR / "ml_bridges.csv"
FEATURE_CSV = DATA_DIR / "ml_feature_importance.csv"

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


# ============================================================
# HELPERS
# ============================================================
def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True):
    existing = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in existing:
            return existing[cand.lower()]
    if required:
        raise KeyError(f"Could not find required column. Tried: {candidates}")
    return None


@st.cache_data
def load_data():
    bridges = pd.read_csv(BRIDGES_CSV)

    feat = pd.DataFrame()
    if FEATURE_CSV.exists():
        feat = pd.read_csv(FEATURE_CSV)

    return bridges, feat


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    colmap = {
        "structure_id": find_col(out, ["Structure_ID", "structure_id", "structure_number", "Structure_Number"]),
        "lat": find_col(out, ["Latitude", "latitude", "lat"]),
        "lon": find_col(out, ["Longitude", "longitude", "lon"]),
        "hwb": find_col(out, ["HWB_Class", "hwb_class", "HWB"]),
        "year_built": find_col(out, ["Year_Built", "year_built"]),
        "design_era": find_col(out, ["Design_Era", "design_era"]),
        "num_spans": find_col(out, ["Num_Spans", "num_spans"], required=False),
        "sa": find_col(out, ["Sa_1s_g", "sa1s_shakemap", "sa_1s_g"]),
        "pga": find_col(out, ["PGA_g", "pga_shakemap", "pga_g"], required=False),
        "obs": find_col(out, ["Observed_DS", "obs", "Observed_damage"]),
        "rf": find_col(out, ["RF_DS", "pred_rf", "rf_ds"], required=False),
        "gmm_map": find_col(out, ["GMM_DS", "pred_gmm_map", "gmm_map", "pred_gmm"], required=False),
        "gmm_cost": find_col(out, ["pred_gmm_cost", "gmm_cost"], required=False),
    }

    rename_dict = {
        colmap["structure_id"]: "Structure_ID",
        colmap["lat"]: "Latitude",
        colmap["lon"]: "Longitude",
        colmap["hwb"]: "HWB_Class",
        colmap["year_built"]: "Year_Built",
        colmap["design_era"]: "Design_Era",
        colmap["sa"]: "Sa_1s_g",
        colmap["obs"]: "Observed_DS",
    }
    if colmap["num_spans"]:
        rename_dict[colmap["num_spans"]] = "Num_Spans"
    if colmap["pga"]:
        rename_dict[colmap["pga"]] = "PGA_g"
    if colmap["rf"]:
        rename_dict[colmap["rf"]] = "RF_DS"
    if colmap["gmm_map"]:
        rename_dict[colmap["gmm_map"]] = "GMM_MAP_DS"
    if colmap["gmm_cost"]:
        rename_dict[colmap["gmm_cost"]] = "GMM_COST_DS"

    out = out.rename(columns=rename_dict)

    # Normalize strings
    for c in ["Observed_DS", "RF_DS", "GMM_MAP_DS", "GMM_COST_DS", "Design_Era", "HWB_Class"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.lower()

    # Derived fields
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
    traces = [("Observed", "Observed_DS", "#4CAF50")]
    if "RF_DS" in df.columns:
        traces.append(("Random Forest", "RF_DS", "#2196F3"))
    if "GMM_MAP_DS" in df.columns:
        traces.append(("GMM MAP", "GMM_MAP_DS", "#FF9800"))
    if "GMM_COST_DS" in df.columns:
        traces.append(("GMM Cost", "GMM_COST_DS", "#9C27B0"))

    fig = go.Figure()
    x = [ds.capitalize() for ds in DS_ORDER]

    for label, col, color in traces:
        y = [(df[col] == ds).sum() for ds in DS_ORDER]
        fig.add_bar(name=label, x=x, y=y, marker_color=color)

    fig.update_layout(
        barmode="group",
        title="Damage State Counts",
        xaxis_title="Damage State",
        yaxis_title="Number of Bridges",
        height=430,
    )
    return fig


def repair_count_chart(df: pd.DataFrame) -> go.Figure:
    traces = [("Observed", "Obs_Repair_Category", "#4CAF50")]
    if "RF_Repair_Category" in df.columns:
        traces.append(("Random Forest", "RF_Repair_Category", "#2196F3"))
    if "GMM_MAP_Repair_Category" in df.columns:
        traces.append(("GMM MAP", "GMM_MAP_Repair_Category", "#FF9800"))
    if "GMM_COST_Repair_Category" in df.columns:
        traces.append(("GMM Cost", "GMM_COST_Repair_Category", "#9C27B0"))

    fig = go.Figure()
    for label, col, color in traces:
        y = [(df[col] == c.lower()).sum() if df[col].dtype == object else (df[col] == c).sum() for c in REPAIR_CATEGORY.values()]
    # safer explicit categories
    cats = ["Safe", "Needs Repair", "Needs Closure", "Needs Replacement"]
    for label, col, color in traces:
        y = [(df[col].astype(str).str.lower() == c.lower()).sum() for c in cats]
        fig.add_bar(name=label, x=cats, y=y, marker_color=color)

    fig.update_layout(
        barmode="group",
        title="Repair Category Counts",
        xaxis_title="Repair Category",
        yaxis_title="Number of Bridges",
        height=430,
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
        height=430,
    )
    return fig


def missed_false_alarm_chart(df: pd.DataFrame) -> go.Figure:
    labels = []
    missed = []
    false_alarms = []

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

    if "GMM_COST_DS" in df.columns:
        m = metrics_from_prediction(df, "GMM_COST_DS")
        labels.append("GMM Cost")
        missed.append(m["missed_damaged"])
        false_alarms.append(m["false_alarms"])

    fig = go.Figure()
    fig.add_bar(name="Missed Damaged", x=labels, y=missed, marker_color="#F44336")
    fig.add_bar(name="False Alarms", x=labels, y=false_alarms, marker_color="#9C27B0")

    fig.update_layout(
        barmode="group",
        title="Missed Damaged Bridges vs False Alarms",
        yaxis_title="Count",
        height=430,
    )
    return fig


def make_feature_chart(feat_df: pd.DataFrame) -> go.Figure:
    feat_cols = {c.lower(): c for c in feat_df.columns}
    feature_col = feat_cols.get("feature")
    importance_col = feat_cols.get("importance")

    chart_df = feat_df[[feature_col, importance_col]].rename(
        columns={feature_col: "Feature", importance_col: "Importance"}
    ).sort_values("Importance", ascending=True)

    fig = go.Figure()
    fig.add_bar(
        x=chart_df["Importance"],
        y=chart_df["Feature"],
        orientation="h",
        marker_color="#2196F3",
        name="RF Importance",
    )
    fig.update_layout(
        title="Random Forest Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=430,
    )
    return fig


def map_figure(df: pd.DataFrame, overlay_model: str) -> go.Figure:
    fig = go.Figure()

    # Observed damaged bridges: solid markers
    obs_dmg = df[df["Observed_DS"] != "none"].copy()
    for ds in [d for d in DS_ORDER if d != "none"]:
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
                text=part["Structure_ID"].astype(str),
                hovertemplate=(
                    "<b>Structure ID:</b> %{text}<br>"
                    "<b>Observed:</b> " + ds + "<br>"
                    "<b>Sa(1.0s):</b> %{customdata[0]:.3f}<br>"
                    "<b>HWB:</b> %{customdata[1]}<extra></extra>"
                ),
                customdata=np.stack(
                    [part["Sa_1s_g"], part["HWB_Class"]],
                    axis=1
                ),
            )
        )

    def add_overprediction_trace(pred_col: str, label: str, color: str):
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
                    size=16,
                    color="rgba(0,0,0,0)",
                    opacity=1.0,
                    symbol="circle-open",
                    line=dict(width=2.5, color=color),
                ),
                text=over["Structure_ID"].astype(str),
                hovertemplate=(
                    "<b>Structure ID:</b> %{text}<br>"
                    "<b>Observed:</b> %{customdata[0]}<br>"
                    "<b>Predicted:</b> %{customdata[1]}<br>"
                    "<b>Sa(1.0s):</b> %{customdata[2]:.3f}<extra></extra>"
                ),
                customdata=np.stack(
                    [over["Observed_DS"], over[pred_col], over["Sa_1s_g"]],
                    axis=1
                ),
            )
        )

    if overlay_model in ["RF", "Both"] and "RF_DS" in df.columns:
        add_overprediction_trace("RF_DS", "RF overprediction", "#2196F3")

    if overlay_model in ["GMM MAP", "Both"] and "GMM_MAP_DS" in df.columns:
        add_overprediction_trace("GMM_MAP_DS", "GMM MAP overprediction", "#FF9800")

    if overlay_model in ["GMM Cost", "Both"] and "GMM_COST_DS" in df.columns:
        add_overprediction_trace("GMM_COST_DS", "GMM Cost overprediction", "#9C27B0")

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=8,
        mapbox_center={"lat": float(df["Latitude"].mean()), "lon": float(df["Longitude"].mean())},
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
        height=700,
        title="Observed Damaged Bridges (solid) and Model Overpredictions (open circles)",
        legend=dict(orientation="h"),
    )
    return fig


# ============================================================
# LOAD
# ============================================================
try:
    bridges_raw, feat_df = load_data()
    df = normalize_columns(bridges_raw)
except Exception as e:
    st.error("Could not load input files.")
    st.exception(e)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Dashboard Controls")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Damage Comparison", "Map View", "Bridge Explorer", "Repair Priority", "Model Insights"],
)

selected_hwb = st.sidebar.multiselect(
    "Filter HWB Class",
    sorted(df["HWB_Class"].dropna().unique().tolist())
)
selected_era = st.sidebar.multiselect(
    "Filter Design Era",
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

filtered = df.copy()
filtered = filtered[(filtered["Sa_1s_g"] >= sa_range[0]) & (filtered["Sa_1s_g"] <= sa_range[1])]
if selected_hwb:
    filtered = filtered[filtered["HWB_Class"].isin(selected_hwb)]
if selected_era:
    filtered = filtered[filtered["Design_Era"].isin(selected_era)]

# ============================================================
# HEADER
# ============================================================
st.title("Bridge Damage Dashboard")
st.caption("Observed vs Random Forest vs Gaussian Mixture Model")

# ============================================================
# PAGES
# ============================================================
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bridges", len(filtered))
    c2.metric("Observed Damaged", int((filtered["Observed_DS"] != "none").sum()))

    if "RF_DS" in filtered.columns:
        rf_m = metrics_from_prediction(filtered, "RF_DS")
        c3.metric("RF Overall Accuracy", f'{rf_m["overall_accuracy"]:.1f}%')
        c4.metric("RF Damaged Accuracy", f'{rf_m["damaged_accuracy"]:.1f}%')

    c5, c6, c7, c8 = st.columns(4)
    if "GMM_MAP_DS" in filtered.columns:
        gmm_m = metrics_from_prediction(filtered, "GMM_MAP_DS")
        c5.metric("GMM MAP Overall", f'{gmm_m["overall_accuracy"]:.1f}%')
        c6.metric("GMM MAP Damaged", f'{gmm_m["damaged_accuracy"]:.1f}%')
    if "GMM_COST_DS" in filtered.columns:
        gmmc_m = metrics_from_prediction(filtered, "GMM_COST_DS")
        c7.metric("GMM Cost Overall", f'{gmmc_m["overall_accuracy"]:.1f}%')
        c8.metric("GMM Cost Damaged", f'{gmmc_m["damaged_accuracy"]:.1f}%')

    left, right = st.columns(2)
    with left:
        st.plotly_chart(damage_count_chart(filtered), use_container_width=True)
    with right:
        st.plotly_chart(repair_count_chart(filtered), use_container_width=True)

elif page == "Damage Comparison":
    left, right = st.columns(2)
    with left:
        st.plotly_chart(damage_count_chart(filtered), use_container_width=True)
    with right:
        st.plotly_chart(repair_count_chart(filtered), use_container_width=True)

    st.plotly_chart(per_ds_correct_chart(filtered), use_container_width=True)
    st.plotly_chart(missed_false_alarm_chart(filtered), use_container_width=True)

    rows = []
    if "RF_DS" in filtered.columns:
        m = metrics_from_prediction(filtered, "RF_DS")
        rows.append(["Random Forest", m["overall_accuracy"], m["damaged_accuracy"], m["missed_damaged"], m["false_alarms"]])
    if "GMM_MAP_DS" in filtered.columns:
        m = metrics_from_prediction(filtered, "GMM_MAP_DS")
        rows.append(["GMM MAP", m["overall_accuracy"], m["damaged_accuracy"], m["missed_damaged"], m["false_alarms"]])
    if "GMM_COST_DS" in filtered.columns:
        m = metrics_from_prediction(filtered, "GMM_COST_DS")
        rows.append(["GMM Cost", m["overall_accuracy"], m["damaged_accuracy"], m["missed_damaged"], m["false_alarms"]])

    if rows:
        summary_df = pd.DataFrame(
            rows,
            columns=["Model", "Overall Accuracy (%)", "Damaged Accuracy (%)", "Missed Damaged", "False Alarms"]
        )
        st.dataframe(summary_df.round(2), use_container_width=True, hide_index=True)

elif page == "Map View":
    overlay_choices = []
    if "RF_DS" in filtered.columns:
        overlay_choices.append("RF")
    if "GMM_MAP_DS" in filtered.columns:
        overlay_choices.append("GMM MAP")
    if "GMM_COST_DS" in filtered.columns:
        overlay_choices.append("GMM Cost")
    if len(overlay_choices) >= 2:
        overlay_choices.append("Both")

    overlay_model = st.selectbox("Overlay model on map", overlay_choices)
    st.plotly_chart(map_figure(filtered, overlay_model), use_container_width=True)
    st.info("Solid markers = observed damaged bridges. Open circles = bridges where model prediction is more severe than the observed damage state.")

elif page == "Bridge Explorer":
    ids = filtered["Structure_ID"].astype(str).tolist()
    selected_id = st.selectbox("Select Structure ID", ids)
    row = filtered[filtered["Structure_ID"].astype(str) == str(selected_id)].iloc[0]

    details = {
        "Structure ID": row["Structure_ID"],
        "HWB Class": row["HWB_Class"],
        "Year Built": row["Year_Built"],
        "Design Era": row["Design_Era"],
        "Num Spans": row.get("Num_Spans", np.nan),
        "Sa(1.0s)": row["Sa_1s_g"],
        "PGA": row.get("PGA_g", np.nan),
        "Observed DS": row["Observed_DS"],
        "RF DS": row.get("RF_DS", "NA"),
        "GMM MAP DS": row.get("GMM_MAP_DS", "NA"),
        "GMM Cost DS": row.get("GMM_COST_DS", "NA"),
        "Observed Repair": row["Obs_Repair_Action"],
        "RF Repair": row.get("RF_Repair_Action", "NA"),
        "GMM MAP Repair": row.get("GMM_MAP_Repair_Action", "NA"),
        "GMM Cost Repair": row.get("GMM_COST_Repair_Action", "NA"),
    }
    st.dataframe(pd.DataFrame(details.items(), columns=["Field", "Value"]), use_container_width=True, hide_index=True)

elif page == "Repair Priority":
    options = []
    if "RF_Priority" in filtered.columns:
        options.append(("Random Forest", "RF_Priority", "RF_DS", "RF_Repair_Action"))
    if "GMM_MAP_Priority" in filtered.columns:
        options.append(("GMM MAP", "GMM_MAP_Priority", "GMM_MAP_DS", "GMM_MAP_Repair_Action"))
    if "GMM_COST_Priority" in filtered.columns:
        options.append(("GMM Cost", "GMM_COST_Priority", "GMM_COST_DS", "GMM_COST_Repair_Action"))

    model_names = [x[0] for x in options]
    selected_model = st.radio("Priority source", model_names, horizontal=True)
    choice = next(x for x in options if x[0] == selected_model)
    _, pri_col, ds_col, action_col = choice

    min_priority = st.slider("Minimum priority", 0, 4, 2)
    top_df = filtered[filtered[pri_col] >= min_priority].sort_values(pri_col, ascending=False)

    show_cols = ["Structure_ID", "HWB_Class", "Year_Built", "Design_Era", "Sa_1s_g", ds_col, action_col, pri_col]
    st.dataframe(top_df[show_cols], use_container_width=True, hide_index=True)

elif page == "Model Insights":
    if not feat_df.empty:
        st.plotly_chart(make_feature_chart(feat_df), use_container_width=True)
    else:
        st.warning("Feature importance file not found.")

    st.plotly_chart(per_ds_correct_chart(filtered), use_container_width=True)
    st.plotly_chart(missed_false_alarm_chart(filtered), use_container_width=True)

st.markdown("---")
st.caption("Data source: repository CSV files under /data")
