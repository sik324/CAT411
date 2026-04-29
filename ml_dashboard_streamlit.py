from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
def find_col(df: pd.DataFrame, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(f"Missing column. Tried: {candidates}")
    return None


@st.cache_data(show_spinner=False)
def load_raw_data():
    bridges = pd.read_csv(BRIDGES_CSV)

    feat = pd.DataFrame()
    if FEATURE_CSV.exists():
        feat = pd.read_csv(FEATURE_CSV)

    return bridges, feat


def normalize_bridges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    colmap = {
        "id": find_col(out, ["Structure_ID", "structure_id", "Structure_Number", "structure_number"]),
        "lat": find_col(out, ["Latitude", "latitude", "lat"]),
        "lon": find_col(out, ["Longitude", "longitude", "lon"]),
        "hwb": find_col(out, ["HWB_Class", "hwb_class"], required=False),
        "year": find_col(out, ["Year_Built", "year_built"], required=False),
        "era": find_col(out, ["Design_Era", "design_era"], required=False),
        "spans": find_col(out, ["Num_Spans", "num_spans"], required=False),
        "sa": find_col(out, ["Sa_1s_g", "sa1s_shakemap", "sa_1s_g"]),
        "pga": find_col(out, ["PGA_g", "pga_shakemap", "pga_g"], required=False),
        "obs": find_col(out, ["Observed_DS", "obs", "Observed_damage"]),
        "rf": find_col(out, ["RF_DS", "pred_rf", "rf_ds"], required=False),
        "gmm": find_col(out, ["GMM_DS", "pred_gmm_map", "GMM_MAP_DS", "gmm_map", "pred_gmm"], required=False),
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

    rename = {
        colmap["id"]: "Structure_ID",
        colmap["lat"]: "Latitude",
        colmap["lon"]: "Longitude",
        colmap["sa"]: "Sa_1s_g",
        colmap["obs"]: "Observed_DS",
    }

    optional_map = {
        "hwb": "HWB_Class",
        "year": "Year_Built",
        "era": "Design_Era",
        "spans": "Num_Spans",
        "pga": "PGA_g",
        "rf": "RF_DS",
        "gmm": "GMM_DS",
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
            rename[colmap[key]] = new_name

    out = out.rename(columns=rename)

    # normalize string columns
    for c in ["Observed_DS", "RF_DS", "GMM_DS", "HWB_Class", "Design_Era"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.lower()

    return out


@st.cache_data(show_spinner=False)
def prepare_data():
    bridges_raw, feat_df = load_raw_data()
    bridges = normalize_bridges(bridges_raw)
    return bridges, feat_df


def metrics(df: pd.DataFrame, pred_col: str):
    obs = df["Observed_DS"]
    pred = df[pred_col]
    overall = (obs == pred).mean() * 100
    damaged_mask = obs != "none"
    damaged_acc = (obs[damaged_mask] == pred[damaged_mask]).mean() * 100 if damaged_mask.any() else 0.0
    missed = int(((obs != "none") & (pred == "none")).sum())
    false_alarms = int(((obs == "none") & (pred != "none")).sum())
    return overall, damaged_acc, missed, false_alarms


def add_repair_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Obs_Repair_Category"] = out["Observed_DS"].map(REPAIR_CATEGORY)
    out["Obs_Repair_Action"] = out["Observed_DS"].map(REPAIR_ACTION)
    out["Obs_Priority"] = out["Observed_DS"].map(REPAIR_PRIORITY)

    if "RF_DS" in out.columns:
        out["RF_Repair_Category"] = out["RF_DS"].map(REPAIR_CATEGORY)
        out["RF_Repair_Action"] = out["RF_DS"].map(REPAIR_ACTION)
        out["RF_Priority"] = out["RF_DS"].map(REPAIR_PRIORITY)

    if "GMM_DS" in out.columns:
        out["GMM_Repair_Category"] = out["GMM_DS"].map(REPAIR_CATEGORY)
        out["GMM_Repair_Action"] = out["GMM_DS"].map(REPAIR_ACTION)
        out["GMM_Priority"] = out["GMM_DS"].map(REPAIR_PRIORITY)

    return out


def filter_df(df: pd.DataFrame, selected_hwb, selected_era, sa_range):
    out = df.copy()
    out = out[(out["Sa_1s_g"] >= sa_range[0]) & (out["Sa_1s_g"] <= sa_range[1])]

    if selected_hwb and "HWB_Class" in out.columns:
        out = out[out["HWB_Class"].isin(selected_hwb)]

    if selected_era and "Design_Era" in out.columns:
        out = out[out["Design_Era"].isin(selected_era)]

    return out


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

    if "GMM_DS" in df.columns:
        fig.add_bar(
            name="GMM",
            x=x,
            y=[(df["GMM_DS"] == ds).sum() for ds in DS_ORDER],
            marker_color="#FF9800",
        )

    fig.update_layout(
        barmode="group",
        title="Damage State Counts",
        xaxis_title="Damage State",
        yaxis_title="Number of Bridges",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def repair_count_chart(df: pd.DataFrame) -> go.Figure:
    cats = ["Safe", "Needs Repair", "Needs Closure", "Needs Replacement"]
    obs_map = {
        "none": "Safe",
        "slight": "Needs Repair",
        "moderate": "Needs Repair",
        "extensive": "Needs Closure",
        "complete": "Needs Replacement",
    }

    fig = go.Figure()

    obs_repair = df["Observed_DS"].map(obs_map)
    fig.add_bar(
        name="Observed",
        x=cats,
        y=[(obs_repair == c).sum() for c in cats],
        marker_color="#4CAF50",
    )

    if "RF_DS" in df.columns:
        rf_repair = df["RF_DS"].map(obs_map)
        fig.add_bar(
            name="RF",
            x=cats,
            y=[(rf_repair == c).sum() for c in cats],
            marker_color="#2196F3",
        )

    if "GMM_DS" in df.columns:
        gmm_repair = df["GMM_DS"].map(obs_map)
        fig.add_bar(
            name="GMM",
            x=cats,
            y=[(gmm_repair == c).sum() for c in cats],
            marker_color="#FF9800",
        )

    fig.update_layout(
        barmode="group",
        title="Repair Category Counts",
        xaxis_title="Repair Category",
        yaxis_title="Number of Bridges",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def feature_chart(feat_df: pd.DataFrame):
    if feat_df.empty:
        return None

    lower = {c.lower(): c for c in feat_df.columns}
    feature_col = lower.get("feature")
    importance_col = lower.get("importance")
    if not feature_col or not importance_col:
        return None

    tmp = feat_df[[feature_col, importance_col]].rename(
        columns={feature_col: "Feature", importance_col: "Importance"}
    )
    tmp = tmp.sort_values("Importance", ascending=True)

    fig = go.Figure()
    fig.add_bar(
        x=tmp["Importance"],
        y=tmp["Feature"],
        orientation="h",
        marker_color="#2196F3",
    )
    fig.update_layout(
        title="Random Forest Feature Importance",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def per_ds_correct_chart(df: pd.DataFrame):
    fig = go.Figure()
    x = [ds.capitalize() for ds in DS_ORDER]

    if "RF_DS" in df.columns:
        rf = [((df["Observed_DS"] == ds) & (df["RF_DS"] == ds)).sum() for ds in DS_ORDER]
        fig.add_bar(name="RF Correct", x=x, y=rf, marker_color="#2196F3")

    if "GMM_DS" in df.columns:
        gm = [((df["Observed_DS"] == ds) & (df["GMM_DS"] == ds)).sum() for ds in DS_ORDER]
        fig.add_bar(name="GMM Correct", x=x, y=gm, marker_color="#FF9800")

    obs = [(df["Observed_DS"] == ds).sum() for ds in DS_ORDER]
    fig.add_scatter(
        name="Observed Total",
        x=x,
        y=obs,
        mode="lines+markers",
        line=dict(color="#2E7D32", width=3),
    )

    fig.update_layout(
        title="Per Damage-State Correct Predictions",
        barmode="group",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def missed_false_alarm_chart(df: pd.DataFrame):
    labels, missed, false_alarms = [], [], []

    if "RF_DS" in df.columns:
        _, _, m, f = metrics(df, "RF_DS")
        labels.append("RF")
        missed.append(m)
        false_alarms.append(f)

    if "GMM_DS" in df.columns:
        _, _, m, f = metrics(df, "GMM_DS")
        labels.append("GMM")
        missed.append(m)
        false_alarms.append(f)

    fig = go.Figure()
    fig.add_bar(name="Missed Damaged", x=labels, y=missed, marker_color="#F44336")
    fig.add_bar(name="False Alarms", x=labels, y=false_alarms, marker_color="#8E24AA")
    fig.update_layout(
        title="Missed Damaged vs False Alarms",
        barmode="group",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_map(df: pd.DataFrame, show_rf: bool, show_gmm: bool) -> go.Figure:
    fig = go.Figure()

    # observed damaged bridges as solid markers
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
                marker=dict(size=10, color=COLOR_MAP[ds]),
                customdata=list(zip(
                    part["Structure_ID"].astype(str),
                    part["Observed_DS"].astype(str),
                    part["Sa_1s_g"].astype(float),
                    part["HWB_Class"].astype(str) if "HWB_Class" in part.columns else ["NA"] * len(part),
                )),
                hovertemplate=(
                    "<b>ID:</b> %{customdata[0]}<br>"
                    "<b>Observed:</b> %{customdata[1]}<br>"
                    "<b>Sa:</b> %{customdata[2]:.3f}<br>"
                    "<b>HWB:</b> %{customdata[3]}<extra></extra>"
                ),
            )
        )

    def add_overpred(pred_col: str, label: str, color: str):
        tmp = df.copy()
        tmp["obs_idx"] = tmp["Observed_DS"].map(DS_INDEX).fillna(-1)
        tmp["pred_idx"] = tmp[pred_col].map(DS_INDEX).fillna(-1)
        over = tmp[tmp["pred_idx"] > tmp["obs_idx"]]
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
                    color=color,
                    symbol="circle-open",
                    opacity=1.0,
                ),
                customdata=list(zip(
                    over["Structure_ID"].astype(str),
                    over["Observed_DS"].astype(str),
                    over[pred_col].astype(str),
                    over["Sa_1s_g"].astype(float),
                )),
                hovertemplate=(
                    "<b>ID:</b> %{customdata[0]}<br>"
                    "<b>Observed:</b> %{customdata[1]}<br>"
                    "<b>Predicted:</b> %{customdata[2]}<br>"
                    "<b>Sa:</b> %{customdata[3]:.3f}<extra></extra>"
                ),
            )
        )

    if show_rf and "RF_DS" in df.columns:
        add_overpred("RF_DS", "RF overprediction", "#2196F3")

    if show_gmm and "GMM_DS" in df.columns:
        add_overpred("GMM_DS", "GMM overprediction", "#FF9800")

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=8,
        mapbox_center={"lat": float(df["Latitude"].mean()), "lon": float(df["Longitude"].mean())},
        margin=dict(l=0, r=0, t=40, b=0),
        title="Observed damaged bridges (solid) + overpredictions (open circles)",
        height=620,
        legend=dict(orientation="h"),
    )
    return fig


def detail_table(row: pd.Series) -> pd.DataFrame:
    data = {
        "Field": [
            "Structure ID",
            "Observed DS",
            "RF DS",
            "GMM DS",
            "Sa(1.0s)",
            "PGA",
            "HWB Class",
            "Year Built",
            "Design Era",
            "Num Spans",
        ],
        "Value": [
            row.get("Structure_ID", "NA"),
            row.get("Observed_DS", "NA"),
            row.get("RF_DS", "NA"),
            row.get("GMM_DS", "NA"),
            row.get("Sa_1s_g", "NA"),
            row.get("PGA_g", "NA"),
            row.get("HWB_Class", "NA"),
            row.get("Year_Built", "NA"),
            row.get("Design_Era", "NA"),
            row.get("Num_Spans", "NA"),
        ],
    }
    return pd.DataFrame(data)


def prob_table(row: pd.Series):
    rows = {"Damage State": DS_ORDER}

    rf_cols = ["RF_P_None", "RF_P_Slight", "RF_P_Moderate", "RF_P_Extensive", "RF_P_Complete"]
    gmm_cols = ["GMM_P_None", "GMM_P_Slight", "GMM_P_Moderate", "GMM_P_Extensive", "GMM_P_Complete"]

    if all(c in row.index for c in rf_cols):
        rows["RF Probability"] = [row[c] for c in rf_cols]

    if all(c in row.index for c in gmm_cols):
        rows["GMM Probability"] = [row[c] for c in gmm_cols]

    if len(rows) == 1:
        return None
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# LOAD + PREP
# ------------------------------------------------------------
try:
    bridges_raw, feat_df = load_raw_data()
    df = normalize_bridges(bridges_raw)
    df = add_repair_fields(df)
except Exception as e:
    st.error("Could not load data.")
    st.exception(e)
    st.stop()

# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.title("Filters")

selected_hwb = []
if "HWB_Class" in df.columns:
    selected_hwb = st.sidebar.multiselect(
        "HWB Class",
        sorted(df["HWB_Class"].dropna().unique().tolist()),
    )

selected_era = []
if "Design_Era" in df.columns:
    selected_era = st.sidebar.multiselect(
        "Design Era",
        sorted(df["Design_Era"].dropna().unique().tolist()),
    )

sa_min = float(df["Sa_1s_g"].min())
sa_max = float(df["Sa_1s_g"].max())
sa_range = st.sidebar.slider("Sa(1.0s) Range", sa_min, sa_max, (sa_min, sa_max))

show_rf = st.sidebar.checkbox("Show RF overprediction", True)
show_gmm = st.sidebar.checkbox("Show GMM overprediction", True)

filtered = filter_df(df, selected_hwb, selected_era, sa_range)

if "selected_bridge_id" not in st.session_state:
    st.session_state["selected_bridge_id"] = None

# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
st.title("Bridge Damage Dashboard")
st.caption("Observed vs Random Forest vs GMM")

tab1, tab2, tab3 = st.tabs(["Damage View", "Bridge Explorer", "Model Insights"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bridges", len(filtered))
    c2.metric("Observed Damaged", int((filtered["Observed_DS"] != "none").sum()))

    if "RF_DS" in filtered.columns:
        _, rf_dmg, _, _ = metrics(filtered, "RF_DS")
        c3.metric("RF Damaged Accuracy", f"{rf_dmg:.1f}%")
    else:
        c3.metric("RF Damaged Accuracy", "NA")

    if "GMM_DS" in filtered.columns:
        _, gm_dmg, _, _ = metrics(filtered, "GMM_DS")
        c4.metric("GMM Damaged Accuracy", f"{gm_dmg:.1f}%")
    else:
        c4.metric("GMM Damaged Accuracy", "NA")

    map_fig = build_map(filtered, show_rf, show_gmm)

    selected = plotly_events(
        map_fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        key="damage_map",
    )

    st.info("Click a marker on the map to view bridge details below.")

    if selected:
        curve_no = selected[0]["curveNumber"]
        point_no = selected[0]["pointIndex"]
        clicked_id = map_fig.data[curve_no].customdata[point_no][0]
        st.session_state["selected_bridge_id"] = str(clicked_id)

    if st.session_state["selected_bridge_id"] is not None:
        selected_id = st.session_state["selected_bridge_id"]
        match = filtered[filtered["Structure_ID"].astype(str) == str(selected_id)]
        if len(match) > 0:
            row = match.iloc[0]
            left, right = st.columns(2)
            with left:
                st.subheader(f"Bridge: {selected_id}")
                st.dataframe(detail_table(row), use_container_width=True, hide_index=True)
            with right:
                probs = prob_table(row)
                if probs is not None:
                    st.dataframe(probs.style.format(precision=3), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(damage_count_chart(filtered), use_container_width=True)
    with right:
        st.plotly_chart(repair_count_chart(filtered), use_container_width=True)

with tab2:
    ids = filtered["Structure_ID"].astype(str).tolist()
    if ids:
        default_index = 0
        if st.session_state["selected_bridge_id"] in ids:
            default_index = ids.index(st.session_state["selected_bridge_id"])

        selected_id = st.selectbox("Select Structure ID", ids, index=default_index)
        row = filtered[filtered["Structure_ID"].astype(str) == str(selected_id)].iloc[0]

        left, right = st.columns(2)
        with left:
            st.dataframe(detail_table(row), use_container_width=True, hide_index=True)
        with right:
            probs = prob_table(row)
            if probs is not None:
                st.dataframe(probs.style.format(precision=3), use_container_width=True, hide_index=True)

with tab3:
    feat_fig = feature_chart(feat_df)
    if feat_fig is not None:
        st.plotly_chart(feat_fig, use_container_width=True)

    rows = []
    if "RF_DS" in filtered.columns:
        rf_overall, rf_dmg, rf_missed, rf_false = metrics(filtered, "RF_DS")
        rows.append(["RF", rf_overall, rf_dmg, rf_missed, rf_false])

    if "GMM_DS" in filtered.columns:
        gm_overall, gm_dmg, gm_missed, gm_false = metrics(filtered, "GMM_DS")
        rows.append(["GMM", gm_overall, gm_dmg, gm_missed, gm_false])

    if rows:
        st.dataframe(
            pd.DataFrame(
                rows,
                columns=["Model", "Overall Accuracy (%)", "Damaged Accuracy (%)", "Missed Damaged", "False Alarms"],
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(per_ds_correct_chart(filtered), use_container_width=True)
    with right:
        st.plotly_chart(missed_false_alarm_chart(filtered), use_container_width=True)

st.markdown("---")
st.caption("Observed damaged bridges are solid. RF and GMM overpredictions are open circles.")
