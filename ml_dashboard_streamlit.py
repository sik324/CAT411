import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Bridge Damage ML Dashboard", layout="wide")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path("output/ml_dashboard")
COLOR_MAP = {
    'none': '#2E8B57',
    'slight': '#1E90FF',
    'moderate': '#FFA500',
    'extensive': '#FF4500',
    'complete': '#8B0000'
}
DS_ORDER = ['none', 'slight', 'moderate', 'extensive', 'complete']
REPAIR_ORDER = ['Safe', 'Needs Repair', 'Needs Closure', 'Needs Replacement']
MODEL_TO_COLS = {
    'Observed': {
        'ds': 'Observed_DS',
        'dr': 'Obs_DR',
        'repair_cat': 'Obs_Repair_Category',
        'repair_action': 'Obs_Repair_Action',
        'priority': 'Obs_Priority',
        'color': 'Obs_Color',
        'correct': None,
        'prob_cols': []
    },
    'Random Forest': {
        'ds': 'RF_DS',
        'dr': 'RF_DR',
        'repair_cat': 'RF_Repair_Category',
        'repair_action': 'RF_Repair_Action',
        'priority': 'RF_Priority',
        'color': 'RF_Color',
        'correct': 'RF_Correct',
        'prob_cols': ['RF_P_None', 'RF_P_Slight', 'RF_P_Moderate', 'RF_P_Extensive', 'RF_P_Complete']
    },
    'Deep Learning': {
        'ds': 'MLP_DS',
        'dr': 'MLP_DR',
        'repair_cat': 'MLP_Repair_Category',
        'repair_action': 'MLP_Repair_Action',
        'priority': 'MLP_Priority',
        'color': 'MLP_Color',
        'correct': 'MLP_Correct',
        'prob_cols': ['MLP_P_None', 'MLP_P_Slight', 'MLP_P_Moderate', 'MLP_P_Extensive', 'MLP_P_Complete']
    }
}


def find_csv(path_text: str, filename: str) -> Path:
    base = Path(path_text).expanduser()
    path = base / filename
    return path


@st.cache_data(show_spinner=False)
def load_data(base_dir: str):
    base = Path(base_dir).expanduser()
    bridges = pd.read_csv(base / 'ml_bridges.csv')
    summary = pd.read_csv(base / 'ml_summary.csv')
    feat = pd.read_csv(base / 'ml_feature_importance.csv')
    long_df = None
    repair = None
    long_path = base / 'ml_long.csv'
    repair_path = base / 'ml_repair_priority.csv'
    if long_path.exists():
        long_df = pd.read_csv(long_path)
    if repair_path.exists():
        repair = pd.read_csv(repair_path)

    # normalize types
    num_cols = ['Latitude', 'Longitude', 'Sa_1s_g', 'PGA_g', 'Obs_DR', 'RF_DR', 'MLP_DR']
    for c in num_cols:
        if c in bridges.columns:
            bridges[c] = pd.to_numeric(bridges[c], errors='coerce')

    # Create helper columns for modeling comparisons
    bridges['Observed_Damaged'] = bridges['Observed_DS'] != 'none'
    bridges['RF_Damaged'] = bridges['RF_DS'] != 'none'
    bridges['MLP_Damaged'] = bridges['MLP_DS'] != 'none'
    return bridges, summary, feat, long_df, repair


def metric_card(col, label, value, delta=None):
    with col:
        st.metric(label, value, delta)


def filtered_df(df, model_name, filters):
    cols = MODEL_TO_COLS[model_name]
    out = df.copy()

    eras = filters.get('eras')
    if eras:
        out = out[out['Design_Era'].isin(eras)]

    hwb = filters.get('hwb')
    if hwb:
        out = out[out['HWB_Class'].isin(hwb)]

    years = filters.get('years')
    if years:
        out = out[(out['Year_Built'] >= years[0]) & (out['Year_Built'] <= years[1])]

    sa_rng = filters.get('sa_rng')
    if sa_rng:
        out = out[(out['Sa_1s_g'] >= sa_rng[0]) & (out['Sa_1s_g'] <= sa_rng[1])]

    obs_states = filters.get('obs_states')
    if obs_states:
        out = out[out['Observed_DS'].isin(obs_states)]

    pred_states = filters.get('pred_states')
    if pred_states and cols['ds'] in out.columns:
        out = out[out[cols['ds']].isin(pred_states)]

    only_mis = filters.get('only_misclassified', False)
    if only_mis and cols['correct']:
        out = out[out[cols['correct']] == 0]

    return out


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.title("Bridge Damage ML Dashboard")
st.caption("Observed vs Random Forest vs Deep Learning (MLP) for Northridge bridge damage")

with st.sidebar:
    st.header("Data Source")
    base_dir = st.text_input(
        "Folder containing exported CSV files",
        value=str(DEFAULT_OUTPUT_DIR),
        help="This folder should contain ml_bridges.csv, ml_summary.csv, and ml_feature_importance.csv"
    )
    reload_btn = st.button("Load / Reload data", use_container_width=True)

if reload_btn:
    st.cache_data.clear()

try:
    bridges, summary, feat, long_df, repair = load_data(base_dir)
except Exception as e:
    st.error(
        "Could not load dashboard files. Make sure you exported the ML dashboard CSV files first.\n\n"
        f"Current folder: {base_dir}\nError: {e}"
    )
    st.stop()

# ------------------------------------------------------------
# Controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    model_name = st.selectbox("Model view", list(MODEL_TO_COLS.keys()), index=1)
    eras = st.multiselect("Design era", sorted(bridges['Design_Era'].dropna().unique().tolist()), default=[])
    hwb = st.multiselect("HWB class", sorted(bridges['HWB_Class'].dropna().unique().tolist()), default=[])

    yr_min = int(np.nanmin(bridges['Year_Built']))
    yr_max = int(np.nanmax(bridges['Year_Built']))
    years = st.slider("Year built", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))

    sa_min = float(np.nanmin(bridges['Sa_1s_g']))
    sa_max = float(np.nanmax(bridges['Sa_1s_g']))
    sa_rng = st.slider("Sa(1.0s) range", min_value=sa_min, max_value=sa_max, value=(sa_min, sa_max))

    obs_states = st.multiselect("Observed damage states", DS_ORDER, default=[])
    pred_states = st.multiselect(f"{model_name} damage states", DS_ORDER, default=[])
    only_misclassified = st.checkbox("Only misclassified bridges", value=False, disabled=(model_name == 'Observed'))

filters = {
    'eras': eras,
    'hwb': hwb,
    'years': years,
    'sa_rng': sa_rng,
    'obs_states': obs_states,
    'pred_states': pred_states,
    'only_misclassified': only_misclassified,
}

df_view = filtered_df(bridges, model_name, filters)
cols = MODEL_TO_COLS[model_name]

# ------------------------------------------------------------
# KPI row
# ------------------------------------------------------------
st.subheader(f"Overview — {model_name}")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
metric_card(kpi1, "Bridges in view", f"{len(df_view):,}")
metric_card(kpi2, "Damaged in view (observed)", f"{(df_view['Observed_DS'] != 'none').sum():,}")
metric_card(kpi3, f"{model_name} damaged", f"{(df_view[cols['ds']] != 'none').sum():,}")
metric_card(kpi4, f"Mean {model_name} DR", f"{df_view[cols['dr']].mean():.3f}")
if cols['correct']:
    acc = 100 * df_view[cols['correct']].mean() if len(df_view) else 0
    metric_card(kpi5, f"{model_name} accuracy in view", f"{acc:.1f}%")
else:
    metric_card(kpi5, "Observed mean DR", f"{df_view['Obs_DR'].mean():.3f}")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Summary", "Map", "Bridge Explorer", "Repair Priority", "Model Diagnostics"
])

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        ds_counts = pd.DataFrame({
            'Damage State': DS_ORDER,
            'Observed': [(df_view['Observed_DS'] == ds).sum() for ds in DS_ORDER],
            model_name: [(df_view[cols['ds']] == ds).sum() for ds in DS_ORDER],
        })
        fig = go.Figure()
        fig.add_bar(name='Observed', x=ds_counts['Damage State'], y=ds_counts['Observed'])
        fig.add_bar(name=model_name, x=ds_counts['Damage State'], y=ds_counts[model_name])
        fig.update_layout(barmode='group', title='Damage State Distribution', height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rep_counts = pd.DataFrame({
            'Repair Category': REPAIR_ORDER,
            'Observed': [(df_view['Obs_Repair_Category'] == c).sum() for c in REPAIR_ORDER],
            model_name: [(df_view[cols['repair_cat']] == c).sum() for c in REPAIR_ORDER],
        })
        fig2 = go.Figure()
        fig2.add_bar(name='Observed', x=rep_counts['Repair Category'], y=rep_counts['Observed'])
        fig2.add_bar(name=model_name, x=rep_counts['Repair Category'], y=rep_counts[model_name])
        fig2.update_layout(barmode='group', title='Repair Category Distribution', height=420)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.scatter(
            df_view,
            x='Sa_1s_g', y=cols['dr'],
            color=cols['ds'],
            color_discrete_map=COLOR_MAP,
            hover_data=['Structure_ID', 'HWB_Class', 'Year_Built', 'Observed_DS', cols['ds']],
            title=f'{model_name} Damage Ratio vs Sa(1.0s)'
        )
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        if cols['correct']:
            perf = pd.DataFrame({
                'Metric': ['Accuracy', 'Error Rate'],
                'Value': [100 * df_view[cols['correct']].mean(), 100 * (1 - df_view[cols['correct']].mean())]
            })
            fig4 = px.bar(perf, x='Metric', y='Value', title=f'{model_name} Performance in Current View')
            fig4.update_layout(height=420, yaxis_title='Percent')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info('Observed view does not have model accuracy metrics.')

with tab2:
    map_col1, map_col2 = st.columns([3, 1])
    with map_col1:
        plot_df = df_view.rename(columns={cols['ds']: 'Display_DS', cols['dr']: 'Display_DR'})
        figm = px.scatter_map(
            plot_df,
            lat='Latitude', lon='Longitude',
            color='Display_DS',
            color_discrete_map=COLOR_MAP,
            hover_name='Structure_ID',
            hover_data={
                'HWB_Class': True,
                'Year_Built': True,
                'Sa_1s_g': ':.3f',
                'PGA_g': ':.3f',
                'Observed_DS': True,
                'Display_DS': True,
                'Display_DR': ':.3f',
            },
            zoom=7,
            title=f'{model_name} Spatial Damage View'
        )
        figm.update_layout(height=650)
        st.plotly_chart(figm, use_container_width=True)
    with map_col2:
        st.markdown('**Map notes**')
        st.write(f"Model shown: **{model_name}**")
        st.write(f"Bridges shown: **{len(df_view):,}**")
        st.write(f"Mean DR: **{df_view[cols['dr']].mean():.3f}**")
        if cols['correct']:
            st.write(f"Accuracy in view: **{100 * df_view[cols['correct']].mean():.1f}%**")
        st.write("Colors represent predicted or observed damage state.")

with tab3:
    st.markdown('**Bridge-level table**')
    display_cols = [
        'Structure_ID', 'HWB_Class', 'Year_Built', 'Design_Era',
        'Sa_1s_g', 'PGA_g', 'Observed_DS', cols['ds'], 'Obs_DR', cols['dr'],
        'Obs_Repair_Category', cols['repair_cat'], cols['repair_action'], cols['priority']
    ]
    if cols['correct']:
        display_cols.append(cols['correct'])
    available = [c for c in display_cols if c in df_view.columns]
    st.dataframe(df_view[available].sort_values('Sa_1s_g', ascending=False), use_container_width=True, height=500)

    if cols['prob_cols']:
        st.markdown('**Predicted probability breakdown**')
        sid = st.selectbox('Choose a bridge', df_view['Structure_ID'].astype(str).tolist())
        row = df_view[df_view['Structure_ID'].astype(str) == sid].iloc[0]
        prob_df = pd.DataFrame({
            'Damage State': DS_ORDER,
            'Probability': [row[p] for p in cols['prob_cols']]
        })
        figp = px.bar(prob_df, x='Damage State', y='Probability', color='Damage State', color_discrete_map=COLOR_MAP,
                      title=f'{model_name} probabilities for bridge {sid}')
        figp.update_layout(height=400)
        st.plotly_chart(figp, use_container_width=True)

with tab4:
    st.markdown('**Repair priority view**')
    priority_df = df_view[[
        'Structure_ID', 'HWB_Class', 'Year_Built', 'Design_Era', 'Sa_1s_g',
        'Observed_DS', cols['ds'], cols['repair_cat'], cols['repair_action'], cols['priority']
    ]].copy()
    priority_df = priority_df.sort_values(cols['priority'], ascending=False)
    st.dataframe(priority_df, use_container_width=True, height=500)

    figpr = px.histogram(priority_df, x=cols['priority'], color=cols['repair_cat'], barmode='group',
                         title=f'{model_name} Repair Priority Distribution')
    figpr.update_layout(height=400)
    st.plotly_chart(figpr, use_container_width=True)

with tab5:
    d1, d2 = st.columns(2)
    with d1:
        figf = px.bar(feat.sort_values('Importance', ascending=True), x='Importance', y='Feature', orientation='h',
                      title='Random Forest Feature Importance')
        figf.update_layout(height=420)
        st.plotly_chart(figf, use_container_width=True)

    with d2:
        if cols['correct']:
            cmp = pd.DataFrame({
                'Type': ['Correct', 'Incorrect'],
                'Count': [int(df_view[cols['correct']].sum()), int((1 - df_view[cols['correct']]).sum())]
            })
            figc = px.pie(cmp, names='Type', values='Count', title=f'{model_name} Correct vs Incorrect')
            figc.update_layout(height=420)
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.info('Observed view does not include model diagnostics.')

    st.markdown('**Model KPI summary from exported file**')
    st.dataframe(summary, use_container_width=True)

st.markdown('---')
st.caption(
    'Expected usage: first run your ML notebook/export script to generate ml_bridges.csv, ml_summary.csv, '
    'ml_feature_importance.csv, ml_long.csv, and ml_repair_priority.csv; then point this Streamlit app to that folder.'
)
