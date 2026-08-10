import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.colors import rgb2hex
import joblib
import shap

# Page config
st.set_page_config(page_title="EV Adoption Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Theme colors — matched to the original mockup (dark bg, black cards, green/amber/red accents)
BG = "#181818"
CARD_BG = "#0d0d0d"
PANEL_BG = "#2b2b2b"
BORDER = "#3d3d3d"
TEXT = "#f2f2f2"
TEXT_MUTED = "#9a9a9a"
GREEN = "#5fb98a"
AMBER = "#e0a234"
RED = "#d9635b"
BLUE = "#4c8bf0"
GREY = "#454a52"

TIER_COLORS = {"High": BLUE, "Medium": AMBER, "Low": RED, "No data": GREY}

# Global CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}

    :root {{
        --bg-success: rgba(95, 185, 138, 0.14);
        --text-success: {GREEN};
        --text-muted: {TEXT_MUTED};
    }}

    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    #MainMenu, header, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.5rem; max-width: 1300px; }}

    .metric-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 18px 22px;
        height: 100%;
    }}
    .metric-label {{ color: {TEXT_MUTED}; font-size: 14px; margin-bottom: 6px; }}
    .metric-value {{ color: {TEXT}; font-size: 26px; font-weight: 600; }}

    .panel {{
        background-color: {PANEL_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 20px;
    }}
    .panel-title {{ font-size: 17px; font-weight: 600; margin-bottom: 14px; color: {TEXT}; }}
    .feature-row {{
        display: flex; justify-content: space-between;
        padding: 10px 0; border-bottom: 1px solid {BORDER}; font-size: 15px;
    }}
    .feature-row:last-child {{ border-bottom: none; }}
    .feature-name {{ color: {TEXT}; }}
    .feature-value {{ color: {TEXT_MUTED}; }}

    /* Real Streamlit bordered containers (st.container(border=True)) — used for
       any section that mixes a heading with native widgets (selectboxes, columns,
       charts). Plain HTML divs opened in one st.markdown call and closed in a
       later one do NOT reliably wrap native Streamlit elements in between, so
       every "boxed section" below uses this instead of the .panel div hack. */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {PANEL_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 16px !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{ background-color: transparent !important; }}

    /* Selectboxes — tidy up default light dropdown to match the dark theme */
    div[data-baseweb="select"] > div {{
        background-color: {CARD_BG} !important;
        border-color: {BORDER} !important;
        border-radius: 10px !important;
    }}
</style>
""", unsafe_allow_html=True)

# Data loading
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load('models/logistic_regression.joblib'),
        "Random Forest": joblib.load('models/random_forest.joblib'),
        "Bagging": joblib.load('models/bagging.joblib'),
        "XGBoost": joblib.load('models/xgboost.joblib'),
    }

@st.cache_resource
def load_shap_values():
    return {
        "Logistic Regression": joblib.load('models/shap_values_lr.joblib'),
        "Random Forest": joblib.load('models/shap_values_rf.joblib'),
        "XGBoost": joblib.load('models/shap_values_xg.joblib'),
        "Bagging": joblib.load('models/shap_values_bg.joblib'),
    }

@st.cache_data
def load_counties_geojson():
    # Loaded once and reused both as the choropleth's geometry source and as the
    # master list of every U.S. county FIPS code, so we can tell "no data for
    # this county" apart from "this shape failed to render".
    import json
    import urllib.request
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)

models = load_models()
shap_values = load_shap_values()
counties_geojson = load_counties_geojson()
ALL_COUNTY_FIPS = [feat["id"] for feat in counties_geojson["features"]]

modelOverviewDF = pd.read_csv('deploymentDfnew.csv')
# Ensure FIPS codes keep their leading zero (e.g. Alabama = 01001, not 1001) —
# without this, every county whose FIPS starts with 0 fails to match the
# geojson's zero-padded string IDs and renders as a blank gap on the map.
modelOverviewDF["StCnty FIPS Code"] = modelOverviewDF["StCnty FIPS Code"].astype(str).str.zfill(5)
featureOverviewDF = pd.read_csv('featureimportanceDF_new.csv')
sentimentOverviewDF = pd.read_csv('deploymentDF3new.csv')
modelmetricsDF = pd.read_csv('model_metrics_new.csv')

top_states_by_mention = pd.read_csv('deploymentDF3new.csv')

MODEL_METRICS = (modelmetricsDF.set_index('Metrics').transpose().iloc[1:].to_dict(orient='index'))
print(modelmetricsDF)

# Readable labels for the raw SHAP/model column names
FEATURE_LABELS = {
    "Total_population": "Total population",
    "pct_college_educated": "Educated population",
    "Median_household_income": "Median household income",
    "population_density": "Population density",
    "chargers_per_100k": "Charging stations per 100k",
    "avg_sentiment": "Reddit sentiment",
}


def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def panel_heading(title, subtitle=None):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<span style="color:{TEXT_MUTED};">{subtitle}</span>', unsafe_allow_html=True)
        st.write("")


def hbar(name, value, max_value, color):
    pct = max(0, min(100, (value / max_value) * 100)) if max_value else 0
    st.markdown(f"""
    <div style="margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:4px;">
            <span style="font-weight:600;">{name}</span>
            <span style="color:{TEXT_MUTED};">{value}</span>
        </div>
        <div style="background:#1a1a1a; border-radius:6px; height:14px; width:100%;">
            <div style="background:{color}; width:{pct}%; height:100%; border-radius:6px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Navigation Bar
st.markdown('<div style="font-size:26px; font-weight:600; margin-bottom:10px;">EV Adoption Dashboard</div>',
            unsafe_allow_html=True)

page = option_menu(
    menu_title=None,
    options=["Overview", "Models", "Features", "Sentiment"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": CARD_BG,
                       "border": f"1px solid {BORDER}", "border-radius": "14px"},
        "icon": {"color": TEXT_MUTED, "font-size": "14px"},
        "nav-link": {"font-size": "15px", "color": TEXT_MUTED, "text-align": "center",
                     "margin": "4px", "border-radius": "10px", "padding": "10px 16px"},
        "nav-link-selected": {"background-color": "#323232", "color": TEXT, "font-weight": "600"},
    },
)

st.write("")  # spacing

# OVERVIEW
if page == "Overview":
    state_sel = st.session_state.get("ov_state", "-")
    county_sel = st.session_state.get("ov_county", "-")
    row = None
    if state_sel != "-" and county_sel != "-":
        match = modelOverviewDF[(modelOverviewDF["State"] == state_sel) & (modelOverviewDF["County"] == county_sel)]
        if not match.empty:
            row = match.iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">State</div>', unsafe_allow_html=True)
        state_choice = st.selectbox("", ["-"] + sorted(modelOverviewDF["State"].dropna().   unique()), key="ov_state",
                                     label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        county_opts = ["-"] + sorted(modelOverviewDF[modelOverviewDF["State"] == state_choice]["County"].unique()) \
            if state_choice != "-" else ["-"]
        st.markdown('<div class="metric-card"><div class="metric-label">County</div>', unsafe_allow_html=True)
        county_choice = st.selectbox("", county_opts, key="ov_county", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        val = row["predicted_tier"] if row is not None else "-"
        metric_card("Predicted Adoption Tier", val)
    with c4:
        # Actual tier only shown for held-out test counties, so this never presents
        # an in-sample match as if it were genuine model performance
        if row is not None and row.get("split") == "test":
            st.markdown(f'''
            <div class="metric-card" style="background:var(--bg-success);">
                <div class="metric-label" style="color:var(--text-success);">Actual Adoption Tier</div>
                <div class="metric-value" style="color:var(--text-success);">{row["actual_tier"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        elif row is not None:
            st.markdown('''
            <div class="metric-card">
                <div class="metric-label">Actual Adoption Tier</div>
                <div class="metric-value" style="font-size:14px; color:var(--text-muted);">
                    Not available &mdash; training county
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            metric_card("Actual Adoption Tier", "-")
    with c5:
        val = f"{row['infrastructure_gap_score']:.2f}" if row is not None else "-"
        metric_card("Infrastructure Gap Score", val)

    st.write("")
    col_map, col_feat = st.columns([1.4, 1])

    with col_map:
        with st.container(border=True):
            panel_heading("Predicted Adoption by County",
                           "Grey counties have no matching record in the training data")

            # The geojson ships ~3,221 counties; modelOverviewDF only covers the
            # ~3,133 counties in the training set. Left-joining onto the full FIPS
            # list means every shape gets a color — the remainder ("No data") —
            # instead of being left blank, which used to look like a rendering bug.
            map_df = pd.DataFrame({"StCnty FIPS Code": ALL_COUNTY_FIPS}).merge(
                modelOverviewDF[["StCnty FIPS Code", "predicted_tier", "County", "State"]],
                on="StCnty FIPS Code", how="left",
            )
            map_df["predicted_tier"] = map_df["predicted_tier"].fillna("No data")
            map_df["County"] = map_df["County"].fillna("—")
            map_df["State"] = map_df["State"].fillna("—")

            fig = px.choropleth(
                map_df, geojson=counties_geojson,
                locations="StCnty FIPS Code", color="predicted_tier", color_discrete_map=TIER_COLORS,
                scope="usa", category_orders={"predicted_tier": ["Low", "High", "No data"]},
                hover_data={"County": True, "State": True, "predicted_tier": True, "StCnty FIPS Code": False},
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), height=460,
                paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
                font=dict(family="Inter", color=TEXT),
                legend=dict(font=dict(color=TEXT)), geo=dict(bgcolor=PANEL_BG),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_feat:
        with st.container(border=True):
            panel_heading("County Features")
            feats = [
                ("Median Household Income", f"${row['Median_household_income']:,.0f}" if row is not None else "-"),
                ("Charging Stations Per 100K", f"{row['chargers_per_100k']:,}" if row is not None else "-"),
                ("Population Density", f"{row['population_density']:,} /mi²" if row is not None else "-"),
                ("Educated Population", f"{row['pct_college_educated']}%" if row is not None else "-"),
                ("Reddit Sentiment", f"{row['avg_sentiment']}" if row is not None else "-"),
                ("Total Population", f"{row['Total_population']:,}" if row is not None else "-")
            ]
            for name, val in feats:
                st.markdown(f"""
                <div class="feature-row"><span class="feature-name">{name}</span>
                <span class="feature-value">{val}</span></div>
                """, unsafe_allow_html=True)


# MODELS
elif page == "Models":
    with st.container(border=True):
        panel_heading("Metrics By Model")
        header = st.columns([2, 1, 1, 1, 1])
        header[0].markdown(f'<span style="color:{TEXT_MUTED}">Model</span>', unsafe_allow_html=True)
        header[1].markdown(f'<span style="color:{TEXT_MUTED}">Precision</span>', unsafe_allow_html=True)
        header[2].markdown(f'<span style="color:{TEXT_MUTED}">Recall</span>', unsafe_allow_html=True)
        header[3].markdown(f'<span style="color:{TEXT_MUTED}">F1</span>', unsafe_allow_html=True)
        header[4].markdown(f'<span style="color:{TEXT_MUTED}">Support</span>', unsafe_allow_html=True)

        for model, m in MODEL_METRICS.items():
            row_cols = st.columns([2, 1, 1, 1, 1])
            row_cols[0].write(model)
            row_cols[1].write(f"{m['precision']:.3f}")
            row_cols[2].write(f"{m['recall']:.3f}")
            row_cols[3].write(f"{m['f1-score']:.3f}")
            row_cols[4].write(f"{m['support']:.0f}")


# FEATURES
elif page == "Features":
    model_choice = option_menu(
        menu_title = None,
        options = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Bagging'],
        orientation = "horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "nav-link": {"font-size": "15px", "color": TEXT, "text-align": "center",
                         "margin": "4px", "border-radius": "12px", "padding": "14px 18px",
                         "background-color": CARD_BG, "border": f"1px solid {BORDER}"},
            "nav-link-selected": {"background-color": "rgba(76, 139, 240, 0.16)", "color": BLUE,
                                   "font-weight": "600", "border": f"1px solid rgba(76, 139, 240, 0.4)"},
        },
    )

    explanation = shap_values[model_choice]
    vals = explanation.values[..., 1] if explanation.values.ndim == 3 else explanation.values
    feat_names = list(explanation.feature_names)

    # --- Panel 1: global feature importance ---------------------------------
    with st.container(border=True):
        panel_heading("What drives EV adoption predictions",
                       "Average impact of each factor on the model's prediction, across all counties")

        mean_abs = np.abs(vals).mean(axis=0)
        max_val = float(mean_abs.max()) if len(mean_abs) else 1.0
        for i in np.argsort(-mean_abs):
            hbar(FEATURE_LABELS.get(feat_names[i], feat_names[i].replace('_', ' ')),
                 round(float(mean_abs[i]), 3), max_val, BLUE)

    st.write("")

    # --- State / county selection --------------------------------------------
    # Same session_state keys as the Overview page pickers, so choosing a
    # county here also updates Overview (and vice versa) — this is what lets
    # the waterfall below know which county to explain.
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">State</div>', unsafe_allow_html=True)
        state_choice = st.selectbox("", ["-"] + sorted(modelOverviewDF["State"].dropna().unique()), key="ov_state",
                                     label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        county_opts = ["-"] + sorted(modelOverviewDF[modelOverviewDF["State"] == state_choice]["County"].unique()) \
            if state_choice != "-" else ["-"]
        st.markdown('<div class="metric-card"><div class="metric-label">County</div>', unsafe_allow_html=True)
        county_choice = st.selectbox("", county_opts, key="ov_county", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    row = None
    if state_choice != "-" and county_choice != "-":
        match = modelOverviewDF[(modelOverviewDF["State"] == state_choice) & (modelOverviewDF["County"] == county_choice)]
        if not match.empty:
            row = match.iloc[0]

    # --- Panel 2: local explanation (waterfall) ------------------------------
    if row is not None:
        panel_title = f'Why {county_choice}, {state_choice} was predicted "{row["predicted_tier"]}"'
    else:
        panel_title = "Why a county was predicted its tier"

    with st.container(border=True):
        if row is not None:
            panel_heading(panel_title,
                          "Starting from the model's baseline expectation, each factor below pushes "
                          "the prediction up or down for this county")

            county_index_shap = modelOverviewDF[(modelOverviewDF["State"] == state_choice) &
                                                 (modelOverviewDF["County"] == county_choice)].index[0]
            county_exp = explanation[..., 1][county_index_shap] if explanation.values.ndim == 3 else explanation[county_index_shap]

            c_values = np.array(county_exp.values).flatten()
            base_value = float(np.array(county_exp.base_values).flatten()[0])
            c_order = np.argsort(-np.abs(c_values))
            c_values = c_values[c_order]
            c_names = [feat_names[i] for i in c_order]

            labels = ["Baseline (avg. county)"] + [
                f"{'+' if v >= 0 else chr(8722)} {FEATURE_LABELS.get(n, n.replace('_', ' '))}"
                for n, v in zip(c_names, c_values)
            ] + ["Final prediction"]
            measures = ["absolute"] + ["relative"] * len(c_values) + ["total"]
            x_vals = [base_value] + list(c_values) + [None]
            text_vals = ([f"{base_value:.2f}"] + [f"{v:+.2f}" for v in c_values]
                          + [f"{base_value + c_values.sum():.2f}"])

            wf = go.Figure(go.Waterfall(
                orientation="h",
                measure=measures,
                y=labels,
                x=x_vals,
                text=text_vals,
                textposition="outside",
                connector={"visible": False},
                decreasing={"marker": {"color": RED}},
                increasing={"marker": {"color": BLUE}},
                totals={"marker": {"color": "#8a8f98"}},
            ))
            wf.update_layout(
                height=70 + 42 * len(labels),
                paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
                font=dict(color=TEXT, family="Inter"),
                margin=dict(l=10, r=70, t=10, b=10),
                showlegend=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(wf, use_container_width=True)
        else:
            panel_heading(panel_title)
            st.markdown("Select a state and county to see what drove that prediction", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SENTIMENT
# ---------------------------------------------------------------------------
elif page == "Sentiment":
    '''c1, c2, c3 = st.columns(3)
    with c1: metric_card("No. of Reddit Posts", "940,106")
    with c2: metric_card("Posts Mentioning States", "11,603")
    with c3: metric_card("NER Model", "RoBERTa-large")
'''
    st.write("")
    with st.container(border=True):
        top_states_by_mention = top_states_by_mention.sort_values(by='post_count', ascending=False)
        max_post_count = top_states_by_mention['post_count'].max()
        top_states_by_mention['pct'] = top_states_by_mention['post_count'] / max_post_count * 100

        def label_sentiment(sentiment_value):
            if sentiment_value >= 0.05:
                return 'positive'
            elif sentiment_value <= -0.05:
                return 'negative'
            return 'neutral'
        top_states_by_mention['sentiment'] = top_states_by_mention['avg_sentiment'].apply(label_sentiment)

        cmapR = matplotlib.colormaps['RdYlGn']
        norm = Normalize(vmin = -1, vmax = 1)
        top_states_by_mention['colour'] = top_states_by_mention['avg_sentiment'].apply(lambda r: rgb2hex(cmapR(norm(r))))

        # KPI row: share of state-tagged mentions (the 11,603 posts that named a
        # state) falling into each sentiment bucket, weighted by post_count.
        # >>> If you have sentiment labels on the full 940,106-post dataset,
        # swap this for the true share of all posts instead of this proxy. <<<
        by_sentiment = top_states_by_mention.groupby('sentiment')['post_count'].sum()
        total_mentions = by_sentiment.sum()
        pos_pct = by_sentiment.get('positive', 0) / total_mentions * 100
        neu_pct = by_sentiment.get('neutral', 0) / total_mentions * 100
        neg_pct = by_sentiment.get('negative', 0) / total_mentions * 100

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px; padding:14px;">
                <div style="font-size:12px; color:{TEXT_MUTED}; margin-bottom:6px;">Positive</div>
                <div style="font-size:22px; font-weight:700; color:{GREEN};">{pos_pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px; padding:14px;">
                <div style="font-size:12px; color:{TEXT_MUTED}; margin-bottom:6px;">Neutral</div>
                <div style="font-size:22px; font-weight:700; color:{AMBER};">{neu_pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px; padding:14px;">
                <div style="font-size:12px; color:{TEXT_MUTED}; margin-bottom:6px;">Negative</div>
                <div style="font-size:22px; font-weight:700; color:{RED};">{neg_pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        panel_heading("Top 10 States By Mention Count")

        for index, row in top_states_by_mention.head(10).iterrows():
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                    <span style="font-weight:600; color:{TEXT};">{row['states_mentioned']}</span>
                    <span style="color:{TEXT_MUTED};">{row['post_count']} &middot; {row['sentiment']}</span>
                </div>
                <div style="background:#1a1a1a; border-radius:6px; height:10px; width:100%;">
                    <div style="background:{row['colour']}; width:{row['pct']}%; height:100%; border-radius:6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
