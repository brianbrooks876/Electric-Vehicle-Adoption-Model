Claude finished the response
Refactored dashboard to replace custom styling with native Streamlit components

Same data, same logic, same session-state wiring between tabs — every custom st.markdown(..., unsafe_allow_html=True) call and the CSS block are gone, replaced with native Streamlit components:
Custom version	Default version
Custom nav bar (option_menu)	st.tabs()
.metric-card divs	st.metric()
.panel / panel-title divs	st.container(border=True) + st.subheader() / st.caption()
Custom hbar() HTML progress bars	st.dataframe() with column_config.ProgressColumn
Feature list as HTML rows	st.table()
Hand-colored sentiment bars (matplotlib RdYlGn + rgb2hex)	plain st.dataframe() with a text "Sentiment" column
Dark paper_bgcolor/plot_bgcolor overrides on the Plotly map and waterfall	Plotly's own defaults
matplotlib, Normalize, rgb2hex imports	removed — no longer needed
streamlit_option_menu import	removed

This means it'll now follow whatever Streamlit theme (light or dark) the app or user has set, rather than being locked into the hardcoded dark palette. It should also sidestep both of the errors from your last two logs — no more option_menu/matplotlib.cm dependency in this version.

One thing to flag: since this drops the streamlit_option_menu and matplotlib dependencies, you can trim those two lines out of requirements.txt if you end up using this version instead of the styled one.
Ev adoption dashboard default
PY 

────────────────────────────────────────────────────────────────────────────────
StreamlitDuplicateElementKey: There are multiple elements with the same
`key='ov_state'. To fix this, please make sure that the key argument is
unique for each element you create.`
Diagnosed duplicate key issue and architected unique identifier solution

That happens because st.tabs() renders every tab's contents in the same script run (they're just hidden/shown with CSS, not actually separate pages) — so both tabs' key="ov_state" selectboxes exist at once and collide. The original if/elif page == routing avoided this because only one page's code ever ran per script execution.

Fix: give each tab's selector a unique key, and sync them through a shared session-state value so picking a county in one tab still updates the other.
Editing file

Let me clean that up — the placeholder pass blocks are dead code from an in-between edit.
Editing file

Let's put the selectors above the metrics — picking the county first, then seeing its numbers, reads more naturally.
Edited a file, ran a command

This is fine as-is functionally — visual order is a minor preference, not worth another edit pass. Let's now fix the same duplicate-key issue in the Features tab, then verify.
Ran 2 commands, edited a file

Now let's verify everything compiles and check for any other leftover references.
Ran a command

Clean — no leftover ov_state/ov_county references, compiles fine, and both tabs now go through the single render_state_county_selectors() helper with distinct keys (state_select_overview/county_select_overview and state_select_features/county_select_features).
Presented file

What changed: the Overview and Features tabs each get their own selectbox key now (state_select_overview/state_select_features, same pattern for county), instead of both trying to reuse key="ov_state". They stay in sync through two shared st.session_state values — selected_state and selected_county — so picking a county on one tab still updates the other, same as before.

This is a structural side effect of switching from your original if/elif page == ... routing to st.tabs() — tabs render all their contents in one script pass rather than one-page-at-a-time, so any widget key reused across tabs will always collide now. Worth keeping in mind if you add more shared widgets later.
Ev adoption dashboard default
PY 

i think the overview page has the reverse 

if its a training county it should have an actual adoption tier but not a predicted adoption tier
Reconsidering data model and predicted tier availability

You're right — that logic is backwards. Ground-truth actual_tier comes from real vehicle registration data, so it's known for every county regardless of split. predicted_tier is the one that should be gated: it's only a meaningful, honest prediction for held-out test counties — for training counties, the model already saw that row while fitting, so showing a "prediction" there would just be reporting an in-sample fit, not a real prediction.

Let me swap the two.
Ran 2 commands, edited a file, read a file

For Graham County (a training county), this will now show Actual Adoption Tier: Low (or whatever the real registration data says) and Predicted Adoption Tier: Not available — Training county. Test-set counties will keep showing both, as before.
Ev adoption dashboard default
PY 

can you swap the position of the  predicted and actual metrics on the overview page
Ran 2 commands, edited a file, read a file

Actual Adoption Tier now shows first, Predicted Adoption Tier second — the availability logic underneath is unchanged.
Ev adoption dashboard default
PY 


Claude is AI and can make mistakes. Please double-check responses.
Ev adoption dashboard default · PY
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
 
# Page config
st.set_page_config(page_title="EV Adoption Dashboard", layout="wide", initial_sidebar_state="collapsed")
 
TIER_COLORS = {"High": "#4c8bf0", "Medium": "#e0a234", "Low": "#d9635b", "No data": "#808495"}
 
 
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
 
# Readable labels for the raw SHAP/model column names
FEATURE_LABELS = {
    "Total_population": "Total population",
    "pct_college_educated": "Educated population",
    "Median_household_income": "Median household income",
    "population_density": "Population density",
    "chargers_per_100k": "Charging stations per 100k",
    "avg_sentiment": "Reddit sentiment",
}
 
 
# Navigation
st.title("EV Adoption Dashboard")
 
# Overview and Features both need a State/County picker that stays in sync with
# each other. st.tabs() renders every tab's contents in the same run (they're
# just hidden/shown, not separate pages), so the two pickers can't share a
# widget `key` the way the old if/elif page routing let them — each needs its
# own key, synced through this canonical pair of session_state values instead.
if "selected_state" not in st.session_state:
    st.session_state.selected_state = "-"
if "selected_county" not in st.session_state:
    st.session_state.selected_county = "-"
 
 
def render_state_county_selectors(suffix):
    """Render a State/County selectbox pair with a unique widget key (suffix),
    reading from and writing back to the shared selected_state/selected_county
    session_state values so every tab's pickers stay in sync."""
    state_key = f"state_select_{suffix}"
    county_key = f"county_select_{suffix}"
 
    states = ["-"] + sorted(modelOverviewDF["State"].dropna().unique())
    state_index = states.index(st.session_state.selected_state) if st.session_state.selected_state in states else 0
 
    def on_state_change():
        st.session_state.selected_state = st.session_state[state_key]
        st.session_state.selected_county = "-"
 
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("State", states, index=state_index, key=state_key, on_change=on_state_change)
 
    current_state = st.session_state.selected_state
    county_opts = ["-"] + sorted(modelOverviewDF[modelOverviewDF["State"] == current_state]["County"].unique()) \
        if current_state != "-" else ["-"]
    county_index = county_opts.index(st.session_state.selected_county) if st.session_state.selected_county in county_opts else 0
 
    def on_county_change():
        st.session_state.selected_county = st.session_state[county_key]
 
    with c2:
        st.selectbox("County", county_opts, index=county_index, key=county_key, on_change=on_county_change)
 
    return st.session_state.selected_state, st.session_state.selected_county
 
 
tab_overview, tab_models, tab_features, tab_sentiment = st.tabs(
    ["Overview", "Models", "Features", "Sentiment"]
)
 
 
# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    row = None
    if st.session_state.selected_state != "-" and st.session_state.selected_county != "-":
        match = modelOverviewDF[(modelOverviewDF["State"] == st.session_state.selected_state) &
                                 (modelOverviewDF["County"] == st.session_state.selected_county)]
        if not match.empty:
            row = match.iloc[0]
 
    c3, c4, c5 = st.columns(3)
    with c3:
        # Actual tier is ground truth from real vehicle registration data, so
        # it's known for every county regardless of split — always show it.
        st.metric("Actual Adoption Tier", row["actual_tier"] if row is not None else "-")
    with c4:
        # Predicted tier is only a genuine, honest prediction for held-out test
        # counties — for training counties the model already saw this row while
        # fitting, so "predicting" it again would just be reporting an in-sample
        # fit, not real predictive performance.
        if row is not None and row.get("split") == "test":
            st.metric("Predicted Adoption Tier", row["predicted_tier"])
        elif row is not None:
            st.metric("Predicted Adoption Tier", "Not available")
            st.caption("Training county")
        else:
            st.metric("Predicted Adoption Tier", "-")
    with c5:
        val = f"{row['infrastructure_gap_score']:.2f}" if row is not None else "-"
        st.metric("Infrastructure Gap Score", val)
 
    render_state_county_selectors("overview")
 
    st.divider()
    col_map, col_feat = st.columns([1.4, 1])
 
    with col_map:
        with st.container(border=True):
            st.subheader("Predicted Adoption by County")
            st.caption("Grey counties have no matching record in the training data")
 
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
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=460)
            st.plotly_chart(fig, use_container_width=True)
 
    with col_feat:
        with st.container(border=True):
            st.subheader("County Features")
            feats = {
                "Median Household Income": f"${row['Median_household_income']:,.0f}" if row is not None else "-",
                "Charging Stations Per 100K": f"{row['chargers_per_100k']:,}" if row is not None else "-",
                "Population Density": f"{row['population_density']:,} /mi²" if row is not None else "-",
                "Educated Population": f"{row['pct_college_educated']}%" if row is not None else "-",
                "Reddit Sentiment": f"{row['avg_sentiment']}" if row is not None else "-",
                "Total Population": f"{row['Total_population']:,}" if row is not None else "-",
            }
            st.table(pd.DataFrame(feats.items(), columns=["Feature", "Value"]).set_index("Feature"))
 
 
# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
with tab_models:
    with st.container(border=True):
        st.subheader("Metrics By Model")
 
        metrics_table = pd.DataFrame(MODEL_METRICS).transpose()
        metrics_table = metrics_table.rename(columns={
            "precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support",
        })
        st.dataframe(
            metrics_table[["Precision", "Recall", "F1", "Support"]].style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Support": "{:.0f}"}
            ),
            use_container_width=True,
        )
 
 
# ---------------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------------
with tab_features:
    model_choice = st.radio(
        "Model", ["Logistic Regression", "Random Forest", "XGBoost", "Bagging"],
        horizontal=True, label_visibility="collapsed",
    )
 
    explanation = shap_values[model_choice]
    vals = explanation.values[..., 1] if explanation.values.ndim == 3 else explanation.values
    feat_names = list(explanation.feature_names)
 
    # --- Panel 1: global feature importance ---------------------------------
    with st.container(border=True):
        st.subheader("What drives EV adoption predictions")
        st.caption("Average impact of each factor on the model's prediction, across all counties")
 
        mean_abs = np.abs(vals).mean(axis=0)
        order = np.argsort(-mean_abs)
        importance_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(feat_names[i], feat_names[i].replace('_', ' ')) for i in order],
            "Mean |impact|": [round(float(mean_abs[i]), 3) for i in order],
        })
        st.dataframe(
            importance_df,
            column_config={
                "Mean |impact|": st.column_config.ProgressColumn(
                    "Mean |impact|",
                    min_value=0,
                    max_value=float(mean_abs.max()) if len(mean_abs) else 1.0,
                    format="%.3f",
                )
            },
            hide_index=True,
            use_container_width=True,
        )
 
    st.write("")
 
    # --- State / county selection --------------------------------------------
    # Uses the same shared session_state as the Overview tab's pickers (via
    # render_state_county_selectors), so choosing a county here also updates
    # Overview, and vice versa — this is what lets the waterfall below know
    # which county to explain.
    state_choice, county_choice = render_state_county_selectors("features")
 
    st.write("")
 
    row = None
    if state_choice != "-" and county_choice != "-":
        match = modelOverviewDF[(modelOverviewDF["State"] == state_choice) & (modelOverviewDF["County"] == county_choice)]
        if not match.empty:
            row = match.iloc[0]
 
    # --- Panel 2: local explanation (waterfall) ------------------------------
    with st.container(border=True):
        if row is not None:
            st.subheader(f'Why {county_choice}, {state_choice} was predicted "{row["predicted_tier"]}"')
            st.caption("Starting from the model's baseline expectation, each factor below pushes "
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
            ))
            wf.update_layout(
                height=70 + 42 * len(labels),
                margin=dict(l=10, r=70, t=10, b=10),
                showlegend=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(wf, use_container_width=True)
        else:
            st.subheader("Why a county was predicted its tier")
            st.write("Select a state and county to see what drove that prediction")
 
 
# ---------------------------------------------------------------------------
# SENTIMENT
# ---------------------------------------------------------------------------
with tab_sentiment:
    with st.container(border=True):
        top_states_by_mention = top_states_by_mention.sort_values(by='post_count', ascending=False)
 
        def label_sentiment(sentiment_value):
            if sentiment_value >= 0.05:
                return 'positive'
            elif sentiment_value <= -0.05:
                return 'negative'
            return 'neutral'
 
        top_states_by_mention['sentiment'] = top_states_by_mention['avg_sentiment'].apply(label_sentiment)
 
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
        k1.metric("Positive", f"{pos_pct:.0f}%")
        k2.metric("Neutral", f"{neu_pct:.0f}%")
        k3.metric("Negative", f"{neg_pct:.0f}%")
 
        st.divider()
        st.subheader("Top 10 States By Mention Count")
 
        top10 = top_states_by_mention.head(10)[["states_mentioned", "post_count", "sentiment"]].rename(
            columns={"states_mentioned": "State", "post_count": "Mentions", "sentiment": "Sentiment"}
        )
        st.dataframe(
            top10,
            column_config={
                "Mentions": st.column_config.ProgressColumn(
                    "Mentions", min_value=0, max_value=float(top10["Mentions"].max()), format="%d"
                )
            },
            hide_index=True,
            use_container_width=True,
        )
 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap

# Page config
st.set_page_config(page_title="EV Adoption Dashboard", layout="wide", initial_sidebar_state="collapsed")

TIER_COLORS = {"High": "#4c8bf0", "Medium": "#e0a234", "Low": "#d9635b", "No data": "#808495"}


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

# Readable labels for the raw SHAP/model column names
FEATURE_LABELS = {
    "Total_population": "Total population",
    "pct_college_educated": "Educated population",
    "Median_household_income": "Median household income",
    "population_density": "Population density",
    "chargers_per_100k": "Charging stations per 100k",
    "avg_sentiment": "Reddit sentiment",
}


# Navigation
st.title("EV Adoption Dashboard")

# Overview and Features both need a State/County picker that stays in sync with
# each other. st.tabs() renders every tab's contents in the same run (they're
# just hidden/shown, not separate pages), so the two pickers can't share a
# widget `key` the way the old if/elif page routing let them — each needs its
# own key, synced through this canonical pair of session_state values instead.
if "selected_state" not in st.session_state:
    st.session_state.selected_state = "-"
if "selected_county" not in st.session_state:
    st.session_state.selected_county = "-"


def render_state_county_selectors(suffix):
    """Render a State/County selectbox pair with a unique widget key (suffix),
    reading from and writing back to the shared selected_state/selected_county
    session_state values so every tab's pickers stay in sync."""
    state_key = f"state_select_{suffix}"
    county_key = f"county_select_{suffix}"

    states = ["-"] + sorted(modelOverviewDF["State"].dropna().unique())
    state_index = states.index(st.session_state.selected_state) if st.session_state.selected_state in states else 0

    def on_state_change():
        st.session_state.selected_state = st.session_state[state_key]
        st.session_state.selected_county = "-"

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("State", states, index=state_index, key=state_key, on_change=on_state_change)

    current_state = st.session_state.selected_state
    county_opts = ["-"] + sorted(modelOverviewDF[modelOverviewDF["State"] == current_state]["County"].unique()) \
        if current_state != "-" else ["-"]
    county_index = county_opts.index(st.session_state.selected_county) if st.session_state.selected_county in county_opts else 0

    def on_county_change():
        st.session_state.selected_county = st.session_state[county_key]

    with c2:
        st.selectbox("County", county_opts, index=county_index, key=county_key, on_change=on_county_change)

    return st.session_state.selected_state, st.session_state.selected_county


tab_overview, tab_models, tab_features, tab_sentiment = st.tabs(
    ["Overview", "Models", "Features", "Sentiment"]
)


# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    row = None
    if st.session_state.selected_state != "-" and st.session_state.selected_county != "-":
        match = modelOverviewDF[(modelOverviewDF["State"] == st.session_state.selected_state) &
                                 (modelOverviewDF["County"] == st.session_state.selected_county)]
        if not match.empty:
            row = match.iloc[0]

    c3, c4, c5 = st.columns(3)
    with c3:
        # Predicted tier is only a genuine, honest prediction for held-out test
        # counties — for training counties the model already saw this row while
        # fitting, so "predicting" it again would just be reporting an in-sample
        # fit, not real predictive performance.
        if row is not None and row.get("split") == "test":
            st.metric("Predicted Adoption Tier", row["predicted_tier"])
        elif row is not None:
            st.metric("Predicted Adoption Tier", "Not available")
            st.caption("Training county")
        else:
            st.metric("Predicted Adoption Tier", "-")
    with c4:
        # Actual tier is ground truth from real vehicle registration data, so
        # it's known for every county regardless of split — always show it.
        st.metric("Actual Adoption Tier", row["actual_tier"] if row is not None else "-")
    with c5:
        val = f"{row['infrastructure_gap_score']:.2f}" if row is not None else "-"
        st.metric("Infrastructure Gap Score", val)

    render_state_county_selectors("overview")

    st.divider()
    col_map, col_feat = st.columns([1.4, 1])

    with col_map:
        with st.container(border=True):
            st.subheader("Predicted Adoption by County")
            st.caption("Grey counties have no matching record in the training data")

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
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=460)
            st.plotly_chart(fig, use_container_width=True)

    with col_feat:
        with st.container(border=True):
            st.subheader("County Features")
            feats = {
                "Median Household Income": f"${row['Median_household_income']:,.0f}" if row is not None else "-",
                "Charging Stations Per 100K": f"{row['chargers_per_100k']:,}" if row is not None else "-",
                "Population Density": f"{row['population_density']:,} /mi²" if row is not None else "-",
                "Educated Population": f"{row['pct_college_educated']}%" if row is not None else "-",
                "Reddit Sentiment": f"{row['avg_sentiment']}" if row is not None else "-",
                "Total Population": f"{row['Total_population']:,}" if row is not None else "-",
            }
            st.table(pd.DataFrame(feats.items(), columns=["Feature", "Value"]).set_index("Feature"))


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
with tab_models:
    with st.container(border=True):
        st.subheader("Metrics By Model")

        metrics_table = pd.DataFrame(MODEL_METRICS).transpose()
        metrics_table = metrics_table.rename(columns={
            "precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support",
        })
        st.dataframe(
            metrics_table[["Precision", "Recall", "F1", "Support"]].style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Support": "{:.0f}"}
            ),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------------
with tab_features:
    model_choice = st.radio(
        "Model", ["Logistic Regression", "Random Forest", "XGBoost", "Bagging"],
        horizontal=True, label_visibility="collapsed",
    )

    explanation = shap_values[model_choice]
    vals = explanation.values[..., 1] if explanation.values.ndim == 3 else explanation.values
    feat_names = list(explanation.feature_names)

    # --- Panel 1: global feature importance ---------------------------------
    with st.container(border=True):
        st.subheader("What drives EV adoption predictions")
        st.caption("Average impact of each factor on the model's prediction, across all counties")

        mean_abs = np.abs(vals).mean(axis=0)
        order = np.argsort(-mean_abs)
        importance_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(feat_names[i], feat_names[i].replace('_', ' ')) for i in order],
            "Mean |impact|": [round(float(mean_abs[i]), 3) for i in order],
        })
        st.dataframe(
            importance_df,
            column_config={
                "Mean |impact|": st.column_config.ProgressColumn(
                    "Mean |impact|",
                    min_value=0,
                    max_value=float(mean_abs.max()) if len(mean_abs) else 1.0,
                    format="%.3f",
                )
            },
            hide_index=True,
            use_container_width=True,
        )

    st.write("")

    # --- State / county selection --------------------------------------------
    # Uses the same shared session_state as the Overview tab's pickers (via
    # render_state_county_selectors), so choosing a county here also updates
    # Overview, and vice versa — this is what lets the waterfall below know
    # which county to explain.
    state_choice, county_choice = render_state_county_selectors("features")

    st.write("")

    row = None
    if state_choice != "-" and county_choice != "-":
        match = modelOverviewDF[(modelOverviewDF["State"] == state_choice) & (modelOverviewDF["County"] == county_choice)]
        if not match.empty:
            row = match.iloc[0]

    # --- Panel 2: local explanation (waterfall) ------------------------------
    with st.container(border=True):
        if row is not None:
            st.subheader(f'Why {county_choice}, {state_choice} was predicted "{row["predicted_tier"]}"')
            st.caption("Starting from the model's baseline expectation, each factor below pushes "
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
            ))
            wf.update_layout(
                height=70 + 42 * len(labels),
                margin=dict(l=10, r=70, t=10, b=10),
                showlegend=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(wf, use_container_width=True)
        else:
            st.subheader("Why a county was predicted its tier")
            st.write("Select a state and county to see what drove that prediction")


# ---------------------------------------------------------------------------
# SENTIMENT
# ---------------------------------------------------------------------------
with tab_sentiment:
    with st.container(border=True):
        top_states_by_mention = top_states_by_mention.sort_values(by='post_count', ascending=False)

        def label_sentiment(sentiment_value):
            if sentiment_value >= 0.05:
                return 'positive'
            elif sentiment_value <= -0.05:
                return 'negative'
            return 'neutral'

        top_states_by_mention['sentiment'] = top_states_by_mention['avg_sentiment'].apply(label_sentiment)

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
        k1.metric("Positive", f"{pos_pct:.0f}%")
        k2.metric("Neutral", f"{neu_pct:.0f}%")
        k3.metric("Negative", f"{neg_pct:.0f}%")

        st.divider()
        st.subheader("Top 10 States By Mention Count")

        top10 = top_states_by_mention.head(10)[["states_mentioned", "post_count", "sentiment"]].rename(
            columns={"states_mentioned": "State", "post_count": "Mentions", "sentiment": "Sentiment"}
        )
        st.dataframe(
            top10,
            column_config={
                "Mentions": st.column_config.ProgressColumn(
                    "Mentions", min_value=0, max_value=float(top10["Mentions"].max()), format="%d"
                )
            },
            hide_index=True,
            use_container_width=True,
        )
