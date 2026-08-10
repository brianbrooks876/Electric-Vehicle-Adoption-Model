# Import necessary packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap

# Page config
st.set_page_config(page_title="EV Adoption Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Hexcodes for chloropleth colours
TIER_COLORS = {"High": "#4c8bf0", "Medium": "#e0a234", "Low": "#d9635b", "No data": "#808495"}


# SHAP Analysis data loading
@st.cache_resource
def load_shap_values():
    return {
        "Logistic Regression": joblib.load('models/shap_values_lr.joblib'),
        "Random Forest": joblib.load('models/shap_values_rf.joblib'),
        "XGBoost": joblib.load('models/shap_values_xg.joblib'),
        "Bagging": joblib.load('models/shap_values_bg.joblib'),
    }

# Chloropleth data loading
@st.cache_data
def load_counties_geojson():
    import json
    import urllib.request
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)

# Load shap values, chloropleth, and all necessary DataFrames, normalize all necessary values. Filter for only the test split
shap_values = load_shap_values()
counties_geojson = load_counties_geojson()
ALL_COUNTY_FIPS = [feat["id"] for feat in counties_geojson["features"]]
modelOverviewDF = pd.read_csv('deploymentDfnew.csv')
modelOverviewDF["StCnty FIPS Code"] = modelOverviewDF["StCnty FIPS Code"].astype(str).str.zfill(5)
MODEL_METRICS = (modelmetricsDF.set_index('Metrics').transpose().iloc[1:].to_dict(orient='index'))
testOnlyDF = modelOverviewDF[modelOverviewDF["split"] == "test"].copy()
modelmetricsDF = pd.read_csv('model_metrics_new.csv')
top_states_by_mention = pd.read_csv('deploymentDF3new.csv')

# Readable labels for the Features tab
FEATURE_LABELS = {
    "Total_population": "Total population",
    "pct_college_educated": "Educated population",
    "Median_household_income": "Median household income",
    "population_density": "Population density",
    "chargers_per_100k": "Charging stations per 100k",
    "avg_sentiment": "Reddit sentiment",
}


# Navigation Page
st.title("EV Adoption Dashboard")
st.info(
    "**This demo only shows counties held out of model training.** "
    "For any county the model was trained on, its \"prediction\" would just be "
    "reporting what the model already saw and fit to — not a genuine test of "
    "accuracy. Restricting the dropdowns to held-out counties means every "
    "prediction shown here is an honest, out-of-sample result.",
)

# State/County picker
st.session_state.selected_state = "-"
if "selected_county" not in st.session_state:
    st.session_state.selected_county = "-"

# State/County selectbox pair
def render_state_county_selectors(suffix):
    state_key = f"state_select_{suffix}"
    county_key = f"county_select_{suffix}"

    states = ["-"] + sorted(testOnlyDF["State"].dropna().unique())
    state_index = states.index(st.session_state.selected_state) if st.session_state.selected_state in states else 0

    def on_state_change():
        st.session_state.selected_state = st.session_state[state_key]
        st.session_state.selected_county = "-"

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("State", states, index=state_index, key=state_key, on_change=on_state_change)

    current_state = st.session_state.selected_state
    county_opts = ["-"] + sorted(testOnlyDF[testOnlyDF["State"] == current_state]["County"].unique()) \
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


# Overview Page
with tab_overview:
    row = None
    if st.session_state.selected_state != "-" and st.session_state.selected_county != "-":
        match = modelOverviewDF[(modelOverviewDF["State"] == st.session_state.selected_state) &
                                 (modelOverviewDF["County"] == st.session_state.selected_county)]
        if not match.empty:
            row = match.iloc[0]

    # Cards
    c3, c4, c5 = st.columns(3)
    with c3:
        st.metric("Actual Adoption Tier", row["actual_tier"] if row is not None else "-")
    with c4:
        st.metric("Predicted Adoption Tier", row["predicted_tier"] if row is not None else "-")
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

            # The geojson ships ~3,221 counties; modelOverviewDF only covers the ~3,133 counties in the training set. Left-joining onto the full FIPS list means every shape gets a color — the remainder ("No data") — instead of being left blank, which used to look like a rendering bug.
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
                "Median Household Income": f"${row['Median_household_income']:,.2f}" if row is not None else "-",
                "Charging Stations Per 100K": f"{row['chargers_per_100k']:,.2f}" if row is not None else "-",
                "Population Density": f"{row['population_density']:,.2f} /mi²" if row is not None else "-",
                "Educated Population": f"{row['pct_college_educated']:.2f}%" if row is not None else "-",
                "Reddit Sentiment": f"{row['avg_sentiment']:.2f}" if row is not None else "-",
                "Total Population": f"{row['Total_population']:,.0f}" if row is not None else "-",
            }
            st.table(pd.DataFrame(feats.items(), columns=["Feature", "Value"]).set_index("Feature"))


# Model Page
with tab_models:
    with st.container(border=True):
        st.subheader("Metrics By Model")

        metrics_table = pd.DataFrame(MODEL_METRICS).transpose()
        metrics_table = metrics_table.rename(columns={
            "precision": "Precision", "recall": "Recall", "f1-score": "F1",
        })
        st.dataframe(
            metrics_table[["Precision", "Recall", "F1", "ROC-AUC"]].style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}
            ),
            use_container_width=True,
        )

# Features Page
with tab_features:
    model_choice = st.radio(
        "Model", ["Logistic Regression", "Random Forest", "XGBoost", "Bagging"],
        horizontal=True, label_visibility="collapsed",
    )

    explanation = shap_values[model_choice]
    vals = explanation.values[..., 1] if explanation.values.ndim == 3 else explanation.values
    feat_names = list(explanation.feature_names)

    # Global feature importance
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

    state_choice, county_choice = render_state_county_selectors("features")

    st.write("")

    row = None
    if state_choice != "-" and county_choice != "-":
        match = modelOverviewDF[(modelOverviewDF["State"] == state_choice) & (modelOverviewDF["County"] == county_choice)]
        if not match.empty:
            row = match.iloc[0]

    # SHAP Waterfall
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

# Sentiment Page
with tab_sentiment:
    with st.container(border=True):
        top_states_by_mention = top_states_by_mention.sort_values(by='post_count', ascending=False)

        # Define a function to tag sentiments as negative, neutral or positive
        def label_sentiment(sentiment_value):
            if sentiment_value >= 0.05:
                return 'positive'
            elif sentiment_value <= -0.05:
                return 'negative'
            return 'neutral'

        top_states_by_mention['sentiment'] = top_states_by_mention['avg_sentiment'].apply(label_sentiment)

        by_sentiment = top_states_by_mention.groupby('sentiment')['post_count'].sum()
        total_mentions = by_sentiment.sum()
        pos_pct = by_sentiment.get('positive', 0) / total_mentions * 100
        neu_pct = by_sentiment.get('neutral', 0) / total_mentions * 100
        neg_pct = by_sentiment.get('negative', 0) / total_mentions * 100

        # Sentiment Cards
        k1, k2, k3 = st.columns(3)
        k1.success(f"**Positive**\n\n{pos_pct:.0f}%")
        k2.warning(f"**Neutral**\n\n{neu_pct:.0f}%")
        k3.error(f"**Negative**\n\n{neg_pct:.0f}%")

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
