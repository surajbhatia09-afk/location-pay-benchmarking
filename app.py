"""
Location Analytics — Pay-by-Geography Benchmarking
Streamlit app: pick a metro, see its cost/pay tier and recommended
location pay multiplier for a given national base salary.
"""
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Location Pay Benchmarking", layout="wide")

# Streamlit Cloud runs the app with the working directory set to the repo
# root, not to this file's own folder -- so a plain relative path like
# "../data/..." breaks there even though it works locally. Anchor to this
# file's actual location instead, which is correct everywhere.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TIER_LABELS = {
    1: "Tier 1 — Major Metro, High Cost & Pay",
    2: "Tier 2 — Large Metro, Upper-Mid Cost",
    3: "Tier 3 — Mid-Size Metro, Near-Average Cost",
    4: "Tier 4 — Smaller Metro, Low Cost / Baseline Pay",
    5: "Tier 5 — Smaller Metro, Low Cost / Pay-Efficient",
}
TIER_COLORS = {1: "#7B2FF7", 2: "#2F7BF7", 3: "#2FC7B4", 4: "#F7A62F", 5: "#F7572F"}


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_DIR / "msa_tiers.parquet")
    mult = pd.read_csv(DATA_DIR / "tier_multipliers.csv").set_index("tier")["pay_multiplier"]
    df["tier_label"] = df["tier"].map(TIER_LABELS)
    df["pay_multiplier"] = df["tier"].map(mult)
    return df, mult


df, mult = load_data()

st.title("📍 Location Analytics — Pay-by-Geography Benchmarking")
st.caption(
    "K-means segmentation of 385 US metro areas into 5 cost/pay tiers, using real BEA "
    "Regional Price Parity and personal-income data (2023). Stands in for the kind of "
    "geographic pay-differential analysis done in compensation consulting (e.g. Aon's "
    "Rewards Innovation Consulting)."
)

tab1, tab2, tab3 = st.tabs(["🔎 Benchmark a Role", "🗺️ Explore All Tiers", "📊 Methodology & Validation"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        metro = st.selectbox("Choose a metro area", sorted(df["area"].unique()),
                              index=sorted(df["area"].unique()).index("Austin-Round Rock-San Marcos, TX")
                              if "Austin-Round Rock-San Marcos, TX" in df["area"].values else 0)
        national_base = st.number_input("National base salary ($)", min_value=20000, max_value=500000,
                                          value=100000, step=5000)

    row = df[df.area == metro].iloc[0]
    recommended = national_base * row.pay_multiplier

    with col2:
        st.metric("Tier", row.tier_label)
        st.metric("Cost index (RPP, national avg = 100)", f"{row.rpp_all:.1f}")
        st.metric("Recommended location-adjusted salary", f"${recommended:,.0f}",
                   delta=f"{(row.pay_multiplier - 1) * 100:+.1f}% vs. national base")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Housing/rent index", f"{row.rpp_housing_rent:.1f}")
    c2.metric("Avg. nominal income/capita (this metro)", f"${row.nominal_income_pc:,.0f}")
    c3.metric("Population", f"{row.population:,.0f}")

    st.info(
        "This is a **predictive-plus-prescriptive** pairing: the clustering + cost index is the "
        "predictive/diagnostic layer (which tier does this metro belong to, and why), and the "
        "multiplier applied to a base salary is the prescriptive layer — a specific, actionable "
        "recommendation, the same shape as the pay-equity remediation logic elsewhere in this portfolio."
    )

with tab2:
    st.subheader("All 5 tiers")
    summary = df.groupby(["tier", "tier_label"]).agg(
        MSAs=("area", "count"),
        Avg_RPP=("rpp_all", "mean"),
        Avg_Housing_RPP=("rpp_housing_rent", "mean"),
        Avg_Nominal_Income=("nominal_income_pc", "mean"),
        Pay_Multiplier=("pay_multiplier", "first"),
    ).round(1).reset_index().drop(columns="tier")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    fig = px.scatter(
        df, x="rpp_all", y="nominal_income_pc", color="tier_label",
        size="population", hover_name="area",
        color_discrete_map={TIER_LABELS[k]: v for k, v in TIER_COLORS.items()},
        labels={"rpp_all": "Cost index (RPP, national avg = 100)",
                "nominal_income_pc": "Nominal personal income per capita ($)"},
        title="Cost vs. pay level by metro — cluster separation",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note Tier 4 vs. Tier 5: both are low-cost (~90 RPP), but Tier 5 metros (Omaha, Kansas City, "
        "Tucson, Albuquerque, several Mountain/Plains energy & ag hubs) carry noticeably higher nominal "
        "pay for the same cost of living — a real 'pay-efficient' pocket the cost index alone would miss. "
        "That's the actual value k-means adds over just ranking metros by a single cost number."
    )

    st.subheader("Full metro table")
    st.dataframe(
        df[["area", "tier_label", "rpp_all", "rpp_housing_rent", "nominal_income_pc", "population", "pay_multiplier"]]
        .sort_values("rpp_all", ascending=False),
        use_container_width=True, hide_index=True
    )

with tab3:
    st.subheader("How the tiers were built")
    st.markdown(
        """
        - **Data**: BEA Regional Price Parities (RPP) and MSA personal income, 2023, 385 metro areas
          with complete data (via the BEA's public MARPP/MSA tables).
        - **Features**: overall RPP, housing/rent RPP, other-services RPP, goods RPP, and nominal
          personal income per capita (backed out from BEA's real income figure using the metro's own RPP).
        - **Method**: features standardized (z-score), then k-means. k was chosen by sweeping 2–8 and
          checking the elbow/silhouette curve — silhouette scores across k=4–8 were close (0.31–0.36),
          so **5 was picked as a business choice** (a comp team can realistically administer 5 tiers),
          not because the statistics uniquely pointed to 5. Same honest framing as the CLV project's
          segment-count choice.
        - **Validation**: every metro independently known to be expensive (San Francisco, San Jose,
          New York, Seattle, Boston, Honolulu) landed in Tier 1 without being told to; every metro
          known to be cheap (Brownsville, Beaumont, Jackson TN) landed in Tier 4. That's the sanity
          check for a diagnostic clustering model where there's no ground-truth label to score against.
        """
    )
    import json
    with open(DATA_DIR / "model_metrics.json") as f:
        metrics = json.load(f)
    st.json(metrics)
