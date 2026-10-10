"""
Location Analytics — Pay-by-Geography Benchmarking
=====================================================
Real data: BEA Regional Price Parities (RPP) and BEA MSA personal income,
sourced via the Unison-Consulting-Inc/EconData GitHub mirror of the
Bureau of Economic Analysis's public MARPP / MSA income tables (2023).

Goal: cluster US metro areas (MSAs) into a small number of location /
cost-of-labor tiers, using k-means, that a comp team could actually use to
set geographic pay differentials — the exact kind of deliverable this
project is standing in for (Aon's "location analytics" consulting work).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

DATA = "../data"
OUT = "../data"

# ---------------------------------------------------------------------------
# 1. Load + merge real BEA data
# ---------------------------------------------------------------------------
rpp = pd.read_parquet(f"{DATA}/rpp.parquet")
bea = pd.read_parquet(f"{DATA}/bea.parquet")

YEAR = 2023
msa_rpp = rpp[(rpp.type == "msa") & (rpp.year == YEAR)].copy()
msa_bea = bea[(bea.type == "msa") & (bea.year == YEAR)].copy()

df = msa_rpp.merge(
    msa_bea[["area", "population", "real_personal_income_pc"]],
    on="area", how="inner"
)
df = df.dropna(subset=["rpp_all", "real_personal_income_pc", "population"]).reset_index(drop=True)

# Nominal (non-cost-of-living-adjusted) personal income per capita —
# BEA's real figure is already deflated by RPP, so back it out:
# real = nominal / (RPP/100)  =>  nominal = real * (RPP/100)
df["nominal_income_pc"] = df["real_personal_income_pc"] * (df["rpp_all"] / 100)

print(f"Usable MSAs for {YEAR}: {len(df)}")

# ---------------------------------------------------------------------------
# 2. Feature set for clustering
# ---------------------------------------------------------------------------
# Cost-of-living sub-indices (RPP, national avg = 100) + nominal pay level.
# Housing/rent is the single biggest driver of metro cost variation, so it's
# kept as its own feature rather than folded only into rpp_all.
features = ["rpp_all", "rpp_housing_rent", "rpp_other_services", "rpp_goods", "nominal_income_pc"]
X_raw = df[features].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ---------------------------------------------------------------------------
# 3. Choose k — elbow + silhouette, same discipline as the CLV project
# ---------------------------------------------------------------------------
results = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

sel_df = pd.DataFrame(results)
print(sel_df)
sel_df.to_csv(f"{OUT}/k_selection.csv", index=False)

# Business constraint, same honest framing as the CLV segments: a comp team
# can realistically administer 5 pay tiers, not 8. Silhouette scores across
# k=4..6 are close (checked above) — tier count is a business choice given
# a roughly-flat statistical curve, not something the math uniquely dictates.
K = 5
km = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster_raw"] = km.fit_predict(X)
final_sil = silhouette_score(X, df["cluster_raw"])
print(f"\nFinal k={K}, silhouette={final_sil:.3f}")

# ---------------------------------------------------------------------------
# 4. Name tiers by actual cost ranking (Tier 1 = highest cost)
# ---------------------------------------------------------------------------
tier_order = (
    df.groupby("cluster_raw")["rpp_all"].mean().sort_values(ascending=False).index.tolist()
)
tier_map = {raw: i + 1 for i, raw in enumerate(tier_order)}
df["tier"] = df["cluster_raw"].map(tier_map)

summary = (
    df.groupby("tier")
    .agg(
        n_msas=("area", "count"),
        avg_rpp_all=("rpp_all", "mean"),
        avg_rpp_housing=("rpp_housing_rent", "mean"),
        avg_nominal_income=("nominal_income_pc", "mean"),
        total_population=("population", "sum"),
    )
    .round(1)
    .sort_index()
)
print("\n", summary)
summary.to_csv(f"{OUT}/tier_summary.csv")

# ---------------------------------------------------------------------------
# 5. Pay-differential multiplier per tier
# ---------------------------------------------------------------------------
# Standard comp-benchmarking practice: index each tier's cost to the
# national average (RPP=100) to get a location pay multiplier.
national_rpp = 100.0
tier_multiplier = (summary["avg_rpp_all"] / national_rpp).round(3)
tier_multiplier.name = "pay_multiplier"
tier_multiplier.to_csv(f"{OUT}/tier_multipliers.csv")
print("\nPay multipliers by tier:\n", tier_multiplier)

# ---------------------------------------------------------------------------
# 6. Validation — sanity-check against metros everyone agrees are expensive
# ---------------------------------------------------------------------------
known_high_cost = ["San Francisco", "San Jose", "New York", "Honolulu", "Boston", "Seattle"]
known_low_cost = ["Brownsville", "Beaumont", "Jackson, TN"]

def flag(name_fragment):
    matches = df[df.area.str.contains(name_fragment, case=False, na=False)]
    return matches[["area", "tier", "rpp_all"]]

print("\n--- Sanity check: known high-cost metros ---")
for c in known_high_cost:
    print(flag(c).to_string(index=False))

print("\n--- Sanity check: known lower-cost metros ---")
for c in known_low_cost:
    print(flag(c).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Save full dataset for the app
# ---------------------------------------------------------------------------
df_out = df[["area", "year", "rpp_all", "rpp_housing_rent", "rpp_other_services",
             "rpp_goods", "rpp_utilities", "population", "nominal_income_pc",
             "real_personal_income_pc", "tier"]].sort_values("rpp_all", ascending=False)
df_out.to_parquet(f"{OUT}/msa_tiers.parquet", index=False)
df_out.to_csv(f"{OUT}/msa_tiers.csv", index=False)

metrics = {
    "year": YEAR,
    "n_msas": int(len(df)),
    "k_selected": K,
    "silhouette_score": round(float(final_sil), 4),
    "k_sweep": results,
}
with open(f"{OUT}/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: msa_tiers.parquet/csv, tier_summary.csv, tier_multipliers.csv, model_metrics.json")
