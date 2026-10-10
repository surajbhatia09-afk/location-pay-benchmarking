# Location Analytics — Pay-by-Geography Benchmarking

K-means segmentation of US metro areas into cost/pay tiers, built to demonstrate
how location analytics, clustering, and segmentation combine with a prescriptive
pay-differential recommendation — the kind of work done in compensation
consulting (e.g. geographic pay-banding at Aon's Rewards Innovation practice).

The code (data pipeline, clustering, app) was written with AI assistance; the
feature choices, validation approach, tier-count decision, and the reading of
the results are mine.

## Business problem

A company setting pay for a role needs to know: how much more (or less) should
someone in Metro X earn than the same role in Metro Y, based purely on cost of
living and local pay levels — not negotiation, not individual performance. This
needs a defensible, data-driven tiering of US metros, not a one-off guess per city.

## Data — real, not synthetic

- **BEA Regional Price Parities (RPP)**, 2023, MSA level — overall, housing/rent,
  goods, other-services, and utilities sub-indices (national average = 100).
- **BEA MSA personal income**, 2023 — real personal income per capita, backed
  out to **nominal** income per capita using each metro's own RPP
  (`nominal = real × RPP/100`).
- Source: BEA's public MARPP / MSA income tables, via the
  [Unison-Consulting-Inc/EconData](https://github.com/Unison-Consulting-Inc/EconData)
  mirror (parquet files built directly from BEA's published data).
- **385 metro areas** with complete data for 2023.

## Method

1. **Features**: RPP (all, housing/rent, goods, other services) + nominal
   personal income per capita, standardized (z-score).
2. **Clustering**: k-means, k swept 2–8. Silhouette scores for k=4–8 were close
   (0.31–0.36) — **k=5 was chosen as a business decision** (a comp team can
   realistically administer five pay tiers), not because the statistics
   uniquely pointed there. Same honest framing as this portfolio's CLV project.
3. **Tier naming**: clusters ranked and labeled 1 (highest cost) through 5
   (one of two lowest-cost clusters), not left as unordered cluster IDs.
4. **Prescriptive layer**: each tier's average RPP is indexed to the national
   average (RPP = 100) to produce a **pay multiplier** — a specific, applicable
   adjustment to a national base salary, not just a diagnostic label.

## Results

| Tier | # MSAs | Avg. cost index (RPP) | Avg. housing RPP | Avg. nominal income/capita | Pay multiplier |
|---|---|---|---|---|---|
| 1 — Major Metro, High Cost & Pay | 20 | 111.5 | 162.1 | $86,486 | **1.115×** |
| 2 — Large Metro, Upper-Mid Cost | 52 | 103.5 | 104.9 | $54,101 | **1.035×** |
| 3 — Mid-Size Metro, Near-Average Cost | 76 | 97.7 | 101.6 | $56,533 | **0.977×** |
| 4 — Smaller Metro, Low Cost / Baseline Pay | 181 | 90.4 | 66.7 | $41,203 | **0.904×** |
| 5 — Smaller Metro, Low Cost / Pay-Efficient | 56 | 90.3 | 74.9 | $44,940 | **0.903×** |

Final silhouette score at k=5: **0.361**.

**Validation (no ground-truth labels exist for this, so this is the actual
check):** every metro independently well-known to be expensive — San Francisco,
San Jose, New York, Seattle, Boston, Honolulu — landed in Tier 1 without being
told to. Every metro well-known to be cheap — Brownsville, Beaumont, Jackson
TN — landed in Tier 4.

**What segmentation adds over a single cost ranking:** Tiers 4 and 5 have
almost identical cost indexes (90.4 vs. 90.3) but materially different nominal
pay ($41.2K vs. $44.9K). Tier 5 — Omaha, Kansas City, Tucson, Albuquerque, and
several Mountain/Plains energy and agriculture hubs — is a genuine "pay-efficient"
pocket that a simple cost-of-living ranking alone would never surface, because
it isn't about cost, it's about local pay levels at a given cost. That's the
actual value k-means adds here.

## Why this is both predictive and prescriptive

- **Predictive/diagnostic layer**: the clustering itself — which cost/pay tier
  does a given metro belong to, and why (its own RPP sub-indices and income
  level), the same statistical-modeling-without-labels territory as this
  portfolio's CLV segmentation and the Aon pay-equity work.
- **Prescriptive layer**: the pay multiplier — a specific, applicable number
  ("adjust this role's national base pay by +11.5% / −9.7% / etc.") derived
  from the diagnostic tier, not just a label. Recommending an action given a
  prediction and a cost/benefit tradeoff is the actual definition of
  prescriptive analytics, and this project does it explicitly.

## What I'd change today

- **No time dimension used.** RPP data goes back to 2008 — a next version
  would track tier migration over time (does a metro's tier change as it
  grows or gentrifies?) rather than a single-year snapshot.
- **No occupation-level granularity.** This uses metro-wide income, not
  role-specific wage data (e.g., BLS OEWS by occupation), so the multiplier
  is a reasonable general-purpose adjustment, not a precise figure for every
  job family — a real compensation engagement would layer in occupation-level
  wage data where it's available.
- **Housing/rent RPP is the single most influential feature** (it has by far
  the widest spread across tiers) — worth testing cluster stability with and
  without it weighted differently, rather than treating all five features equally.
