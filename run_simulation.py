"""
run_simulation.py
End-to-end prototype demonstration for Section 10 (Reduction to Practice).

Pipeline:
  1. Generate a labeled training batch and train the RiskClassifier.
  2. Generate an unseen TEST batch (Batch A) and route it through three
     systems: No Protection, Naive Keyword Blocklist, and the proposed
     invention (risk+cost+cache-aware router).
  3. Generate a DRIFTED test batch (Batch B) containing an evasive
     exhaustion-attack variant not seen in training, to show how a static
     classifier degrades -- then simulate the feedback loop (retrain on
     logged outcomes) and re-evaluate on a fresh sample of the same
     distribution to show recovery.
  4. Emit: classifier report, per-system metrics table (CSV + printed),
     and comparison charts (PNG).
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_gen import generate_batch
from features import RequestFeatureExtractor, CacheLocalityEstimator, FEATURE_COLUMNS
from classifier import RiskClassifier
from router import assign_tier, estimate_cost_dollars, FEATURE_EXTRACTION_LATENCY_MS
from baselines import no_protection_tier, keyword_blocklist_tier

OUT_DIR = "results"
import os
os.makedirs(OUT_DIR, exist_ok=True)

KEYWORD_LATENCY_MS = 0.8  # cheap regex/embedding check, no ML model


# ---------------------------------------------------------------------------
# 1. Train classifier
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Training risk classifier on initial traffic sample")
print("=" * 70)

train_raw = generate_batch(n_total=6000, attack_fraction=0.15, seed=100)
extractor = RequestFeatureExtractor()
X_train_full = extractor.fit_transform(train_raw)
y_train_full = train_raw["true_type"]

X_tr, X_val, y_tr, y_val, raw_tr, raw_val = train_test_split(
    X_train_full, y_train_full, train_raw, test_size=0.25, random_state=7, stratify=y_train_full
)

clf = RiskClassifier(random_state=0)
clf.fit(X_tr, y_tr)

val_pred = clf.predict(X_val)
report_initial = classification_report(y_val, val_pred, output_dict=True)
print(classification_report(y_val, val_pred))

cache_est = CacheLocalityEstimator()


def route_and_score(raw_df, X_df, rng_seed, system_weights=None):
    """Run all three systems over the same batch and return a metrics dict + per-request dataframe."""
    rng = np.random.default_rng(rng_seed)
    cache_hit_prob = cache_est.estimate(raw_df, rng=rng)
    risk, subtype, proba = clf.risk_scores(X_df)

    results = {}
    per_request = raw_df[["request_id", "true_type"]].copy()
    per_request["risk_score"] = risk.values
    per_request["cache_hit_prob"] = cache_hit_prob

    systems = {
        "no_protection": (no_protection_tier(raw_df), 0.0),
        "keyword_blocklist": (keyword_blocklist_tier(raw_df), KEYWORD_LATENCY_MS),
        "proposed_invention": (assign_tier(risk), FEATURE_EXTRACTION_LATENCY_MS),
    }

    for name, (tier, latency_ms) in systems.items():
        cost = estimate_cost_dollars(raw_df, tier, cache_hit_prob)
        is_attack = raw_df["true_type"].isin(["injection", "exhaustion"]).values
        mitigated = tier.isin(["sandbox", "quarantine"]).values  # counted as "caught"

        detection_rate = mitigated[is_attack].mean() if is_attack.any() else np.nan
        false_positive_rate = mitigated[~is_attack].mean() if (~is_attack).any() else np.nan

        # subtype-specific detection, useful to show exhaustion coverage specifically
        inj_mask = (raw_df["true_type"] == "injection").values
        exh_mask = (raw_df["true_type"] == "exhaustion").values
        detection_injection = mitigated[inj_mask].mean() if inj_mask.any() else np.nan
        detection_exhaustion = mitigated[exh_mask].mean() if exh_mask.any() else np.nan

        results[name] = {
            "total_cost_usd": float(cost.sum()),
            "avg_latency_overhead_ms": float(latency_ms),
            "detection_rate_overall": float(detection_rate),
            "detection_rate_injection": float(detection_injection),
            "detection_rate_exhaustion": float(detection_exhaustion),
            "false_positive_rate": float(false_positive_rate),
            "n_requests": int(len(raw_df)),
        }
        per_request[f"tier_{name}"] = tier.values

    return results, per_request


# ---------------------------------------------------------------------------
# 2. Batch A: unseen, same distribution as training
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Routing Batch A (held-out, same distribution)")
print("=" * 70)

batch_a_raw = generate_batch(n_total=5000, attack_fraction=0.12, seed=200)
X_a = extractor.transform(batch_a_raw)
results_a, per_request_a = route_and_score(batch_a_raw, X_a, rng_seed=1)
print(json.dumps(results_a, indent=2))

# ---------------------------------------------------------------------------
# 3. Batch B: drifted exhaustion-attack variant (before retraining)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Routing Batch B (drifted/evasive exhaustion attacks) -- BEFORE feedback retrain")
print("=" * 70)

batch_b_raw = generate_batch(n_total=5000, attack_fraction=0.12, drifted=True, seed=300)
X_b = extractor.transform(batch_b_raw)
results_b_before, per_request_b_before = route_and_score(batch_b_raw, X_b, rng_seed=2)
print(json.dumps(results_b_before, indent=2))

# ---------------------------------------------------------------------------
# 4. Feedback loop: log Batch B outcomes (with ground-truth confirmation for
#    a sample, as would come from confirmed-abuse reports / anomaly audits)
#    and retrain the classifier.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Feedback loop -- retraining classifier on logged Batch B outcomes")
print("=" * 70)

# Simulate that 40% of Batch B's true labels get confirmed via outcome
# monitoring (actual resource consumption + abuse reports) and fed back,
# per Sec 6.1 "Feedback Loop".
confirmed = batch_b_raw.sample(frac=0.4, random_state=9)
X_confirmed = X_b.loc[confirmed.index]
y_confirmed = confirmed["true_type"]

clf.partial_retrain(X_confirmed, y_confirmed, X_tr, y_tr, hist_sample_frac=0.5, random_state=3)

# Re-validate on the original validation set to confirm no catastrophic forgetting
val_pred_after = clf.predict(X_val)
report_after_val = classification_report(y_val, val_pred_after, output_dict=True)
print("Validation performance on ORIGINAL distribution after retrain (should stay strong):")
print(classification_report(y_val, val_pred_after))

# ---------------------------------------------------------------------------
# 5. Batch C: fresh drifted traffic, AFTER retraining
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Routing Batch C (fresh drifted traffic) -- AFTER feedback retrain")
print("=" * 70)

batch_c_raw = generate_batch(n_total=5000, attack_fraction=0.12, drifted=True, seed=400)
X_c = extractor.transform(batch_c_raw)
results_c_after, per_request_c_after = route_and_score(batch_c_raw, X_c, rng_seed=4)
print(json.dumps(results_c_after, indent=2))

# ---------------------------------------------------------------------------
# 6. Assemble summary table + save artifacts
# ---------------------------------------------------------------------------
summary_rows = []
for label, res in [
    ("Batch A (in-distribution)", results_a),
    ("Batch B (drift, pre-retrain)", results_b_before),
    ("Batch C (drift, post-retrain)", results_c_after),
]:
    for system, m in res.items():
        row = {"batch": label, "system": system}
        row.update(m)
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df["cost_savings_vs_no_protection_pct"] = np.nan
for batch in summary_df["batch"].unique():
    base_cost = summary_df[(summary_df.batch == batch) & (summary_df.system == "no_protection")]["total_cost_usd"].values[0]
    mask = summary_df.batch == batch
    summary_df.loc[mask, "cost_savings_vs_no_protection_pct"] = (
        (base_cost - summary_df.loc[mask, "total_cost_usd"]) / base_cost * 100
    )

summary_df.to_csv(f"{OUT_DIR}/summary_metrics.csv", index=False)

with open(f"{OUT_DIR}/classifier_report_initial.json", "w") as f:
    json.dump(report_initial, f, indent=2)
with open(f"{OUT_DIR}/classifier_report_after_retrain.json", "w") as f:
    json.dump(report_after_val, f, indent=2)

feat_importance = clf.feature_importances()
feat_importance.to_csv(f"{OUT_DIR}/feature_importances.csv", header=["importance"])
print("\nFeature importances (post-retrain model):")
print(feat_importance.round(4).to_string())

per_request_a.to_csv(f"{OUT_DIR}/per_request_batchA.csv", index=False)
per_request_b_before.to_csv(f"{OUT_DIR}/per_request_batchB_pre_retrain.csv", index=False)
per_request_c_after.to_csv(f"{OUT_DIR}/per_request_batchC_post_retrain.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
pd.set_option("display.width", 160)
print(summary_df[["batch", "system", "detection_rate_overall", "detection_rate_injection",
                   "detection_rate_exhaustion", "false_positive_rate",
                   "total_cost_usd", "cost_savings_vs_no_protection_pct",
                   "avg_latency_overhead_ms"]].round(4).to_string(index=False))

print(f"\nArtifacts written to ./{OUT_DIR}/")