"""
features.py
Implements:
  - RequestFeatureExtractor: turns raw request attributes into the feature
    vector consumed by the ML risk classifier (Sec 6.1 "Request Feature
    Extractor" + "ML Risk Classifier" inputs).
  - CacheLocalityEstimator: estimates P(prefix already resident in KV-cache)
    for a serving instance, with a simple time-decaying warm-prefix pool
    (Sec 6.1 "Cache-Locality Estimator").
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "prompt_length_tokens",
    "jailbreak_similarity",
    "semantic_similarity",
    "repetition_ratio",
    "predicted_output_tokens",
    "session_repeat_count",
]


class RequestFeatureExtractor:
    """Normalizes raw attributes into model-ready features. In a production
    system this stage would call an embedding model + prefix search; here we
    take the already-simulated raw signals and just do scaling/clipping,
    mirroring a real preprocessing stage."""

    def __init__(self):
        self._fit_stats = None

    def fit(self, df: pd.DataFrame):
        self._fit_stats = {
            col: (df[col].mean(), df[col].std() + 1e-6) for col in FEATURE_COLUMNS
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fit_stats is not None, "call fit() first"
        out = pd.DataFrame(index=df.index)
        for col in FEATURE_COLUMNS:
            mean, std = self._fit_stats[col]
            out[col] = (df[col] - mean) / std
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class CacheLocalityEstimator:
    """Simulates a KV-cache warm-prefix pool. Requests flagged
    `uses_cached_prefix=True` in the synthetic generator represent traffic
    that shares a prefix with recently served requests; the estimator
    returns a hit probability that decays as the pool churns, standing in
    for a real prefix-similarity lookup against serving-instance KV-caches.
    """

    def __init__(self, base_hit_prob=0.85, base_miss_prob=0.03, churn=0.0):
        self.base_hit_prob = base_hit_prob
        self.base_miss_prob = base_miss_prob
        self.churn = churn  # fraction of warm prefixes evicted per batch (cache pressure)

    def estimate(self, df: pd.DataFrame, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        hit_prob = self.base_hit_prob * (1 - self.churn)
        probs = np.where(
            df["uses_cached_prefix"].values,
            rng.normal(hit_prob, 0.05, size=len(df)).clip(0, 1),
            rng.normal(self.base_miss_prob, 0.02, size=len(df)).clip(0, 1),
        )
        return probs
