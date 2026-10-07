"""
baselines.py
Two comparison systems referenced in Sec 10 (results should be compared
against (a) no protection and (b) a naive keyword-blocklist baseline).
"""

import pandas as pd
from router import estimate_cost_dollars, SANDBOX_MAX_OUTPUT_TOKENS  # noqa: F401


def no_protection_tier(df: pd.DataFrame) -> pd.Series:
    """Everything runs at full resources; no defense at all."""
    return pd.Series("full", index=df.index)


def keyword_blocklist_tier(df: pd.DataFrame, threshold=0.5) -> pd.Series:
    """Naive binary content-only filter: blocks (quarantines) requests whose
    jailbreak-similarity feature exceeds a fixed threshold, and otherwise
    allows full~-resource~ execution. Cannot see resource-consumption signals
    at all (mirrors prior-art keyword/embedding-similarity blocklists and
    binary allow/terminate patents such as US 12,437,058)."""
    tier = pd.Series("full", index=df.index)
    tier[df["jailbreak_similarity"] > threshold] = "quarantine"
    return tier
