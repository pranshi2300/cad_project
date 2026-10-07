"""
router.py
The "Tier-Selection / Decision Engine" (Sec 6.1) and a small cost model used
to estimate actual GPU-seconds consumed under each system, so we can report
cost savings in the Section 10 results.

Composite score (Sec 6.1):
    Score = w1*CostSignal + w2*CacheLocalitySignal - w3*RiskSignal

Tier assignment logic (Sec 6.1 / Claim 1 / Claim 4):
    risk > REJECT_THRESHOLD   -> quarantine  (reject, no execution)
    risk > SANDBOX_THRESHOLD  -> sandbox     (hard token/time caps, isolated)
    else                      -> full tier   (normal serving pool)
Tier assignment by risk happens FIRST and overrides cost/cache placement,
per the disclosure ("regardless of their cost/cache profile").
"""

import numpy as np
import pandas as pd

SANDBOX_THRESHOLD = 0.35
REJECT_THRESHOLD = 0.75

# Sandbox tier hard resource caps (Claim 4)
SANDBOX_MAX_OUTPUT_TOKENS = 256
SANDBOX_MAX_COMPUTE_SECONDS = 5.0

# Cost model constants (illustrative, not calibrated to any specific GPU SKU)
COST_PER_1K_PROMPT_TOKENS = 0.0015     # $ prefill cost
COST_PER_1K_OUTPUT_TOKENS = 0.006      # $ decode cost (dominant cost driver)
CACHE_HIT_PREFILL_DISCOUNT = 0.85      # cached prefix tokens are ~free to reprocess
FEATURE_EXTRACTION_LATENCY_MS = 4.5    # added latency for extraction + classification + scoring


def composite_score(cost_signal, cache_signal, risk_signal, w1=0.4, w2=0.3, w3=1.0):
    return w1 * cost_signal + w2 * cache_signal - w3 * risk_signal


def assign_tier(risk: pd.Series) -> pd.Series:
    tier = pd.Series("full", index=risk.index)
    tier[risk > SANDBOX_THRESHOLD] = "sandbox"
    tier[risk > REJECT_THRESHOLD] = "quarantine"
    return tier


def estimate_cost_dollars(df: pd.DataFrame, tier: pd.Series, cache_hit_prob: np.ndarray) -> pd.Series:
    """Estimate the dollar GPU cost actually incurred, given the assigned
    tier. Quarantined requests incur ~0 execution cost (rejected before
    running). Sandboxed requests are capped at SANDBOX_MAX_OUTPUT_TOKENS
    regardless of what they requested -- this is where the invention's
    cost savings on resource-exhaustion attacks come from."""
    prompt_cost = (df["prompt_length_tokens"] / 1000.0) * COST_PER_1K_PROMPT_TOKENS
    prompt_cost = prompt_cost * (1 - CACHE_HIT_PREFILL_DISCOUNT * cache_hit_prob)

    effective_output_tokens = df["predicted_output_tokens"].clip(upper=None).astype(float)
    capped = tier.eq("sandbox")
    effective_output_tokens = effective_output_tokens.where(~capped, np.minimum(effective_output_tokens, SANDBOX_MAX_OUTPUT_TOKENS))
    output_cost = (effective_output_tokens / 1000.0) * COST_PER_1K_OUTPUT_TOKENS

    total = prompt_cost + output_cost
    total = total.where(~tier.eq("quarantine"), 0.0)
    return total
