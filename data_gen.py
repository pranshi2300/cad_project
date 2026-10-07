"""
data_gen.py
Synthetic traffic generator for the risk-, cost-, and cache-aware routing prototype.

Generates requests of three underlying types (unknown to the system at runtime):
  - benign
  - injection            (content-based threat: prompt injection / jailbreak)
  - exhaustion           (resource-consumption threat: "sponge" / denial-of-wallet)

For each request we generate RAW attributes (as if produced by an upstream
tokenizer / embedding similarity search / session tracker), then a separate
`features.py` module derives the actual feature vector the classifier sees.
This keeps generation and feature extraction decoupled, similar to how a real
pipeline would look.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# A small pool of "recently cached" prompt-prefix IDs. Requests that reuse one
# of these prefixes get a high cache-hit probability; others are cache misses.
CACHE_POOL_SIZE = 40
_cache_pool = [f"prefix_{i}" for i in range(CACHE_POOL_SIZE)]


def _sample_benign(n, rng):
    return pd.DataFrame({
        "true_type": ["benign"] * n,
        "prompt_length_tokens": rng.gamma(shape=6, scale=40, size=n).astype(int) + 10,
        "jailbreak_similarity": rng.beta(1.2, 12, size=n),
        "semantic_similarity": rng.beta(1.2, 12, size=n),
        "repetition_ratio": rng.beta(1.5, 20, size=n),
        "predicted_output_tokens": rng.gamma(shape=4, scale=60, size=n).astype(int) + 20,
        "session_repeat_count": rng.poisson(0.4, size=n),
        "uses_cached_prefix": rng.random(n) < 0.55,   # benign traffic often repeats common prompts
    })


def _sample_injection(n, rng):
    return pd.DataFrame({
        "true_type": ["injection"] * n,
        "prompt_length_tokens": rng.gamma(shape=5, scale=55, size=n).astype(int) + 10,
        "jailbreak_similarity": rng.beta(9, 2.5, size=n),          # strong content signal
        "semantic_similarity": rng.beta(8, 2.5, size=n),           # strong dense semantic signal
        "repetition_ratio": rng.beta(2, 15, size=n),                # not repetitive
        "predicted_output_tokens": rng.gamma(shape=4, scale=70, size=n).astype(int) + 20,
        "session_repeat_count": rng.poisson(0.6, size=n),
        "uses_cached_prefix": rng.random(n) < 0.10,                 # novel adversarial prompts rarely hit cache
    })


def _sample_exhaustion(n, rng):
    return pd.DataFrame({
        "true_type": ["exhaustion"] * n,
        "prompt_length_tokens": rng.gamma(shape=20, scale=180, size=n).astype(int) + 200,  # very long
        "jailbreak_similarity": rng.beta(1.2, 15, size=n),           # no injection-style content
        "semantic_similarity": rng.beta(1.2, 15, size=n),            # no injection-style content
        "repetition_ratio": rng.beta(10, 2, size=n),                 # highly repetitive filler
        "predicted_output_tokens": rng.gamma(shape=15, scale=250, size=n).astype(int) + 500,  # asks for huge output
        "session_repeat_count": rng.poisson(6, size=n),               # bursty, repeated near-identical calls
        "uses_cached_prefix": rng.random(n) < 0.05,
    })


def _sample_exhaustion_v2(n, rng):
    """A *drifted*, evasive "low-and-slow" exhaustion-attack variant used to
    test the feedback loop's adaptive retraining. The attacker deliberately
    keeps prompt length, repetition, and per-request output demand close to
    the benign range (to evade a classifier trained only on the "obvious"
    v1 pattern), and instead achieves resource exhaustion through a much
    higher rate of near-duplicate session requests (many cheap-looking
    requests rather than one obviously expensive one)."""
    return pd.DataFrame({
        "true_type": ["exhaustion"] * n,
        "prompt_length_tokens": rng.gamma(shape=7, scale=55, size=n).astype(int) + 60,   # close to benign
        "jailbreak_similarity": rng.beta(1.3, 13, size=n),                                 # no content signal
        "semantic_similarity": rng.beta(1.3, 13, size=n),                                  # no content signal
        "repetition_ratio": rng.beta(2.5, 9, size=n),                                      # close to benign
        "predicted_output_tokens": rng.gamma(shape=5, scale=85, size=n).astype(int) + 90,  # close to benign
        "session_repeat_count": rng.poisson(5, size=n),        # main remaining tell: burst rate
        "uses_cached_prefix": rng.random(n) < 0.05,
    })


def generate_batch(n_total=4000, attack_fraction=0.12, drifted=False, seed=None):
    """Generate a labeled batch of synthetic requests.

    attack_fraction: fraction of traffic that is malicious (split evenly
        between injection and exhaustion).
    drifted: if True, exhaustion attacks use the evasive v2 distribution,
        simulating an attacker adapting after the first deployment period.
    """
    rng = np.random.default_rng(seed) if seed is not None else RNG
    n_attack = int(n_total * attack_fraction)
    n_injection = n_attack // 2
    n_exhaustion = n_attack - n_injection
    n_benign = n_total - n_injection - n_exhaustion

    parts = [_sample_benign(n_benign, rng), _sample_injection(n_injection, rng)]
    parts.append(_sample_exhaustion_v2(n_exhaustion, rng) if drifted else _sample_exhaustion(n_exhaustion, rng))

    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    df["request_id"] = [f"req_{i:06d}" for i in range(len(df))]
    df["tenant_id"] = rng.integers(0, 25, size=len(df))
    return df


if __name__ == "__main__":
    batch = generate_batch(2000, seed=1)
    print(batch["true_type"].value_counts())
    print(batch.head())
