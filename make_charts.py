import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/summary_metrics.csv")
systems = ["no_protection", "keyword_blocklist", "proposed_invention"]
labels = {"no_protection": "No Protection", "keyword_blocklist": "Keyword Blocklist", "proposed_invention": "Proposed Invention"}
colors = {"no_protection": "#c0392b", "keyword_blocklist": "#e67e22", "proposed_invention": "#2471a3"}
batches = df["batch"].unique().tolist()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Chart 1: exhaustion detection rate across batches (the key novelty claim)
ax = axes[0]
width = 0.25
x = range(len(batches))
for i, sysname in enumerate(systems):
    vals = [df[(df.batch == b) & (df.system == sysname)]["detection_rate_exhaustion"].values[0] for b in batches]
    ax.bar([xi + i * width for xi in x], vals, width=width, label=labels[sysname], color=colors[sysname])
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(["Batch A\n(in-dist.)", "Batch B\n(drift,\npre-retrain)", "Batch C\n(drift,\npost-retrain)"], fontsize=9)
ax.set_ylabel("Resource-Exhaustion Detection Rate")
ax.set_title("Exhaustion-Attack Detection\n(prior-art content filters cannot see this at all)")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)

# Chart 2: false positive rate
ax = axes[1]
for i, sysname in enumerate(systems):
    vals = [df[(df.batch == b) & (df.system == sysname)]["false_positive_rate"].values[0] for b in batches]
    ax.bar([xi + i * width for xi in x], vals, width=width, label=labels[sysname], color=colors[sysname])
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(["Batch A", "Batch B\n(pre-retrain)", "Batch C\n(post-retrain)"], fontsize=9)
ax.set_ylabel("False Positive Rate (benign requests mitigated)")
ax.set_title("False-Positive Impact on Legitimate Users")
ax.legend(fontsize=8)

# Chart 3: cost savings vs no-protection
ax = axes[2]
for i, sysname in enumerate(systems):
    vals = [df[(df.batch == b) & (df.system == sysname)]["cost_savings_vs_no_protection_pct"].values[0] for b in batches]
    ax.bar([xi + i * width for xi in x], vals, width=width, label=labels[sysname], color=colors[sysname])
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(["Batch A", "Batch B\n(pre-retrain)", "Batch C\n(post-retrain)"], fontsize=9)
ax.set_ylabel("Cost Savings vs. No Protection (%)")
ax.set_title("Serving-Cost Reduction")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/comparison_charts.png", dpi=150)
print("Saved results/comparison_charts.png")
