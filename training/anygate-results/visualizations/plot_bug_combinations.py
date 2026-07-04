import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import os
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comprehensive_test_metrics.json")
OUT_DIR   = r"C:\Users\100998539\Desktop\training\anygate-results\visualizations"

# ── Load data ─────────────────────────────────────────────────────────────────
with open(JSON_PATH, "r") as f:
    data = json.load(f)

bug_combination_metrics = data["bug_combination_metrics"]

# ── Pretty labels: format raw keys into readable multi-line strings ───────────
def format_key(key):
    parts = key.split("|")
    pretty_parts = []
    for p in parts:
        p = p.strip().replace("_", " ")
        pretty_parts.append(p.title())
    return "\n".join(pretty_parts)

keys   = list(bug_combination_metrics.keys())
labels = [format_key(k) for k in keys]
prec   = [bug_combination_metrics[k]["precision"] * 100 for k in keys]
rec    = [bug_combination_metrics[k]["recall"]    * 100 for k in keys]
f1     = [bug_combination_metrics[k]["f1"]        * 100 for k in keys]

x     = np.arange(len(labels))
width = 0.26

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(28, 12))

bars_p = ax.bar(x - width, prec, width, label="Precision", color="#4C72B0", edgecolor="white")
bars_r = ax.bar(x,         rec,  width, label="Recall",    color="#DD8452", edgecolor="white")
bars_f = ax.bar(x + width, f1,   width, label="F1 Score",  color="#55A868", edgecolor="white")

def add_labels(bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            f"{h:.2f}%",
            ha="center", va="bottom",
            fontsize=6.5, rotation=90,
            color="#222222"
        )

add_labels(bars_p)
add_labels(bars_r)
add_labels(bars_f)

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Evaluation metrics on each bug combination", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "bug_combination_metrics.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to: {out_path}")
plt.show()
