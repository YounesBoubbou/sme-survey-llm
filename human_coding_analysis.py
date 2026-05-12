"""
Inter-rater reliability analysis: human coders vs LLM vs ground truth.

Usage:
    Place coder CSV files in coding_task/ (named coder_<name>.csv).
    Run: python human_coding_analysis.py

Each coder CSV must have columns: coder_id, row_id, sentiment, adoption_stage, main_barrier
(exported automatically by coding_task/coder.html).
"""

import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import cohen_kappa_score, classification_report

CODING_DIR = "coding_task"
TRUTH_FILE = os.path.join(CODING_DIR, "coding_truth.csv")
OUTPUTS_DIR = "outputs"

DIMENSIONS = ["sentiment", "adoption_stage", "main_barrier"]

KAPPA_BANDS = [(0.0, "Slight"), (0.2, "Fair"), (0.4, "Moderate"),
               (0.6, "Substantial"), (0.8, "Almost perfect")]
BAND_COLORS = ["#ffcccc", "#ffe0cc", "#fff9c4", "#dcedc8", "#c8e6c9"]


# ── Kappa helpers ────────────────────────────────────────────────────────────

def kappa_label(k):
    lbl = KAPPA_BANDS[0][1]
    for t, name in KAPPA_BANDS:
        if k >= t:
            lbl = name
    return lbl


def safe_kappa(a, b):
    try:
        return cohen_kappa_score(a, b)
    except Exception:
        return float("nan")


def bootstrap_kappa(a, b, n_boot=1000):
    """Returns (kappa, lower_95ci, upper_95ci)."""
    k = safe_kappa(a, b)
    n = len(a)
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        bk = safe_kappa([a[i] for i in idx], [b[i] for i in idx])
        if not np.isnan(bk):
            boot.append(bk)
    if not boot:
        return k, float("nan"), float("nan")
    return k, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def fleiss_kappa(ratings):
    """
    Fleiss's kappa for 3+ raters.
    ratings: list of lists, shape (n_items, n_raters), each entry is a category label.
    """
    n_items = len(ratings)
    n_raters = len(ratings[0])
    categories = sorted(set(v for row in ratings for v in row))
    n_cats = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    table = np.zeros((n_items, n_cats))
    for i, row in enumerate(ratings):
        for v in row:
            table[i, cat_idx[v]] += 1

    P_i = (1.0 / (n_raters * (n_raters - 1))) * (
        (table ** 2).sum(axis=1) - n_raters
    )
    P_o = P_i.mean()
    p_j = table.sum(axis=0) / (n_items * n_raters)
    P_e = (p_j ** 2).sum()
    if P_e >= 1:
        return 1.0
    return (P_o - P_e) / (1 - P_e)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_truth():
    truth = {}
    with open(TRUTH_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = int(row["row_id"])
            truth[rid] = {
                "sentiment":      row["true_sentiment"],
                "adoption_stage": row["true_adoption_stage"],
                "main_barrier":   row["true_main_barrier"],
                "pred_sentiment":      row["pred_sentiment"],
                "pred_adoption_stage": row["pred_adoption_stage"],
                "pred_main_barrier":   row["pred_main_barrier"],
            }
    return truth


def load_coder(path):
    labels = {}
    coder_id = None
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            coder_id = row["coder_id"]
            labels[int(row["row_id"])] = {
                "sentiment":      row["sentiment"],
                "adoption_stage": row["adoption_stage"],
                "main_barrier":   row["main_barrier"],
            }
    return {"name": coder_id or os.path.basename(path), "labels": labels}


# ── Analysis ─────────────────────────────────────────────────────────────────

def aligned_labels(ids, source, dim):
    return [source[i][dim] for i in ids]


def compute_all_kappas(coders, truth, common_ids):
    ids = sorted(common_ids)
    results = {}

    for dim in DIMENSIONS:
        true_labels  = aligned_labels(ids, truth, dim)
        pred_labels  = aligned_labels(ids, {i: {"pred_sentiment": truth[i]["pred_sentiment"],
                                                 "pred_adoption_stage": truth[i]["pred_adoption_stage"],
                                                 "pred_main_barrier": truth[i]["pred_main_barrier"]}
                                            for i in ids},
                                      f"pred_{dim}")
        coder_label_lists = [aligned_labels(ids, c["labels"], dim) for c in coders]

        # Human-human pairwise kappas
        hh_kappas = []
        hh_pairs = []
        for (i, ci), (j, cj) in combinations(enumerate(coders), 2):
            k, lo, hi = bootstrap_kappa(coder_label_lists[i], coder_label_lists[j])
            hh_kappas.append((ci["name"], cj["name"], k, lo, hi))
            hh_pairs.append(k)

        # Fleiss kappa (3+ raters)
        fleiss_k = None
        if len(coders) >= 3:
            fleiss_k = fleiss_kappa(list(zip(*coder_label_lists)))

        # LLM-human pairwise kappas
        lh_kappas = []
        for c, cl in zip(coders, coder_label_lists):
            k, lo, hi = bootstrap_kappa(pred_labels, cl)
            lh_kappas.append((c["name"], k, lo, hi))

        # LLM-ground truth
        lgt_k, lgt_lo, lgt_hi = bootstrap_kappa(true_labels, pred_labels)

        results[dim] = {
            "hh_pairs":  hh_kappas,
            "hh_mean":   float(np.nanmean(hh_pairs)) if hh_pairs else float("nan"),
            "hh_min":    float(np.nanmin(hh_pairs)) if hh_pairs else float("nan"),
            "hh_max":    float(np.nanmax(hh_pairs)) if hh_pairs else float("nan"),
            "fleiss":    fleiss_k,
            "lh_pairs":  lh_kappas,
            "lh_mean":   float(np.nanmean([x[1] for x in lh_kappas])),
            "lh_min":    float(np.nanmin([x[1] for x in lh_kappas])),
            "lh_max":    float(np.nanmax([x[1] for x in lh_kappas])),
            "lgt":       (lgt_k, lgt_lo, lgt_hi),
        }

    return results


# ── Output ───────────────────────────────────────────────────────────────────

def print_report(coders, results):
    names = [c["name"] for c in coders]
    print(f"\n{'='*60}")
    print(f"  INTER-RATER RELIABILITY REPORT")
    print(f"  Coders: {', '.join(names)}")
    print(f"{'='*60}")

    for dim in DIMENSIONS:
        r = results[dim]
        print(f"\n── {dim.upper()} ──")

        print("  Human–Human pairwise κ:")
        for a, b, k, lo, hi in r["hh_pairs"]:
            print(f"    {a} vs {b}: {k:.3f}  [95% CI: {lo:.3f}–{hi:.3f}]  {kappa_label(k)}")
        print(f"    Average: {r['hh_mean']:.3f}")
        if r["fleiss"] is not None:
            print(f"    Fleiss's κ: {r['fleiss']:.3f}  {kappa_label(r['fleiss'])}")

        print("  LLM–Human κ:")
        for name, k, lo, hi in r["lh_pairs"]:
            print(f"    LLM vs {name}: {k:.3f}  [95% CI: {lo:.3f}–{hi:.3f}]  {kappa_label(k)}")
        print(f"    Average: {r['lh_mean']:.3f}")

        k, lo, hi = r["lgt"]
        print(f"  LLM–Ground Truth κ: {k:.3f}  [95% CI: {lo:.3f}–{hi:.3f}]  {kappa_label(k)}")

        gap = r["lh_mean"] - r["hh_mean"]
        verdict = "≥ human-human (LLM matches coders)" if gap >= 0 else f"{abs(gap):.3f} below human-human"
        print(f"  ▶  LLM–Human avg is {verdict}")

    print(f"\n{'='*60}\n")


def plot_comparison(results):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    dims = DIMENSIONS
    x = np.arange(len(dims))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5))

    thresholds = [t for t, _ in KAPPA_BANDS] + [1.0]
    for i in range(len(KAPPA_BANDS)):
        ax.axhspan(thresholds[i], thresholds[i + 1], alpha=0.1,
                   color=BAND_COLORS[i], zorder=1)
        ax.text(len(dims) - 0.05,
                (thresholds[i] + thresholds[i + 1]) / 2,
                KAPPA_BANDS[i][1], va="center", ha="right",
                fontsize=7.5, color="#777", zorder=4)

    def yerr(mn, lo, hi):
        return [[mn - lo], [hi - mn]]

    hh_means = [results[d]["hh_mean"] for d in dims]
    lh_means = [results[d]["lh_mean"] for d in dims]
    lgt_k    = [results[d]["lgt"][0] for d in dims]

    hh_lo = [results[d]["hh_min"] for d in dims]
    hh_hi = [results[d]["hh_max"] for d in dims]
    lh_lo = [results[d]["lh_min"] for d in dims]
    lh_hi = [results[d]["lh_max"] for d in dims]

    hh_err = [[hh_means[i] - hh_lo[i] for i in range(len(dims))],
              [hh_hi[i] - hh_means[i] for i in range(len(dims))]]
    lh_err = [[lh_means[i] - lh_lo[i] for i in range(len(dims))],
              [lh_hi[i] - lh_means[i] for i in range(len(dims))]]

    b1 = ax.bar(x - width, hh_means, width, label="Human–Human κ (avg)",
                color="#5c6bc0", zorder=3,
                yerr=hh_err, capsize=5, error_kw={"elinewidth": 1.5})
    b2 = ax.bar(x,          lh_means, width, label="LLM–Human κ (avg)",
                color="#ef6c00", zorder=3,
                yerr=lh_err, capsize=5, error_kw={"elinewidth": 1.5})
    b3 = ax.bar(x + width,  lgt_k,   width, label="LLM–Ground Truth κ",
                color="#388e3c", zorder=3)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.018,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8, zorder=5)

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Cohen's κ")
    ax.set_title(
        "Inter-Rater Reliability: Human Coders vs LLM vs Ground Truth\n"
        "(error bars show min/max across pairs; "
        "background bands: Landis & Koch 1977)",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=2)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "irr_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    coder_files = sorted(glob.glob(os.path.join(CODING_DIR, "coder_*.csv")))
    if not coder_files:
        print(f"No coder CSV files found in {CODING_DIR}/")
        print("Expected files named  coder_<name>.csv  (exported from coder.html)")
        return
    if len(coder_files) < 2:
        print(f"Found only 1 coder file. Need at least 2 to compute human-human agreement.")
        print("Proceeding with LLM-human analysis only.\n")

    truth = load_truth()
    coders = [load_coder(f) for f in coder_files]
    print(f"Loaded {len(coders)} coder(s): {[c['name'] for c in coders]}")

    # Common row IDs across all coders and truth
    common = set(truth.keys())
    for c in coders:
        common &= set(c["labels"].keys())
    print(f"Common rows for analysis: {len(common)}")

    if not common:
        print("No common rows found. Check that row_id values match between files.")
        return

    results = compute_all_kappas(coders, truth, common)
    print_report(coders, results)
    plot_comparison(results)


if __name__ == "__main__":
    main()
