import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
)

PREDICTIONS_FILE = "data/llm_predictions.csv"
OUTPUTS_DIR = "outputs"

DIMENSIONS = [
    ("sentiment",      "true_sentiment",      "pred_sentiment",      ["positive", "neutral", "negative"]),
    ("adoption_stage", "true_adoption_stage", "pred_adoption_stage", ["exploring", "piloting", "scaling", "not_interested"]),
    ("main_barrier",   "true_main_barrier",   "pred_main_barrier",   ["cost", "skills", "trust", "none"]),
]

# Landis & Koch (1977) kappa interpretation scale
KAPPA_BANDS = [
    (0.00, "Slight"),
    (0.20, "Fair"),
    (0.40, "Moderate"),
    (0.60, "Substantial"),
    (0.80, "Almost perfect"),
]
BAND_COLORS = ["#ffcccc", "#ffe0cc", "#fff9c4", "#dcedc8", "#c8e6c9"]


def kappa_label(k):
    label = KAPPA_BANDS[0][1]
    for threshold, name in KAPPA_BANDS:
        if k >= threshold:
            label = name
    return label


def load_predictions():
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def export_misclassifications(rows):
    out_rows = []
    for row in rows:
        errors = {
            name: f"{row[tc]} → {row[pc]}"
            for name, tc, pc, _ in DIMENSIONS
            if row[tc] != row[pc]
        }
        if errors:
            out_rows.append({
                "row_id": row["row_id"],
                "response": row["response"],
                **{f"{name}_error": errors.get(name, "") for name, _, _, _ in DIMENSIONS},
            })
    if out_rows:
        path = os.path.join(OUTPUTS_DIR, "misclassifications.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Saved: {path}  ({len(out_rows)} rows with at least one misclassified label)")


def plot_summary(metrics):
    labels = [m["name"] for m in metrics]
    accuracies = [m["accuracy"] for m in metrics]
    kappas = [m["kappa"] for m in metrics]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    thresholds = [t for t, _ in KAPPA_BANDS] + [1.0]
    for i in range(len(KAPPA_BANDS)):
        ax.axhspan(thresholds[i], thresholds[i + 1], alpha=0.12,
                   color=BAND_COLORS[i], zorder=1)
        ax.text(len(labels) - 0.08,
                (thresholds[i] + thresholds[i + 1]) / 2,
                KAPPA_BANDS[i][1], va="center", ha="right",
                fontsize=7.5, color="#666", zorder=4)

    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy",
                   color="#4C72B0", zorder=3)
    bars2 = ax.bar(x + width / 2, kappas, width, label="Cohen's Kappa",
                   color="#DD8452", zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.016,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", zorder=5)
    for bar, m in zip(bars2, metrics):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.016,
                f"{bar.get_height():.2f}\n({kappa_label(m['kappa'])})",
                ha="center", va="bottom", fontsize=7.5, zorder=5)

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(
        "LLM vs Human Coder Agreement by Label Dimension\n"
        "(SME AI Adoption Survey Validation, n=500, Claude Haiku)",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=2)
    ax.legend(loc="upper left")

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "summary_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrices(all_true, all_pred, all_labels, all_names):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, true, pred, labels, name in zip(axes, all_true, all_pred, all_labels, all_names):
        cm = confusion_matrix(true, pred, labels=labels).astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0%")

        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}\n(row-normalised recall)", fontsize=10)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = cm_norm[i, j]
                count = int(cm[i, j])
                ax.text(j, i, f"{pct:.0%}\n({count})",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if pct > 0.6 else "black")

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_error_analysis(all_true, all_pred, all_labels, all_names):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, true, pred, labels, name in zip(axes, all_true, all_pred, all_labels, all_names):
        errors = {}
        for t, p in zip(true, pred):
            if t != p:
                errors[f"{t} → {p}"] = errors.get(f"{t} → {p}", 0) + 1

        if not errors:
            ax.text(0.5, 0.5, "No errors", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{name}\nError Pairs")
            continue

        sorted_errors = sorted(errors.items(), key=lambda x: -x[1])[:10]
        error_labels, counts = zip(*sorted_errors)
        total_errors = sum(errors.values())
        error_rate = total_errors / len(true)

        y = np.arange(len(error_labels))
        bars = ax.barh(y, counts, color="#C44E52", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(error_labels, fontsize=8)
        ax.set_xlabel("Error count")
        ax.set_title(
            f"{name}\n"
            f"Top misclassifications  ({total_errors} errors, {error_rate:.1%} error rate)",
            fontsize=10,
        )
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=8)
        ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "error_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    rows = load_predictions()
    print(f"Loaded {len(rows)} predictions\n")

    metrics = []
    all_true, all_pred, all_labels, all_names = [], [], [], []

    for name, true_col, pred_col, labels in DIMENSIONS:
        true = [r[true_col] for r in rows]
        pred = [r[pred_col] for r in rows]

        acc = accuracy_score(true, pred)
        kappa = cohen_kappa_score(true, pred)
        metrics.append({"name": name, "accuracy": acc, "kappa": kappa})

        print(f"=== {name.upper()} ===")
        print(f"  Accuracy:      {acc:.4f}")
        print(f"  Cohen's Kappa: {kappa:.4f}  [{kappa_label(kappa)}]")
        print(classification_report(true, pred, labels=labels, zero_division=0))

        all_true.append(true)
        all_pred.append(pred)
        all_labels.append(labels)
        all_names.append(name)

    export_misclassifications(rows)
    plot_summary(metrics)
    plot_confusion_matrices(all_true, all_pred, all_labels, all_names)
    plot_error_analysis(all_true, all_pred, all_labels, all_names)
    print("\nAll outputs saved to outputs/")


if __name__ == "__main__":
    main()
