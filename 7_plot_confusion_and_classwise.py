import os, argparse, numpy as np, matplotlib.pyplot as plt, seaborn as sns, json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ------------------------------
# Confusion Matrix Plotter
# ------------------------------
def plot_confmat(y_true, y_pred, labels, out_dir, normalize=True):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        cbar=False,
        square=True,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    save_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved confusion matrix to {save_path}")

    # Also save as npy
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)
    return cm


# ------------------------------
# Classwise Metrics
# ------------------------------
def save_classwise(y_true, y_pred, labels, out_dir):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    json_path = os.path.join(out_dir, "classwise_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"📊 Class-wise metrics saved to {json_path}")

    # Markdown version (for README)
    import pandas as pd
    df = pd.DataFrame(report).T
    md_path = os.path.join(out_dir, "classwise_metrics.md")
    df.to_markdown(md_path)
    print(f"🧾 Markdown version saved: {md_path}")

    # Return summary for accuracy/macro F1
    acc = report.get("accuracy", 0.0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    return acc, macro_f1


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="e.g., experiments/exp01_resnet")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    exp_dir = args.exp_dir
    figs_dir = os.path.join(exp_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Locate probs file
    npz_path = None
    for f in ("probs_test.npz", "probs_val.npz", "probs.npz"):
        p = os.path.join(exp_dir, f)
        if os.path.exists(p):
            npz_path = p
            break
    assert npz_path, f"No probability file found in {exp_dir}"

    # Load predictions
    data = np.load(npz_path, allow_pickle=True)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    labels = list(data["classes"])
    y_pred = np.argmax(y_prob, axis=1)

    print(f"🧩 Loaded {len(y_true)} samples from {npz_path}")
    print(f"📈 Labels: {labels}")

    # Plot Confusion Matrix
    plot_confmat(y_true, y_pred, labels, figs_dir, normalize=True)

    # Save Classwise Metrics
    acc, macro_f1 = save_classwise(y_true, y_pred, labels, exp_dir)

    # Save simple evaluation summary for update_readme
    summary = {
        "model": os.path.basename(exp_dir),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }
    with open(os.path.join(exp_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary JSON: {exp_dir}/evaluation_summary.json")


if __name__ == "__main__":
    main()
