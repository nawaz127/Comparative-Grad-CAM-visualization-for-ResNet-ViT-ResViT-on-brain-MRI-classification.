import os, argparse, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

# ------------------------------
# Plot Confusion Matrix
# ------------------------------
def plot_confusion_matrix(y_true, y_pred, labels, out_dir, normalize=True):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=labels, yticklabels=labels, cmap="Blues",
                cbar=False, square=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix{" (Normalized)" if normalize else ""}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # Save raw matrix
    np.save(os.path.join(out_dir, 'confusion_matrix.npy'), cm)
    print(f"✅ Confusion matrix saved to {out_dir}/confusion_matrix.png")

# ------------------------------
# Generate Report (per-class F1)
# ------------------------------
def save_classwise_metrics(y_true, y_pred, labels, out_dir):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    json_path = os.path.join(out_dir, "classwise_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"📊 Saved class-wise metrics to {json_path}")

    # Markdown table (for README)
    import pandas as pd
    df = pd.DataFrame(report).T
    md_path = os.path.join(out_dir, "classwise_metrics.md")
    df.to_markdown(md_path)
    print(f"🧾 Markdown version saved: {md_path}")


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="e.g., experiments/exp01_resnet")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    exp_dir = args.exp_dir
    figs_dir = os.path.join(exp_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Load .npz
    npz_path = None
    for f in ("probs_test.npz", "probs_val.npz", "probs.npz"):
        p = os.path.join(exp_dir, f)
        if os.path.exists(p):
            npz_path = p
            break
    assert npz_path, f"No probability file found in {exp_dir}"

    data = np.load(npz_path, allow_pickle=True)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    labels = list(data["classes"])
    y_pred = np.argmax(y_prob, axis=1)

    print(f"🧩 Loaded {len(y_true)} samples from {npz_path}")
    print(f"📈 Labels: {labels}")

    # Plot Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, labels, figs_dir, normalize=True)

    # Save class-wise report
    save_classwise_metrics(y_true, y_pred, labels, exp_dir)


if __name__ == "__main__":
    main()
