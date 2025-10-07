import argparse, os, json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

# ------------------------------
# Utility plotting functions
# ------------------------------
def plot_pr(y_true, y_prob, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summaries = {}
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        p, r, _ = precision_recall_curve(y_bin, y_prob[:, i])
        ap = average_precision_score(y_bin, y_prob[:, i])
        summaries[cls] = {"average_precision": float(ap)}
        plt.figure()
        plt.step(r, p, where='post', lw=2)
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'PR — {cls} (AP={ap:.3f})')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, f'pr_{i}_{cls}.png'), dpi=200, bbox_inches='tight')
        plt.close()
    return summaries


def plot_roc(y_true, y_prob, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summaries = {}
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        summaries[cls] = {"roc_auc": float(roc_auc)}
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC — {cls}')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, f'roc_{i}_{cls}.png'), dpi=200, bbox_inches='tight')
        plt.close()
    return summaries


def find_npz(exp_dir, split):
    for name in (f'probs_{split}.npz','probs.npz','probs_val.npz','probs_test.npz'):
        p = os.path.join(exp_dir, name)
        if os.path.exists(p):
            return p
    return None


# ------------------------------
# Macro-average curves
# ------------------------------
def plot_macro_curves(y_true, y_prob, classes, out_dir):
    n_classes = len(classes)
    y_true_bin = np.eye(n_classes)[y_true]
    # --- Macro ROC ---
    plt.figure()
    all_auc = []
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        A = auc(fpr, tpr)
        all_auc.append(A)
        plt.plot(fpr, tpr, lw=1.2, label=f"{cls} ({A:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC — Macro")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    macro_auc = np.mean(all_auc)
    plt.savefig(os.path.join(out_dir, "roc_macro.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- Macro PR ---
    plt.figure()
    all_ap = []
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        p, r, _ = precision_recall_curve(y_bin, y_prob[:, i])
        ap = average_precision_score(y_bin, y_prob[:, i])
        all_ap.append(ap)
        plt.step(r, p, where="post", lw=1.2, label=f"{cls} ({ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR — Macro")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    macro_ap = np.mean(all_ap)
    plt.savefig(os.path.join(out_dir, "pr_macro.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return {"macro_auc": float(macro_auc), "macro_ap": float(macro_ap)}


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="experiments/exp01_resnet")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    npz_path = find_npz(args.exp_dir, args.split)
    assert npz_path is not None, f"Could not find probs file in {args.exp_dir}. Run 3_eval.py first."

    data = np.load(npz_path, allow_pickle=True)
    y_true, y_prob, classes = data["y_true"], data["y_prob"], list(data["classes"])
    figs = os.path.join(args.exp_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    print(f"📊 Plotting PR/ROC curves for {args.exp_dir} ({len(classes)} classes)")

    pr_summary = plot_pr(y_true, y_prob, classes, figs)
    roc_summary = plot_roc(y_true, y_prob, classes, figs)
    macro_summary = plot_macro_curves(y_true, y_prob, classes, figs)

    summary = {"classes": classes,
               "macro_summary": macro_summary,
               "per_class_pr": pr_summary,
               "per_class_roc": roc_summary}

    with open(os.path.join(figs, "pr_roc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ Saved all PR/ROC plots and summary JSON to", figs)


if __name__ == "__main__":
    main()
