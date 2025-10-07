import os, argparse, json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import random

# ------------------------------
# Plot Calibration Curve
# ------------------------------
def plot_calibration_curve(y_true, y_prob, n_bins=10, out_dir="figs", prefix="calibration_overall"):
    os.makedirs(out_dir, exist_ok=True)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}.png"), dpi=250)
    plt.close()
    print(f"✅ Saved calibration curve: {out_dir}/{prefix}.png")


# ------------------------------
# Compute Per-class Thresholds
# ------------------------------
def compute_best_thresholds(y_true, y_prob, classes):
    results = {"per_class": {}}
    macro_scores = []

    for i, cls in enumerate(classes):
        best_t, best_f1 = 0.5, 0
        for t in np.linspace(0.1, 0.9, 81):
            preds = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score((y_true == i).astype(int), preds)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        results["per_class"][cls] = {
            "best_threshold": float(best_t),
            "f1_at_best": float(best_f1),
        }
        macro_scores.append(best_f1)

    results["macro_f1_default"] = float(np.mean(macro_scores))
    print("✅ Computed per-class thresholds")
    return results


# ------------------------------
# Compute Calibration Metrics
# ------------------------------
def compute_calibration_metrics(y_true, y_prob, classes):
    metrics = {}
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))

    def expected_calibration_error(y_true_bin, y_prob, n_bins=15):
        ece = 0.0
        for i in range(len(classes)):
            prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_prob[:, i], n_bins=n_bins)
            if len(prob_true) > 0 and len(prob_pred) > 0:
                ece += np.mean(np.abs(prob_true - prob_pred))
        return ece / len(classes)

    brier = np.mean(
        [brier_score_loss(y_true_bin[:, i], y_prob[:, i]) for i in range(len(classes))]
    )
    ece = expected_calibration_error(y_true_bin, y_prob)

    metrics["ECE"] = float(ece)
    metrics["Brier"] = float(brier)
    print(f"📏 ECE={ece:.4f}, Brier={brier:.4f}")
    return metrics


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="e.g., experiments/exp01_resnet")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--num_samples", type=int, default=10, help="Number of random samples to use")
    args = ap.parse_args()

    exp_dir = args.exp_dir
    figs_dir = os.path.join(exp_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Load predictions
    npz_path = None
    for name in ("probs_test.npz", "probs_val.npz", "probs.npz"):
        path = os.path.join(exp_dir, name)
        if os.path.exists(path):
            npz_path = path
            break
    assert npz_path, f"No probs file found in {exp_dir}"
    data = np.load(npz_path, allow_pickle=True)
    y_true, y_prob, classes = data["y_true"], data["y_prob"], list(data["classes"])

    # --- Limit to N random samples ---
    subset = min(args.num_samples, len(y_true))
    idx = random.sample(range(len(y_true)), subset)
    y_true = y_true[idx]
    y_prob = y_prob[idx]
    print(f"⚡ Using only {subset} random samples for calibration & threshold visualization")

    print(f"🧩 Loaded {len(y_true)} samples from {npz_path}")
    print(f"📈 Classes: {classes}")

    # --- Per-class calibration (binary)
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        plot_calibration_curve(y_bin, y_prob[:, i], n_bins=10, out_dir=figs_dir, prefix=f"calibration_{cls}")

    # --- Macro-average calibration
    from sklearn.calibration import calibration_curve

    y_true_bin = np.eye(len(classes))[y_true]
    valid_curves = []

    for i in range(len(classes)):
        prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_prob[:, i], n_bins=10)
        if len(prob_pred) > 1 and len(prob_true) > 1:
            valid_curves.append((prob_pred, prob_true))

    mean_pred = np.linspace(0, 1, 20)
    interp_true = []

    for prob_pred, prob_true in valid_curves:
        interp_true.append(np.interp(mean_pred, prob_pred, prob_true))

    if interp_true:
        mean_true_interp = np.mean(interp_true, axis=0)
    else:
        mean_true_interp = mean_pred

    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, mean_true_interp, "s-", label="Macro-avg model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Curve (Macro Average)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "calibration_overall.png"), dpi=250)
    plt.close()
    print("✅ Saved macro-average calibration curve")

    # --- Metrics
    cal_metrics = compute_calibration_metrics(y_true, y_prob, classes)
    with open(os.path.join(exp_dir, "calibration_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(cal_metrics, f, indent=2)

    # --- Thresholds
    thresholds = compute_best_thresholds(y_true, y_prob, classes)
    with open(os.path.join(exp_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(f"✅ Saved all calibration & threshold outputs under {exp_dir}")


if __name__ == "__main__":
    main()
