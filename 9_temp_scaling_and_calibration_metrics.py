import os, json, argparse, numpy as np, random
from sklearn.metrics import brier_score_loss, log_loss

# ------------------------------
# Expected Calibration Error
# ------------------------------
def ece_score(y_true, y_prob, n_bins=15):
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


# ------------------------------
# Temperature Scaling helpers
# ------------------------------
def temp_scale_logits(y_logits, T):
    z = y_logits / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(y_logits, y_true, grid=None):
    if grid is None:
        grid = np.concatenate([[0.5], np.linspace(0.75, 3.0, 23)])
    best_T, best_nll = None, float("inf")
    for T in grid:
        p = temp_scale_logits(y_logits, T)
        nll = log_loss(y_true, p, labels=list(range(p.shape[1])))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T, best_nll


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="experiments/expXX_*")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--num_samples", type=int, default=10, help="limit to random subset for demo")
    args = ap.parse_args()

    npz_path = os.path.join(args.exp_dir, f"probs_{args.split}.npz")
    if not os.path.exists(npz_path):
        npz_path = os.path.join(args.exp_dir, "probs_test.npz")
    assert os.path.exists(npz_path), f"Not found: {npz_path}"

    data = np.load(npz_path, allow_pickle=True)
    y_true, y_prob, y_logits, classes = (
        data["y_true"],
        data["y_prob"],
        data["y_logits"],
        list(data["classes"]),
    )

    # --- Fast subset for quick runs ---
    subset = min(args.num_samples, len(y_true))
    idx = random.sample(range(len(y_true)), subset)
    y_true, y_prob, y_logits = y_true[idx], y_prob[idx], y_logits[idx]
    print(f"⚡ Using only {subset} random samples for temperature scaling demo")

    # --- Pre-calibration metrics
    ece_before = ece_score(y_true, y_prob, n_bins=args.bins)
    briers = [brier_score_loss((y_true == i).astype(int), y_prob[:, i]) for i in range(len(classes))]
    brier_before = float(np.mean(briers))

    # --- Fit temperature
    T, _ = fit_temperature(y_logits, y_true)
    y_prob_cal = temp_scale_logits(y_logits, T)

    # --- Post-calibration metrics
    ece_after = ece_score(y_true, y_prob_cal, n_bins=args.bins)
    briers_after = [
        brier_score_loss((y_true == i).astype(int), y_prob_cal[:, i])
        for i in range(len(classes))
    ]
    brier_after = float(np.mean(briers_after))

    # --- Save outputs
    report = {
        "split": args.split,
        "classes": classes,
        "temperature": float(T),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "brier_before": brier_before,
        "brier_after": brier_after,
    }
    with open(os.path.join(args.exp_dir, f"calibration_report_{args.split}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    np.savez_compressed(
        os.path.join(args.exp_dir, f"probs_{args.split}_calibrated.npz"),
        y_true=y_true,
        y_prob=y_prob_cal,
        classes=np.array(classes, dtype=object),
    )
    print("✅ Saved calibration report and calibrated probabilities")
    print(f"   Temperature = {T:.3f} | ECE: {ece_before:.4f}→{ece_after:.4f} | Brier: {brier_before:.4f}→{brier_after:.4f}")


if __name__ == "__main__":
    main()
