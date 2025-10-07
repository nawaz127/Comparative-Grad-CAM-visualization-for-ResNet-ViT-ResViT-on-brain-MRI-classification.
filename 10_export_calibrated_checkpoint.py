import os, json, argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="experiments/expXX_*")
    ap.add_argument("--ckpt", default=None, help="path to original checkpoint (default: <exp_dir>/best.pt)")
    ap.add_argument("--calib_json", default=None, help="calibration_report_*.json with 'temperature'")
    ap.add_argument("--out", default=None, help="output path (default: <exp_dir>/best_temp_scaled.pt)")
    args = ap.parse_args()

    if args.ckpt is None:
        args.ckpt = os.path.join(args.exp_dir, "best.pt")
    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"

    # --- Load temperature
    T = None
    if args.calib_json is None:
        for split in ["val", "test"]:
            cand = os.path.join(args.exp_dir, f"calibration_report_{split}.json")
            if os.path.exists(cand):
                args.calib_json = cand
                break
    if args.calib_json and os.path.exists(args.calib_json):
        with open(args.calib_json, "r", encoding="utf-8") as f:
            T = json.load(f).get("temperature", None)
    if T is None:
        raise SystemExit("❌ No temperature found. Run 9_temp_scaling_and_calibration_metrics.py first.")

    # --- Export scaled checkpoint
    obj = torch.load(args.ckpt, map_location="cpu")
    payload = {"state_dict": obj, "temperature": float(T)}
    out = args.out or os.path.join(args.exp_dir, "best_temp_scaled.pt")
    torch.save(payload, out)
    print(f"✅ Wrote: {out} with temperature T={T:.3f}")

if __name__ == "__main__":
    main()
