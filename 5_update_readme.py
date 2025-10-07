import os, json, glob, pandas as pd, argparse, re

# ------------------------------
# Load summary rows from each experiment
# ------------------------------
def load_rows(exp_root="experiments"):
    rows = []
    for ej in glob.glob(os.path.join(exp_root, "*/evaluation.json")):
        with open(ej, "r", encoding="utf-8") as f:
            d = json.load(f)
        exp = os.path.basename(os.path.dirname(ej))
        report = d.get("classification_report", {})
        rows.append({
            "experiment": exp,
            "model": d.get("model"),
            "accuracy": report.get("accuracy", 0.0),
            "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
            "auc_ovr": d.get("auc_ovr", float("nan")),
        })
    return pd.DataFrame(rows)


# ------------------------------
# Markdown Table Generator
# ------------------------------
def to_markdown_table(df):
    if df.empty:
        return (
            "| experiment | model | accuracy | macro_f1 | auc_ovr |\n"
            "|---|---|---:|---:|---:|\n"
            "| _no results yet_ |  |  |  |  |"
        )

    df = df.copy()
    for c in ["accuracy", "macro_f1", "auc_ovr"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    lines = [
        "| experiment | model | accuracy | macro_f1 | auc_ovr |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        acc = f"{r.get('accuracy', 0.0):.4f}"
        f1 = f"{r.get('macro_f1', 0.0):.4f}"
        auc = f"{r.get('auc_ovr', 0.0):.4f}"
        lines.append(
            f"| {r.get('experiment','')} | {r.get('model','')} | {acc} | {f1} | {auc} |"
        )
    return "\n".join(lines)


# ------------------------------
# Add experiment-specific visual summaries
# ------------------------------
def gradcam_md(exp_dir):
    gradcam_dir = os.path.join(exp_dir, "gradcam")
    if not os.path.isdir(gradcam_dir):
        return ""
    imgs = sorted([f for f in os.listdir(gradcam_dir) if f.endswith(".png")])
    if not imgs:
        return ""
    lines = ["\n**Grad-CAM Visualizations:**"]
    for f in imgs:
        rel = os.path.join(exp_dir, "gradcam", f).replace("\\", "/")
        lines.append(f"\n- {f}\n\n  ![]({rel})")
    return "\n".join(lines)


def calibration_md(exp_dir):
    figs_dir = os.path.join(exp_dir, "figs")
    if not os.path.isdir(figs_dir):
        return ""
    items = sorted(
        [f for f in os.listdir(figs_dir) if f.startswith("calibration_") and f.endswith(".png")]
    )
    if not items:
        return ""
    lines = ["\n**Calibration Curves:**"]
    for f in items:
        rel = os.path.join(exp_dir, "figs", f).replace("\\", "/")
        lines.append(f"\n- {f}\n\n  ![]({rel})")
    return "\n".join(lines)


def thresholds_md(exp_dir):
    path = os.path.join(exp_dir, "thresholds.json")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    lines = ["\n**Optimal Thresholds (per class):**", ""]
    lines.append(f"- Default macro-F1 = {d.get('macro_f1_default', 0):.4f}")
    for cls, v in d.get("per_class", {}).items():
        lines.append(f"- {cls}: t={v['best_threshold']:.2f}, F1={v['f1_at_best']:.4f}")
    return "\n".join(lines)


# ------------------------------
# Inject Markdown Summary into README
# ------------------------------
def inject_readme(readme_path="README.md", table_md=""):
    with open(readme_path, "r", encoding="utf-8") as f:
        txt = f.read()

    start_marker = "<!-- MODEL_COMPARISON_START -->"
    end_marker = "<!-- MODEL_COMPARISON_END -->"

    exp_root = "experiments"
    fig_blocks = []

    for exp in sorted(os.listdir(exp_root)):
        exp_dir = os.path.join(exp_root, exp)
        if not os.path.isdir(exp_dir):
            continue

        # load per-experiment visuals
        gradcam = gradcam_md(exp_dir)
        cal = calibration_md(exp_dir)
        th = thresholds_md(exp_dir)

        section = f"\n### {exp}\n"
        if gradcam:
            section += gradcam + "\n"
        if cal:
            section += cal + "\n"
        if th:
            section += th + "\n"

        # Include summary.csv if available
        csv_path = os.path.join(exp_dir, "summary.csv")
        if os.path.exists(csv_path):
            section += f"\n📊 **Summary metrics:** [{os.path.basename(csv_path)}]({csv_path})\n"

        fig_blocks.append(section)

    figs_section = "\n".join(fig_blocks)
    block = f"{start_marker}\n\n## 🧪 Model Comparison (Test Set)\n\n{table_md}\n\n{figs_section}\n\n{end_marker}"

    if start_marker in txt and end_marker in txt:
        new_txt = re.sub(
            re.escape(start_marker) + r".*?" + re.escape(end_marker),
            block,
            txt,
            flags=re.S,
        )
    else:
        new_txt = txt.strip() + "\n\n" + block + "\n"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_txt)

    print("✅ README.md updated with latest experiment results.")


# ------------------------------
# Main entry
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="experiments")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    df = load_rows(args.exp_root)
    table = to_markdown_table(df)
    inject_readme(args.readme, table)


if __name__ == "__main__":
    main()
