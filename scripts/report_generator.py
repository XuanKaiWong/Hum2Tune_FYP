"""
REPORT_GENERATOR.PY — Automated Evaluation Report Builder
FYP Hum2Tune: Melody-Based Music Recognition

Generates a self-contained HTML evaluation report from saved results,
including confusion matrix, per-class F1 bars, training curves,
and a model vs. DTW baseline comparison table.

Usage:
    from scripts.report_generator import generate_report
    generate_report()

    # or via CLI:
    python scripts/report_generator.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
RESULTS_DIR = project_root / "results" / "evaluations"
VIZ_DIR = project_root / "results" / "visualizations"
REPORT_DIR = project_root / "results"


# ─── HTML Template Helpers ────────────────────────────────────────────────────

def _badge(value: str, color: str) -> str:
    """Return an inline HTML badge span."""
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;font-size:0.85em;">{value}</span>'
    )


def _metric_card(label: str, value: str, color: str = "#2E75B6") -> str:
    """Return an HTML metric card div."""
    return f"""
    <div style="background:{color};color:white;padding:16px 24px;
                border-radius:8px;text-align:center;min-width:150px;">
      <div style="font-size:2em;font-weight:bold;">{value}</div>
      <div style="font-size:0.9em;opacity:0.9;">{label}</div>
    </div>"""


def _img_tag(path: Path, alt: str, width: str = "100%") -> str:
    """Return an HTML img tag if the file exists, else a placeholder."""
    if path.exists():
        import base64
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = path.suffix.lstrip(".")
        return (
            f'<img src="data:image/{ext};base64,{b64}" '
            f'alt="{alt}" style="width:{width};border-radius:6px;" />'
        )
    return f'<p style="color:#999;">[{alt} — image not found at {path}]</p>'


def _per_class_table(report_dict: dict, class_names: list) -> str:
    """Build an HTML table for per-class metrics."""
    rows = ""
    for name in class_names:
        if name not in report_dict:
            continue
        m = report_dict[name]
        f1 = m["f1-score"]
        color = ("#27AE60" if f1 >= 0.7 else
                 "#F39C12" if f1 >= 0.4 else "#E74C3C")
        bar = (
            f'<div style="background:#eee;border-radius:3px;height:12px;">'
            f'<div style="background:{color};width:{f1*100:.0f}%;'
            f'height:12px;border-radius:3px;"></div></div>'
        )
        rows += f"""
        <tr>
          <td>{name}</td>
          <td>{m['precision']:.3f}</td>
          <td>{m['recall']:.3f}</td>
          <td>{bar} {f1:.3f}</td>
          <td>{int(m['support'])}</td>
        </tr>"""
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.95em;">
      <thead>
        <tr style="background:#1F3864;color:white;">
          <th style="padding:8px 12px;text-align:left;">Class</th>
          <th style="padding:8px 12px;">Precision</th>
          <th style="padding:8px 12px;">Recall</th>
          <th style="padding:8px 12px;">F1 Score</th>
          <th style="padding:8px 12px;">Support</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def generate_report() -> Path:
    """
    Generate a self-contained HTML evaluation report.

    Reads evaluation_results.json and training_summary.json from
    results/evaluations/, embeds visualisation images as base64,
    and writes a single-file HTML report to results/evaluation_report.html.

    Returns:
        Path to the generated HTML report file.

    Raises:
        FileNotFoundError: If evaluation_results.json is missing (run
        evaluate.py first).
    """
    # ── Load Evaluation Results ───────────────────────────────────────────
    eval_path = RESULTS_DIR / "evaluation_results.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"evaluation_results.json not found at {eval_path}. "
            "Run 'python main.py evaluate' first."
        )

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # ── Load Training Summary (optional) ─────────────────────────────────
    train_summary = {}
    train_path = RESULTS_DIR / "training_summary.json"
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_summary = json.load(f)

    metrics = eval_data.get("metrics", {})
    class_map = eval_data.get("class_map", {})
    report_dict = eval_data.get("classification_report", {})
    class_names = list(class_map.values())
    num_classes = len(class_names)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Metric Cards ──────────────────────────────────────────────────────
    top1 = metrics.get("top_1_accuracy", metrics.get("accuracy", 0))
    top3 = metrics.get("top_3_accuracy", 0)
    top5 = metrics.get("top_5_accuracy", 0)
    macro_f1 = metrics.get("macro_f1", 0)

    cards = "".join([
        _metric_card("Top-1 Accuracy", f"{top1:.2%}", "#27AE60"),
        _metric_card("Top-3 Accuracy", f"{top3:.2%}", "#2E75B6"),
        _metric_card("Top-5 Accuracy", f"{top5:.2%}", "#8E44AD"),
        _metric_card("Macro F1", f"{macro_f1:.3f}", "#E67E22"),
        _metric_card("Classes", str(num_classes), "#1F3864"),
    ])

    # ── Training Info ─────────────────────────────────────────────────────
    train_html = ""
    if train_summary:
        epochs = train_summary.get("epochs_trained", "—")
        best_val = train_summary.get("best_val_accuracy", 0)
        seed = train_summary.get("seed", "—")
        train_html = f"""
        <div style="background:#f8f9fa;border-left:4px solid #2E75B6;
                    padding:12px 16px;margin:16px 0;border-radius:4px;">
          <strong>Training Summary:</strong>
          Epochs trained: {epochs} &nbsp;|&nbsp;
          Best val accuracy: {best_val:.2%} &nbsp;|&nbsp;
          Random seed: {seed} (reproducible)
        </div>"""

    # ── Confusion Matrix Section ──────────────────────────────────────────
    cm_img = _img_tag(VIZ_DIR / "confusion_matrix.png", "Confusion Matrix")

    # ── Per-Class F1 Chart ────────────────────────────────────────────────
    f1_img = _img_tag(VIZ_DIR / "per_class_f1.png", "Per-Class F1")

    # ── Training Curves ───────────────────────────────────────────────────
    curves_img = _img_tag(VIZ_DIR / "training_curves.png", "Training Curves")

    # ── Per-Class Table ───────────────────────────────────────────────────
    class_table = _per_class_table(report_dict, class_names)

    # ── Full HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Hum2Tune — Evaluation Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; color: #333; background: #f0f2f5; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    .header {{ background: linear-gradient(135deg,#1F3864,#2E75B6);
               color: white; padding: 32px; border-radius: 10px;
               margin-bottom: 24px; }}
    .header h1 {{ font-size: 2em; margin-bottom: 6px; }}
    .header p {{ opacity: 0.85; }}
    .card {{ background: white; border-radius: 10px; padding: 24px;
             margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
    .card h2 {{ font-size: 1.2em; color: #1F3864; margin-bottom: 16px;
                padding-bottom: 8px; border-bottom: 2px solid #EBF5FB; }}
    .metrics-row {{ display: flex; gap: 12px; flex-wrap: wrap;
                    margin-bottom: 8px; }}
    table th, table td {{ padding: 8px 12px; border-bottom: 1px solid #eee;
                          text-align: center; }}
    table tr:hover td {{ background: #f7fbff; }}
    .footer {{ text-align: center; color: #999; font-size: 0.85em;
               margin-top: 32px; padding: 16px; }}
  </style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>🎵 Hum2Tune — Evaluation Report</h1>
    <p>FYP: Melody-Based Music Recognition using Frequency Analysis</p>
    <p style="margin-top:8px;font-size:0.85em;">
      Model: CNN-LSTM &nbsp;|&nbsp; Generated: {timestamp}
    </p>
  </div>

  <div class="card">
    <h2>📊 Overall Performance</h2>
    <div class="metrics-row">{cards}</div>
    {train_html}
  </div>

  <div class="card">
    <h2>📈 Training Curves</h2>
    {curves_img}
  </div>

  <div class="card">
    <h2>🔥 Confusion Matrix</h2>
    {cm_img}
    <p style="color:#666;font-size:0.85em;margin-top:8px;">
      Left: raw counts. Right: row-normalised (per-class recall).
    </p>
  </div>

  <div class="card">
    <h2>🏅 Per-Class F1 Score</h2>
    {f1_img}
  </div>

  <div class="card">
    <h2>📋 Per-Class Breakdown</h2>
    {class_table}
  </div>

  <div class="footer">
    Hum2Tune FYP &nbsp;•&nbsp; Report generated {timestamp}
  </div>

</div>
</body>
</html>"""

    # ── Save ──────────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "evaluation_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"✅ Evaluation report saved to {report_path}")
    print(f"\n✅ Report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_report()