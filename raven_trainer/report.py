import html as _html
import json
from pathlib import Path

# This script will be called after training to generate a training report
# It will be integrated into train.py

TRAIN_REPORT_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>Training Report</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{font-family:-apple-system,sans-serif;background:#f5f6fa;color:#2c3e50;padding:40px;}}
  .container{{max-width:960px;margin:0 auto;}}
  h1{{font-size:24px;}}
  .stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin:20px 0;}}
  .stat-card{{background:white;border-radius:8px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
  .stat-card .value{{font-size:28px;font-weight:700;}}
  .stat-card .label{{font-size:13px;color:#666;}}
  .card{{background:white;border-radius:8px;padding:24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
  .card h2{{font-size:16px;margin-bottom:12px;border-bottom:1px solid #eee;padding-bottom:8px;}}
  table{{width:100%;border-collapse:collapse;}}
  th{{text-align:left;font-size:12px;color:#666;padding:6px 4px;border-bottom:2px solid #eee;}}
  td{{padding:6px 4px;border-bottom:1px solid #f0f0f0;font-size:14px;}}
</style></head>
<body><div class="container">
<h1>{title}</h1>
<p style="color:#666;margin-bottom:20px">{subtitle}</p>
{body}
</div></body></html>
"""

def _esc(val) -> str:
    """HTML-escape any value before inserting into report HTML (M-11 XSS fix)."""
    return _html.escape(str(val))


def generate_training_report(exp_path, project_name, config, metrics, model_path, onnx_path, output_path):
    """Generate an HTML training report after training completes."""
    exp_name = Path(exp_path).stem

    metrics_html = ""
    if metrics:
        metrics_html = '<div class="stats-grid">'
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float)):
                metrics_html += '<div class="stat-card"><div class="value">{v:.4f}</div><div class="label">{_esc(k)}</div></div>'
        metrics_html += "</div>"

    config_html = '<div class="card"><h2>Training Configuration</h2><table>'
    for k, v in sorted(config.items()):
        if k == "dataset":
            config_html += f'<tr><td>dataset.name</td><td>{_esc(v.get("name", ""))}</td></tr>'
            config_html += f'<tr><td>dataset.val</td><td>{_esc(v.get("val", ""))}</td></tr>'
        elif isinstance(v, dict):
            config_html += f'<tr><td>{_esc(k)}</td><td>{_esc(json.dumps(v, ensure_ascii=False))}</td></tr>'
        else:
            config_html += f'<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>'
    config_html += "</table></div>"

    artifacts_html = '<div class="card"><h2>Artifacts</h2><table>'
    artifacts_html += f'<tr><td>Experiment</td><td>{_esc(exp_path)}</td></tr>'
    artifacts_html += f'<tr><td>Model (best.pt -> models/)</td><td>{_esc(model_path or "N/A")}</td></tr>'
    artifacts_html += f'<tr><td>ONNX</td><td>{_esc(onnx_path or "N/A")}</td></tr>'
    artifacts_html += "</table></div>"

    body = metrics_html + config_html + artifacts_html
    html = TRAIN_REPORT_HTML.format(
        title=_esc(f"Training Report - {project_name}"),
        subtitle=_esc(f"{exp_name} | generated automatically"),
        body=body,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def generate_eval_report(eval_path, project_name, metrics, model_used, dataset_used, output_path):
    """Generate an HTML evaluation report."""
    metrics_html = '<div class="stats-grid">'
    for k, v in sorted(metrics.items()):
        if isinstance(v, (int, float)):
            metrics_html += '<div class="stat-card"><div class="value">{v:.4f}</div><div class="label">{_esc(k)}</div></div>'
    metrics_html += "</div>"

    info_html = f'<div class="card"><h2>Evaluation Info</h2><table><tr><td>Model</td><td>{_esc(model_used)}</td></tr><tr><td>Dataset</td><td>{_esc(dataset_used)}</td></tr></table></div>'

    body = metrics_html + info_html
    html = TRAIN_REPORT_HTML.format(
        title=_esc(f"Evaluation Report - {project_name}"),
        subtitle=_esc("generated automatically"),
        body=body,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def generate_export_report(project_name, model_name, format_type, output_path, metadata, out_file):
    """Generate an HTML export report."""
    info = '<div class="card"><h2>Export Info</h2><table>'
    info += f'<tr><td>Model</td><td>{_esc(model_name)}</td></tr>'
    info += f'<tr><td>Format</td><td>{_esc(format_type)}</td></tr>'
    info += f'<tr><td>Output</td><td>{_esc(out_file)}</td></tr>'
    info += f'<tr><td>SHA256</td><td>{_esc(metadata.get("sha256", "N/A"))}</td></tr>'
    info += '</table></div>'

    body = info
    html = TRAIN_REPORT_HTML.format(
        title=_esc(f"Export Report - {project_name}"),
        subtitle=_esc("generated automatically"),
        body=body,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
