"""Dataset Report -- auto-generated HTML report for each ingestion run.

Includes:
  - Basic statistics (total frames, sources)
  - Quality filter breakdown
  - Resolution distribution
  - Scene distribution (if scene analysis was run)
  - Source (camera) distribution
  - Sample preview (first N frames)
  - Occlusion statistics (future)
"""

from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Optional

from ..core import IngestionReport


def _esc(val) -> str:
    """HTML-escape any value before inserting into report HTML (M-12 XSS fix)."""
    return _html.escape(str(val))


class ReportGenerator:
    """Generate HTML data quality report from IngestionReport."""

    def generate(self, report: IngestionReport, output_path: str | Path) -> None:
        """Write self-contained HTML report."""
        html = self._build_html(report)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _build_html(self, r: IngestionReport) -> str:
        bars = self._bar(r.total_sampled, r.total_sampled, "#4A90D9")
        quality_bars = self._bar(
            r.total_kept_after_quality,
            r.total_sampled,
            "#27AE60",
        )

        def pct(val: int, total: int) -> str:
            if total == 0:
                return "0%"
            return f"{val / total * 100:.1f}%"

        # Build scene rows
        scene_rows = ""
        if r.scene_distribution:
            max_scene = max(r.scene_distribution.values()) if r.scene_distribution else 1
            for label, count in sorted(r.scene_distribution.items(), key=lambda x: -x[1]):
                scene_rows += f"""
                <tr>
                    <td>{_esc(label)}</td>
                    <td>{count}</td>
                    <td>{pct(count, r.total_after_dedup)}</td>
                    <td><div class="bar" style="width:{count/max_scene*100:.1f}%"></div></td>
                </tr>"""
        else:
            scene_rows = '<tr><td colspan="4" style="color:#999">Scene analysis not run (requires torch)</td></tr>'

        # Build source rows
        source_rows = ""
        if r.source_distribution:
            max_src = max(r.source_distribution.values()) if r.source_distribution else 1
            for name, count in sorted(r.source_distribution.items(), key=lambda x: -x[1]):
                source_rows += f"""
                <tr>
                    <td>{_esc(name)}</td>
                    <td>{count}</td>
                    <td>{pct(count, r.total_sampled)}</td>
                    <td><div class="bar" style="width:{(count / max(max_src, 1)) * 100:.1f}%"></div></td>
                </tr>"""

        # Build resolution rows
        res_rows = ""
        if r.resolution_distribution:
            max_res = max(r.resolution_distribution.values()) if r.resolution_distribution else 1
            for res, count in sorted(r.resolution_distribution.items(), key=lambda x: -x[1]):
                res_rows += f"""
                <tr>
                    <td>{_esc(res)}</td>
                    <td>{count}</td>
                    <td>{pct(count, r.total_after_dedup)}</td>
                    <td><div class="bar" style="width:{(count / max(max_res, 1)) * 100:.1f}%"></div></td>
                </tr>"""

        # Build sample preview
        preview_rows = ""
        for i, sample_path in enumerate(r.samples[:30]):
            rel = sample_path.replace("\\", "/").split("/")[-1]
            preview_rows += f"""
                <tr><td>{i + 1}</td><td style="font-size:12px;max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(sample_path)}</td></tr>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Ingestion Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f6fa; color: #2c3e50; padding: 40px; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin-bottom: 8px; }}
  .subtitle {{ color: #666; margin-bottom: 32px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat-card .value {{ font-size: 28px; font-weight: 700; }}
  .stat-card .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
  .stat-card .sub {{ font-size: 12px; color: #999; margin-top: 2px; }}
  .card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .card h2 {{ font-size: 16px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #eee; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; font-size: 12px; color: #666; text-transform: uppercase; padding: 8px 4px; border-bottom: 2px solid #eee; }}
  td {{ padding: 8px 4px; border-bottom: 1px solid #f0f0f0; font-size: 14px; }}
  .bar {{ height: 8px; background: #4A90D9; border-radius: 4px; min-width: 4px; }}
  .bar-green {{ background: #27AE60; }}
  .bar-orange {{ background: #E67E22; }}
  .bar-red {{ background: #E74C3C; }}
  .quality-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }}
  .quality-item {{ padding: 12px; border-radius: 6px; text-align: center; }}
  .quality-item .count {{ font-size: 20px; font-weight: 700; }}
  .quality-item .desc {{ font-size: 12px; }}
  .bg-green {{ background: #eafaf1; }} .bg-orange {{ background: #fef5e7; }} .bg-red {{ background: #fdedec; }}
  .text-green {{ color: #27AE60; }} .text-orange {{ color: #E67E22; }} .text-red {{ color: #E74C3C; }}
  .sample-list {{ max-height: 300px; overflow-y: auto; }}
</style>
</head>
<body>
<div class="container">
  <h1>Dataset Ingestion Report</h1>
  <p class="subtitle">Generated automatically by Raven Trainer Ingestion Pipeline</p>

  <!-- Basic Stats -->
  <div class="stats-grid">
    <div class="stat-card"><div class="value">{r.total_sources}</div><div class="label">Raw Sources</div></div>
    <div class="stat-card"><div class="value">{r.total_raw_frames:,}</div><div class="label">Raw Frames</div></div>
    <div class="stat-card"><div class="value">{r.total_sampled:,}</div><div class="label">After Sampling</div></div>
    <div class="stat-card"><div class="value">{r.total_kept_after_quality:,}</div><div class="label">After Quality</div></div>
    <div class="stat-card"><div class="value">{r.total_after_dedup:,}</div><div class="label">After Dedup</div></div>
    <div class="stat-card"><div class="value">{pct(r.total_after_dedup, r.total_sampled)}</div><div class="label">Keep Rate</div></div>
  </div>

  <!-- Quality Breakdown -->
  <div class="card">
    <h2>Quality Filter Breakdown</h2>
    <div class="quality-grid">
      <div class="quality-item bg-green">
        <div class="count">{r.total_kept_after_quality}</div>
        <div class="desc">Kept</div>
      </div>
      <div class="quality-item bg-orange">
        <div class="count text-orange">{r.quality_removed_blurry}</div>
        <div class="desc">Blurry</div>
      </div>
      <div class="quality-item bg-red">
        <div class="count text-red">{r.quality_removed_black}</div>
        <div class="desc">Black Frame</div>
      </div>
      <div class="quality-item bg-red">
        <div class="count text-red">{r.quality_removed_overexposed}</div>
        <div class="desc">Overexposed</div>
      </div>
      <div class="quality-item bg-orange">
        <div class="count text-orange">{r.quality_removed_small}</div>
        <div class="desc">Too Small</div>
      </div>
      <div class="quality-item bg-red">
        <div class="count text-red">{r.dedup_removed}</div>
        <div class="desc">Duplicates</div>
      </div>
    </div>
  </div>

  <!-- Source Distribution -->
  <div class="card">
    <h2>Source (Camera) Distribution</h2>
    <table>
      <tr><th>Source</th><th>Count</th><th>Ratio</th><th></th></tr>
      {source_rows}
    </table>
  </div>

  <!-- Scene Distribution -->
  <div class="card">
    <h2>Scene Distribution</h2>
    <table>
      <tr><th>Scene</th><th>Count</th><th>Ratio</th><th></th></tr>
      {scene_rows}
    </table>
  </div>

  <!-- Resolution Distribution -->
  <div class="card">
    <h2>Resolution Distribution</h2>
    <table>
      <tr><th>Resolution</th><th>Count</th><th>Ratio</th><th></th></tr>
      {res_rows}
    </table>
  </div>

  <!-- Sample Preview -->
  <div class="card">
    <h2>Sample Preview (first {min(30, len(r.samples))} of {len(r.samples)})</h2>
    <div class="sample-list">
      <table>
        <tr><th>#</th><th>Path</th></tr>
        {preview_rows}
      </table>
    </div>
  </div>
</div>
</body>
</html>"""
        return html

    @staticmethod
    def _bar(value: int, total: int, color: str) -> str:
        pct = (value / total * 100) if total > 0 else 0
        return f'<div class="bar" style="width:{pct:.1f}%;background:{color}"></div>'

