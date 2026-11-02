"""Label Studio annotation converter — LS JSON ↔ YOLO TXT format conversion.

Converts Label Studio export JSON to YOLO TXT labels and vice versa.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class LabelStudioConverter:
    """Label Studio annotation format converter."""

    def __init__(self, label_map: Optional[Dict[int, str]] = None):
        self.label_map = label_map or {}

    def ls_to_yolo(
        self,
        ls_json: Path,
        image_dir: Path,
        output_label_dir: Path,
    ) -> List[Path]:
        """Convert Label Studio export JSON to YOLO TXT labels.

        Label Studio export format (JSON array):
        [
            {
                "id": 1,
                "annotations": [{
                    "result": [{
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": ["person"],
                            "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4
                        }
                    }]
                }],
                "file_upload": "frame_0001.jpg"
            }
        ]

        Note: x, y, width, height in Label Studio are percentages (0-100).
        """
        output_label_dir.mkdir(parents=True, exist_ok=True)
        with open(ls_json, encoding="utf-8") as f:
            tasks = json.load(f)

        # Build reverse map: label name → class id
        name_to_id = {v: k for k, v in self.label_map.items()}

        processed_images: List[Path] = []

        for task in tasks:
            filename = task.get("file_upload", "")
            if not filename:
                continue

            img_path = image_dir / filename
            if not img_path.exists():
                continue

            annotations = task.get("annotations", [])
            if not annotations:
                continue

            results = annotations[0].get("result", [])
            label_path = output_label_dir / f"{img_path.stem}.txt"

            with open(label_path, "w", encoding="utf-8") as f:
                for r in results:
                    if r.get("type") != "rectanglelabels":
                        continue
                    value = r.get("value", {})
                    labels = value.get("rectanglelabels", [])
                    if not labels:
                        continue

                    cls_id = name_to_id.get(labels[0])
                    if cls_id is None:
                        continue

                    # Label Studio: x, y, width, height are percentages (0-100)
                    x_pct = value.get("x", 0)
                    y_pct = value.get("y", 0)
                    w_pct = value.get("width", 0)
                    h_pct = value.get("height", 0)

                    # Convert to YOLO normalized (0-1) format: center_x, center_y, width, height
                    xc = (x_pct + w_pct / 2) / 100
                    yc = (y_pct + h_pct / 2) / 100
                    bw = w_pct / 100
                    bh = h_pct / 100

                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            processed_images.append(img_path)

        return processed_images

    def yolo_to_ls(
        self,
        image_paths: List[Path],
        label_dir: Path,
        output_json: Path,
    ) -> None:
        """Convert YOLO TXT labels to Label Studio export JSON format."""
        tasks = []

        for img_path in image_paths:
            label_file = label_dir / f"{img_path.stem}.txt"
            if not label_file.exists():
                continue

            results = []
            with open(label_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])

                    label_name = self.label_map.get(cls_id, str(cls_id))

                    # YOLO normalized → Label Studio percentage (0-100)
                    x_pct = (xc - bw / 2) * 100
                    y_pct = (yc - bh / 2) * 100
                    w_pct = bw * 100
                    h_pct = bh * 100

                    results.append({
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": [label_name],
                            "x": x_pct,
                            "y": y_pct,
                            "width": w_pct,
                            "height": h_pct,
                        },
                    })

            if results:
                tasks.append({
                    "file_upload": img_path.name,
                    "annotations": [{"result": results}],
                })

        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
