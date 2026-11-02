"""CVAT annotation converter — CVAT XML ↔ YOLO TXT format conversion.

Converts between CVAT XML 1.1 (Ultralytics YOLO Detection format)
and YOLO TXT labels for training.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _read_image_dimensions(img_path: Path) -> Tuple[int, int]:
    """Read the real (width, height) of an image file.

    Tries PIL first, falls back to cv2, then to structural probe via imghdr.
    Returns (0, 0) if the dimensions cannot be determined.
    """
    # Try PIL (Pillow)
    try:
        from PIL import Image as PILImage
        with PILImage.open(img_path) as img:
            return img.size  # (width, height)
    except Exception:
        pass

    # Try cv2 (OpenCV)
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            return (w, h)
    except Exception:
        pass

    logger.warning("Could not read dimensions for %s, using defaults", img_path)
    return (0, 0)


class CvatExporter:
    """Export images + labels from a YOLO-format dataset to CVAT-compatible XML.

    Generates CVAT XML 1.1 with per-image real dimensions, so that
    YOLO normalized coordinates are converted to correct absolute pixels
    regardless of image resolution.
    """

    def __init__(self, label_map: Optional[Dict[int, str]] = None):
        self.label_map = label_map or {}

    def yolo_to_cvat_xml(
        self,
        image_paths: List[Path],
        label_dir: Path,
        output_xml: Path,
        image_width: int = 1920,
        image_height: int = 1080,
        auto_detect_size: bool = True,
    ) -> None:
        """Convert YOLO-format labels to CVAT XML 1.1.

        Args:
            image_paths: list of image file paths.
            label_dir: directory containing YOLO .txt files (same stem as images).
            output_xml: output CVAT XML path.
            image_width: fallback width when auto-detection is disabled or fails.
            image_height: fallback height when auto-detection is disabled or fails.
            auto_detect_size: if True (default), read each image's real
                dimensions so that coordinate conversion is accurate even
                when images have mixed resolutions.  Set to False to use
                the fixed *image_width* / *image_height* for every image
                (legacy behaviour, not recommended).
        """
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"

        for img_idx, img_path in enumerate(image_paths):
            # Determine real image dimensions
            if auto_detect_size:
                w, h = _read_image_dimensions(Path(img_path))
                if w == 0 or h == 0:
                    w, h = image_width, image_height
            else:
                w, h = image_width, image_height

            # Image entry
            img_elem = ET.SubElement(root, "image")
            img_elem.set("id", str(img_idx))
            img_elem.set("name", str(img_path))
            img_elem.set("width", str(w))
            img_elem.set("height", str(h))

            # Corresponding label file
            label_file = label_dir / f"{Path(img_path).stem}.txt"
            if not label_file.exists():
                continue

            with open(label_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])

                    # YOLO normalized → CVAT absolute (per-image real dimensions)
                    abs_x = (xc - bw / 2) * w
                    abs_y = (yc - bh / 2) * h
                    abs_w = bw * w
                    abs_h = bh * h

                    label = self.label_map.get(cls_id, str(cls_id))
                    box = ET.SubElement(img_elem, "box")
                    box.set("label", label)
                    box.set("source", "manual")
                    box.set("occluded", "0")
                    box.set("xtl", f"{abs_x:.1f}")
                    box.set("ytl", f"{abs_y:.1f}")
                    box.set("xbr", f"{abs_x + abs_w:.1f}")
                    box.set("ybr", f"{abs_y + abs_h:.1f}")
                    box.set("z_order", "0")

        # Ensure output directory exists
        output_xml = Path(output_xml)
        output_xml.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(root)
        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        logger.info("Exported %d images to CVAT XML: %s", len(image_paths), output_xml)


class CvatImporter:
    """Import CVAT-annotated data back to YOLO format.

    Reads CVAT XML 1.1 and generates YOLO TXT labels + image list.
    """

    def __init__(self, label_map: Optional[Dict[str, int]] = None):
        self.label_map = label_map or {}

    def cvat_xml_to_yolo(
        self,
        cvat_xml: Path,
        image_dir: Path,
        output_label_dir: Path,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> List[Path]:
        """Convert CVAT XML to YOLO TXT labels.

        Args:
            cvat_xml: CVAT XML 1.1 file
            image_dir: directory containing the images referenced in XML
            output_label_dir: output directory for YOLO .txt files

        Returns:
            List of image paths that have annotations
        """
        output_label_dir.mkdir(parents=True, exist_ok=True)
        tree = ET.parse(cvat_xml)
        root = tree.getroot()

        processed_images: List[Path] = []

        # Build reverse label map: name → id (M-24: moved outside loop)
        label_to_id = {v: k for k, v in self.label_map.items()}

        for img_elem in root.findall("image"):
            img_name = img_elem.get("name", "")
            w = int(img_elem.get("width", str(image_width)))
            h = int(img_elem.get("height", str(image_height)))

            img_path = image_dir / img_name
            if not img_path.exists():
                continue

            label_path = output_label_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for box in img_elem.findall("box"):
                    label_name = box.get("label", "")
                    cls_id = label_to_id.get(label_name)
                    if cls_id is None:
                        continue
                    xtl = float(box.get("xtl", "0"))
                    ytl = float(box.get("ytl", "0"))
                    xbr = float(box.get("xbr", "0"))
                    ybr = float(box.get("ybr", "0"))

                    # Convert to YOLO normalized
                    xc = ((xtl + xbr) / 2) / w
                    yc = ((ytl + ybr) / 2) / h
                    bw = (xbr - xtl) / w
                    bh = (ybr - ytl) / h

                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            processed_images.append(img_path)

        return processed_images
