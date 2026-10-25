"""Conservative local detection of candidate diagrams, figures and image regions.

This is a geometry-based prototype, not a trained semantic figure classifier.
Scores are heuristic ranking values; all detections require human verification.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..preprocessing.page import Prepped, preprocess


@dataclass(frozen=True)
class FigureDetector:
    """Find large two-dimensional ink structures after suppressing page rulings.

    Thresholds scale with page dimensions. Ordinary disconnected text lines and
    small margin marks are excluded. Handwriting touching drawings, large words,
    tables and complicated page layouts can cause false positives or misses.
    """

    min_width_fraction: float = 0.06
    min_height_fraction: float = 0.035
    min_area_fraction: float = 0.0025

    def __post_init__(self):
        for name in ("min_width_fraction", "min_height_fraction", "min_area_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between zero and one")

    def detect(self, source, page: int = 1) -> list[dict]:
        """Accept the same image sources as preprocess(), or an existing Prepped page.

        Output boxes use exclusive x1/y1 in the deskewed standard-width image,
        matching the page coordinate system used by the existing preview API.
        """
        if page < 1:
            raise ValueError("page must be one-based")
        pp = source if isinstance(source, Prepped) else preprocess(source)
        gray = pp.gray
        height, width = gray.shape
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = (gray < min(max(threshold, 90), 185)).astype(np.uint8)

        # Detect from gray: the OCR clean image can erase vertical figure strokes.
        # Remove only page-spanning lines here, preserving smaller axes/boxes.
        for kernel in ((max(2, int(width * 0.65)), 1), (1, max(2, int(height * 0.65)))):
            ruling = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
            ink[cv2.dilate(ruling, np.ones((3, 3), np.uint8)) > 0] = 0
        if pp.header_bottom is not None:
            ink[:min(height, pp.header_bottom + 10)] = 0
        ink[int(height * 0.962):] = 0  # Match the OCR layout's footer exclusion.

        # A small closing joins broken strokes without joining whole text rows.
        gap = max(2, int(round(min(height, width) * 0.003)))
        joined = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        candidates = []
        for index in range(1, count):
            x, y, w, h, area = map(int, stats[index])
            if w < max(24, width * self.min_width_fraction) or h < max(24, height * self.min_height_fraction):
                continue
            if w * h < width * height * self.min_area_fraction:
                continue
            aspect = w / h
            if not 0.12 <= aspect <= 10:
                continue
            component = (labels[y:y + h, x:x + w] == index).astype(np.uint8)
            # Text made of separated horizontal rows should not become a figure.
            row_support = float(np.mean(component.any(axis=1)))
            col_support = float(np.mean(component.any(axis=0)))
            if min(row_support, col_support) < 0.75:
                continue
            density = area / (w * h)
            score = min(0.95, 0.4 + 0.25 * min(h / max(1, height * 0.15), 1)
                        + 0.2 * min(density / 0.2, 1) + 0.1 * min(row_support, col_support))
            padding = max(2, gap)
            candidates.append({
                "page": page,
                "bbox": [max(0, x - padding), max(0, y - padding),
                         min(width, x + w + padding), min(height, y + h + padding)],
                "label": "figure",
                "score": round(score, 3),
                "score_kind": "heuristic",
                "needs_review": True,
            })
        # Stable reading order. Nested elements of a connected drawing already
        # belong to the same component, so they yield one enclosing region.
        return sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0]))
