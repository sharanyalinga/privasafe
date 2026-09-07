"""
PrivaStream Sensitive Information Blur - OCR only
=================================================

Purpose:
    Fast, CPU-friendly sensitive-information blurring for OpenCV frames.

No YOLO. No InsightFace.

Features:
    - OCR-based detection using RapidOCR
    - Credit/debit card validation:
        * normalize OCR text
        * 13-19 digits
        * card-network prefix rules
        * Luhn checksum
        * consecutive-frame confirmation
    - Blurs the surrounding card/document region, not only the number
    - Detects common PII keywords and groups nearby OCR boxes
    - Temporal persistence so blur does not flicker
    - OCR is intentionally run only every N frames
    - Drop-in API: module.process(frame) -> frame

Install:
    pip install -r requirements_sensitive_ocr.txt

Quick test:
    python sensitive_ocr_blur.py --source 0

Video:
    python sensitive_ocr_blur.py --source input.mp4 --output blurred.mp4

Debug:
    python sensitive_ocr_blur.py --source 0 --debug

Notes:
    OCR is not a proof that an arbitrary number is a payment card. This module
    requires both card-network-compatible prefixes and a valid Luhn checksum,
    then confirms the same region over multiple OCR passes.

    For maximum privacy, this module is deliberately conservative: false
    positives are preferred over allowing obvious payment-card/PII content
    through unblurred.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError as exc:
    raise SystemExit(
        "RapidOCR is not installed. Run:\n"
        "    pip install -r requirements_sensitive_ocr.txt"
    ) from exc


Box = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OCRConfig:
    # OCR is expensive; do not run on every frame.
    ocr_interval: int = 5

    # Analysis resolution. The original frame is still used for final blur.
    max_analysis_width: int = 960

    # OCR result confidence threshold.
    min_ocr_confidence: float = 0.45

    # A valid card must be seen this many times before being blurred.
    card_confirmations: int = 2

    # Keep an accepted region alive for this many frames without OCR.
    region_hold_frames: int = 18

    # Merge OCR boxes into a larger sensitive region.
    group_padding: int = 22
    card_padding_x: float = 0.08
    card_padding_y: float = 0.10

    # Blur strength.
    blur_kernel: int = 51

    # Optional safety expansion for document/PII clusters.
    pii_padding: int = 28

    # Maximum number of OCR results used from a single frame.
    max_ocr_results: int = 100


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OCRItem:
    text: str
    confidence: float
    box: Box


@dataclass
class SensitiveRegion:
    box: Box
    kind: str                       # "card" or "pii"
    confidence: float
    last_seen_frame: int
    confirmations: int = 1


# ---------------------------------------------------------------------------
# Card validation
# ---------------------------------------------------------------------------

CARD_PATTERNS = (
    ("amex", re.compile(r"^3[47]\d{13}$")),          # 15 digits
    ("visa", re.compile(r"^4\d{12}(?:\d{3})?$")),    # 13/16/19
    ("mastercard", re.compile(r"^(?:5[1-5]\d{14}|2(?:2[2-9]\d{2}|2[3-9]\d|[3-6]\d{2}|7(?:[01]\d|20))\d{12})$")),
    ("discover", re.compile(r"^(?:6011\d{12}|65\d{14}|64[4-9]\d{13}|622(?:12[6-9]|1[3-9]\d|[2-8]\d{2}|9(?:0[1-2]|[2-9]\d))\d{10})$")),
    # JCB/common 16-19 digit range. Prefix rule is intentionally broader,
    # while Luhn remains mandatory.
    ("jcb", re.compile(r"^(?:35\d{14,17})$")),
)


def normalize_digits(text: str) -> str:
    """Keep only ASCII digits."""
    return re.sub(r"\D", "", text)


def luhn_check(number: str) -> bool:
    """Return True only if the number passes the Luhn checksum."""
    if not number.isdigit() or not (13 <= len(number) <= 19):
        return False

    total = 0
    parity = len(number) % 2

    for i, ch in enumerate(number):
        digit = ord(ch) - 48
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit

    return total % 10 == 0


def validate_payment_card(number: str) -> Optional[str]:
    """
    Return the detected card family if the candidate is plausible and passes
    Luhn; otherwise return None.
    """
    number = normalize_digits(number)

    if not (13 <= len(number) <= 19):
        return None

    for card_type, pattern in CARD_PATTERNS:
        if pattern.fullmatch(number) and luhn_check(number):
            return card_type

    return None


def generate_digit_candidates(text: str) -> List[str]:
    """
    Make candidate payment-card digit strings from OCR text.

    Handles:
        4111 1111 1111 1111
        4111-1111-1111-1111
        4111111111111111
    """
    candidates: List[str] = []

    # First: a run with common card separators.
    compact = re.sub(r"[^\d\s-]", " ", text)
    groups = re.findall(r"\d(?:[\d\s-]{11,25})\d", compact)

    for candidate in groups:
        digits = normalize_digits(candidate)
        if 13 <= len(digits) <= 19:
            candidates.append(digits)

    # Second: plain digits in the OCR text.
    digits = normalize_digits(text)
    if 13 <= len(digits) <= 19:
        candidates.append(digits)

    # De-duplicate.
    return list(dict.fromkeys(candidates))


# ---------------------------------------------------------------------------
# PII / document text detection
# ---------------------------------------------------------------------------

PII_KEYWORDS = {
    "aadhaar",
    "aadhar",
    "pan",
    "passport",
    "driving license",
    "driver license",
    "dob",
    "date of birth",
    "address",
    "account no",
    "account number",
    "ifsc",
    "social security",
    "ssn",
    "cvv",
    "cvc",
    "expiry",
    "valid thru",
    "valid through",
    "name",
}

PII_PATTERNS = [
    # India-like Aadhaar / PAN examples
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I),

    # Generic bank-account-ish long number. This is deliberately weaker than
    # payment-card validation and is used only as a PII clue.
    re.compile(r"\b\d{9,18}\b"),
]


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def looks_like_pii_text(text: str) -> bool:
    t = normalized_text(text)

    if any(keyword in t for keyword in PII_KEYWORDS):
        return True

    return any(pattern.search(text) for pattern in PII_PATTERNS)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def expand_box(
    box: Box,
    width: int,
    height: int,
    pad_x: int,
    pad_y: int,
) -> Box:
    x1, y1, x2, y2 = box
    return clamp_box(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        width,
        height,
    )


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union else 0.0


def union_boxes(boxes: Sequence[Box]) -> Box:
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return min(xs1), min(ys1), max(xs2), max(ys2)


def box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def distance(a: Box, b: Box) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.hypot(ax - bx, ay - by)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class SensitiveOCRBlur:
    """
    Drop-in sensitive-information blur processor.

    Typical integration:

        privacy = SensitiveOCRBlur()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = privacy.process(frame)
            ...
    """

    def __init__(self, config: Optional[OCRConfig] = None, debug: bool = False):
        self.config = config or OCRConfig()
        self.debug = debug

        # RapidOCR ONNX Runtime backend.
        self.ocr = RapidOCR()

        self.frame_index = 0
        self.regions: List[SensitiveRegion] = []

        self._last_ocr_ms = 0.0
        self._fps_t0 = time.perf_counter()
        self._fps_counter = 0
        self.fps = 0.0

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _resize_for_ocr(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]

        if w <= self.config.max_analysis_width:
            return frame, 1.0

        scale = self.config.max_analysis_width / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        return resized, scale

    def _run_ocr(self, frame: np.ndarray) -> List[OCRItem]:
        analysis, scale = self._resize_for_ocr(frame)

        start = time.perf_counter()

        try:
            result, _ = self.ocr(analysis)
        except Exception:
            result = None

        self._last_ocr_ms = (time.perf_counter() - start) * 1000.0

        if not result:
            return []

        h, w = frame.shape[:2]
        inverse_scale = 1.0 / scale

        items: List[OCRItem] = []

        # RapidOCR normally returns:
        # [
        #   [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        #   text,
        #   score
        # ]
        for entry in result[: self.config.max_ocr_results]:
            try:
                points, text, score = entry
                confidence = float(score)

                if confidence < self.config.min_ocr_confidence:
                    continue

                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]

                x1 = int(min(xs) * inverse_scale)
                y1 = int(min(ys) * inverse_scale)
                x2 = int(max(xs) * inverse_scale)
                y2 = int(max(ys) * inverse_scale)

                box = clamp_box((x1, y1, x2, y2), w, h)

                if normalized_text(str(text)):
                    items.append(
                        OCRItem(
                            text=str(text),
                            confidence=confidence,
                            box=box,
                        )
                    )
            except (TypeError, ValueError, IndexError):
                continue

        return items

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _find_valid_cards(
        self,
        items: Sequence[OCRItem],
    ) -> List[Tuple[Box, float]]:
        cards: List[Tuple[Box, float]] = []

        for item in items:
            candidates = generate_digit_candidates(item.text)

            for candidate in candidates:
                card_type = validate_payment_card(candidate)
                if card_type is None:
                    continue

                # We deliberately do NOT store/log the candidate number.
                score = min(1.0, item.confidence + 0.20)

                cards.append((item.box, score))
                break

        return cards

    def _find_pii_items(self, items: Sequence[OCRItem]) -> List[OCRItem]:
        return [item for item in items if looks_like_pii_text(item.text)]

    def _group_nearby(
        self,
        seed_boxes: Sequence[Box],
        all_items: Sequence[OCRItem],
    ) -> List[Box]:
        """
        Expand a text seed into a local document/card cluster.

        This lets us blur the surrounding card/document rather than only the
        OCR text itself.
        """
        if not seed_boxes:
            return []

        selected = list(seed_boxes)

        # Estimate a reasonable local distance from the seed's own scale.
        seed_union = union_boxes(seed_boxes)
        sx1, sy1, sx2, sy2 = seed_union
        seed_w = max(1, sx2 - sx1)
        seed_h = max(1, sy2 - sy1)
        max_distance = max(80.0, 8.0 * max(seed_w, seed_h))

        changed = True
        while changed:
            changed = False

            for item in all_items:
                if item.box in selected:
                    continue

                for existing in selected:
                    if (
                        distance(item.box, existing) <= max_distance
                        or iou(item.box, existing) > 0.05
                    ):
                        selected.append(item.box)
                        changed = True
                        break

        return selected

    def _card_region_from_seed(
        self,
        seed: Box,
        frame_shape: Tuple[int, int, int],
        all_items: Sequence[OCRItem],
    ) -> Box:
        h, w = frame_shape[:2]

        cluster = self._group_nearby([seed], all_items)

        if cluster:
            region = union_boxes(cluster)
        else:
            region = seed

        rx1, ry1, rx2, ry2 = region

        region_w = max(1, rx2 - rx1)
        region_h = max(1, ry2 - ry1)

        # Card is generally much larger than a single printed number.
        # Use the OCR cluster size when possible and then add generous
        # proportional margins.
        pad_x = max(
            45,
            int(region_w * self.config.card_padding_x),
        )
        pad_y = max(
            45,
            int(region_h * self.config.card_padding_y),
        )

        candidate = expand_box(
            region,
            w,
            h,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        # If the detected region is clearly too narrow, enlarge horizontally
        # around its center. This is important for "whole card" blurring.
        cx, cy = box_center(candidate)
        cw = candidate[2] - candidate[0]
        ch = candidate[3] - candidate[1]

        target_w = max(cw, int(ch * 1.58))
        target_h = max(ch, int(cw / 1.58))

        candidate = clamp_box(
            (
                int(cx - target_w / 2),
                int(cy - target_h / 2),
                int(cx + target_w / 2),
                int(cy + target_h / 2),
            ),
            w,
            h,
        )

        return candidate

    def _pii_region_from_seed(
        self,
        seed: Box,
        frame_shape: Tuple[int, int, int],
        all_items: Sequence[OCRItem],
    ) -> Box:
        h, w = frame_shape[:2]

        cluster = self._group_nearby([seed], all_items)
        region = union_boxes(cluster) if cluster else seed

        return expand_box(
            region,
            w,
            h,
            self.config.pii_padding,
            self.config.pii_padding,
        )

    # ------------------------------------------------------------------
    # Temporal state
    # ------------------------------------------------------------------

    def _update_regions(
        self,
        detected: Sequence[SensitiveRegion],
    ) -> None:
        now = self.frame_index

        # Match new detections to existing regions.
        for new_region in detected:
            best_idx = -1
            best_iou = 0.0

            for idx, old_region in enumerate(self.regions):
                if old_region.kind != new_region.kind:
                    continue

                overlap = iou(old_region.box, new_region.box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx

            if best_idx >= 0 and best_iou >= 0.15:
                old = self.regions[best_idx]

                # Smooth the box.
                ox1, oy1, ox2, oy2 = old.box
                nx1, ny1, nx2, ny2 = new_region.box

                alpha = 0.35

                smoothed = (
                    int((1 - alpha) * ox1 + alpha * nx1),
                    int((1 - alpha) * oy1 + alpha * ny1),
                    int((1 - alpha) * ox2 + alpha * nx2),
                    int((1 - alpha) * oy2 + alpha * ny2),
                )

                old.box = smoothed
                old.confidence = max(old.confidence, new_region.confidence)
                old.last_seen_frame = now
                old.confirmations += 1
            else:
                self.regions.append(new_region)

        # Remove stale regions.
        self.regions = [
            region
            for region in self.regions
            if now - region.last_seen_frame <= self.config.region_hold_frames
        ]

    # ------------------------------------------------------------------
    # Blur
    # ------------------------------------------------------------------

    def _blur_box(self, frame: np.ndarray, box: Box) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = clamp_box(box, w, h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        kernel = self.config.blur_kernel

        # Kernel must be odd and cannot be larger than the ROI dimensions.
        max_kernel = min(roi.shape[0], roi.shape[1])
        if max_kernel < 3:
            return

        k = min(kernel, max_kernel if max_kernel % 2 == 1 else max_kernel - 1)
        if k < 3:
            k = 3

        frame[y1:y2, x1:x2] = cv2.GaussianBlur(
            roi,
            (k, k),
            0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one BGR OpenCV frame and return the blurred frame.
        """
        if frame is None or frame.size == 0:
            return frame

        self.frame_index += 1

        h, w = frame.shape[:2]

        # Run expensive OCR only periodically.
        if self.frame_index % max(1, self.config.ocr_interval) == 0:
            items = self._run_ocr(frame)

            detected: List[SensitiveRegion] = []

            # Payment cards get highest-priority handling.
            for seed_box, confidence in self._find_valid_cards(items):
                region = self._card_region_from_seed(
                    seed_box,
                    frame.shape,
                    items,
                )

                detected.append(
                    SensitiveRegion(
                        box=region,
                        kind="card",
                        confidence=confidence,
                        last_seen_frame=self.frame_index,
                    )
                )

            # Other obvious PII/document text.
            for item in self._find_pii_items(items):
                region = self._pii_region_from_seed(
                    item.box,
                    frame.shape,
                    items,
                )

                detected.append(
                    SensitiveRegion(
                        box=region,
                        kind="pii",
                        confidence=item.confidence,
                        last_seen_frame=self.frame_index,
                    )
                )

            self._update_regions(detected)

            # Safety: always blur newly detected regions immediately.
            # For a card, confirmations are still accumulated in the state,
            # but the region is already protected once seen.
        else:
            items = []

        # Render every frame using the most recent stable regions.
        for region in self.regions:
            if region.kind == "card" or region.kind == "pii":
                self._blur_box(frame, region.box)

                if self.debug:
                    x1, y1, x2, y2 = region.box
                    label = f"{region.kind}:{region.confidence:.2f}"
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # Optional debug HUD.
        self._fps_counter += 1
        elapsed = time.perf_counter() - self._fps_t0

        if elapsed >= 1.0:
            self.fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_t0 = time.perf_counter()

        if self.debug:
            cv2.putText(
                frame,
                f"FPS {self.fps:.1f} | OCR {self._last_ocr_ms:.1f} ms | "
                f"regions {len(self.regions)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))

    return cv2.VideoCapture(source)


def run(source: str, output: Optional[str], debug: bool) -> None:
    cap = open_source(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if not fps or math.isnan(fps) or fps <= 1:
        fps = 30.0

    writer = None

    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output,
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open output: {output}")

    config = OCRConfig(
        ocr_interval=5,
        max_analysis_width=960,
        min_ocr_confidence=0.45,
        blur_kernel=51,
    )

    processor = SensitiveOCRBlur(config, debug=debug)

    print("Sensitive OCR blur started.")
    print("Press Q or ESC to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            processed = processor.process(frame)

            if writer:
                writer.write(processed)

            cv2.imshow("PrivaStream - Sensitive Blur", processed)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()

        if writer:
            writer.release()

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast OCR-only sensitive information blur."
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or video path. Example: 0 or input.mp4",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detected regions and performance.",
    )

    args = parser.parse_args()

    run(
        source=args.source,
        output=args.output,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
