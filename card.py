import re
import cv2
import numpy as np
import mss
from rapidocr_onnxruntime import RapidOCR

# -------------------------------------------------------------
# 1. Validation & Detection Helpers
# -------------------------------------------------------------
def is_valid_luhn(card_number: str) -> bool:
    digits = [int(d) for d in re.sub(r'\D', '', card_number)]
    if not (13 <= len(digits) <= 19):
        return False
    
    checksum = 0
    for i, digit in enumerate(digits[::-1]):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0

# Regex patterns for candidates
CARD_CANDIDATE_REGEX = re.compile(r'\b(?:\d[ -]*?){13,19}\b')
SSN_REGEX = re.compile(r'\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b')

def contains_sensitive_data(text: str) -> bool:
    # Check SSN
    if SSN_REGEX.search(text):
        return True
    
    # Check Credit Card with Luhn calculation
    candidates = CARD_CANDIDATE_REGEX.findall(text)
    for cand in candidates:
        clean = re.sub(r'\D', '', cand)
        if is_valid_luhn(clean):
            return True
            
    return False

def blur_region(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Blurs the polygon/quadrilateral region defined by OCR bbox."""
    # Convert points to bounding rectangle
    x_min = max(0, int(np.min(box[:, 0])))
    y_min = max(0, int(np.max([0, np.min(box[:, 1])])))
    x_max = min(image.shape[1], int(np.max(box[:, 0])))
    y_max = min(image.shape[0], int(np.max(box[:, 1])))

    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return image

    # Heavy Gaussian Blur
    ksize = (int((x_max - x_min) | 1), int((y_max - y_min) | 1))
    # Cap kernel size to odd numbers within reasonable limits
    kx = max(15, (ksize[0] // 2) * 2 + 1)
    ky = max(15, (ksize[1] // 2) * 2 + 1)
    blurred_roi = cv2.GaussianBlur(roi, (kx, ky), 30)

    image[y_min:y_max, x_min:x_max] = blurred_roi
    return image

# -------------------------------------------------------------
# 2. Main Capture & Redaction Loop
# -------------------------------------------------------------
def run_live_redactor():
    engine = RapidOCR()
    sct = mss.mss()

    # Capture main monitor (change if you want specific ROI/window)
    monitor = sct.monitors[1]

    print("Running sensitive screen redactor. Press 'q' to stop.")

    while True:
        # 1. Ultra-fast screen grab
        screenshot = sct.grab(monitor)
        # Convert raw buffer to BGR numpy array
        frame = np.array(screenshot)[:, :, :3]

        # 2. Run OCR directly on BGR frame
        results, _ = engine(frame)

        # 3. Inspect recognized regions
        if results:
            for item in results:
                bbox, text, score = item[0], item[1], item[2]
                
                if score > 0.4 and contains_sensitive_data(text):
                    box_points = np.array(bbox, dtype=np.int32)
                    frame = blur_region(frame, box_points)

        # 4. Display live output window
        # Downscale display preview so it fits on screen comfortably
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Redacted Stream (Press Q to quit)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_redactor()