import cv2
import numpy as np
import re
import threading
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
ocr_lock = threading.Lock()

# Detected sensitive regions: list of dicts {bbox, text, score}
active_redactions = []
latest_detected_text = []

def luhn_checksum(card_number: str) -> bool:
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(digits[::-1]):
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d
    return (checksum % 10) == 0

def is_sensitive(text: str) -> bool:
    cleaned = re.sub(r'[^0-9]', '', text)
    # Match any 12-19 digit card, 4-digit cluster, 3-digit CVV, or expiry MM/YY
    if 12 <= len(cleaned) <= 19:
        return True
    if len(cleaned) == 4 and cleaned.isdigit():
        return True
    if len(cleaned) in [3, 4] and cleaned.isdigit():
        return True
    if re.search(r'\b(0[1-9]|1[0-2])[\/\-]([0-9]{2}|[0-9]{4})\b', text):
        return True
    return False

def ocr_worker(frame_bgr):
    global active_redactions, latest_detected_text
    
    # 1. Enhance contrast for OCR
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    result, _ = engine(enhanced)
    
    new_boxes = []
    text_log = []

    if result:
        for box, text, score in result:
            clean_text = text.strip()
            text_log.append(f"'{clean_text}' ({score:.2f})")
            
            if score > 0.30 and is_sensitive(clean_text):
                pts = np.array(box, dtype=np.int32)
                x1 = max(0, int(np.min(pts[:, 0])) - 20)
                y1 = max(0, int(np.min(pts[:, 1])) - 15)
                x2 = min(frame_bgr.shape[1], int(np.max(pts[:, 0])) + 20)
                y2 = min(frame_bgr.shape[0], int(np.max(pts[:, 1])) + 15)
                new_boxes.append((x1, y1, x2, y2, clean_text))

    with ocr_lock:
        active_redactions = new_boxes
        latest_detected_text = text_log

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

win_name = "Card OCR Diagnostic [Hold Card to Camera]"
cv2.namedWindow(win_name)

ocr_thread = None
frame_idx = 0

print("\n--- Card OCR Diagnostic Started ---")
print("Hold your card steady in front of the camera.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Run OCR inference continuously in background
    if ocr_thread is None or not ocr_thread.is_alive():
        ocr_thread = threading.Thread(target=ocr_worker, args=(frame.copy(),), daemon=True)
        ocr_thread.start()

    output = frame.copy()

    with ocr_lock:
        # Draw all red blocks on detected numbers
        for x1, y1, x2, y2, txt in active_redactions:
            # Solid censor plate
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 200), -1)
            cv2.putText(output, "BLOCKED", (x1 + 4, max(y1 + 15, y2 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        # Display latest text read in top-left HUD
        hud_text = " | ".join(latest_detected_text[:4]) if latest_detected_text else "Scanning..."
        cv2.putText(output, f"OCR Sees: {hud_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow(win_name, output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()