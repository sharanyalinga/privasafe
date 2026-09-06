import cv2
import numpy as np
import pyvirtualcam
from insightface.app import FaceAnalysis

# 1. Initialize detector
app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

# --- Settings ---
# ArcFace threshold: 0.63 is strict enough to eliminate false-positive cross matching
SIMILARITY_THRESHOLD = 0.63
PERSISTENCE_FRAMES = 25

whitelist_embeddings = []
active_tracks = {}  # track_id -> {"bbox": [...], "embedding": [...], "whitelisted": bool, "lost_frames": int}
next_track_id = 0
clicked_coords = None

def on_mouse_click(event, x, y, flags, param):
    global clicked_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords = (x, y)

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-6)

def compute_cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2))

def is_whitelisted(face_embedding):
    for saved in whitelist_embeddings:
        if compute_cosine_similarity(face_embedding, saved) >= SIMILARITY_THRESHOLD:
            return True
    return False

def get_centroid(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])

def apply_smooth_blur(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] < 6 or roi.shape[1] < 6:
        return

    # Feathered ellipse mask
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    axes = (roi.shape[1] // 2, roi.shape[0] // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 9) / 255.0

    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
    for c in range(3):
        roi[:, :, c] = (roi[:, :, c] * (1 - mask) + blurred_roi[:, :, c] * mask).astype(np.uint8)
    frame[y1:y2, x1:x2] = roi

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

cv2.namedWindow("Privacy Stream Pro")
cv2.setMouseCallback("Privacy Stream Pro", on_mouse_click)

with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fast inference on scaled 480p buffer without dropping detection quality
        scale_ratio = 480.0 / height
        det_w = int(width * scale_ratio)
        det_h = 480
        small_frame = cv2.resize(frame, (det_w, det_h))

        faces = app.get(small_frame)
        current_detections = []

        inv_scale = 1.0 / scale_ratio
        for face in faces:
            bbox = face.bbox * inv_scale
            emb = normalize(face.embedding)
            whitelisted = is_whitelisted(emb)
            current_detections.append({"bbox": bbox, "embedding": emb, "whitelisted": whitelisted})

        matched_tracks = set()

        # Track matching: feature-first comparison + dynamic spatial radius
        for det in current_detections:
            det_center = get_centroid(det["bbox"])
            det_diag = np.linalg.norm([det["bbox"][2] - det["bbox"][0], det["bbox"][3] - det["bbox"][1]])
            
            best_id = None
            best_score = -1.0

            for t_id, t_data in active_tracks.items():
                if t_id in matched_tracks:
                    continue

                track_center = get_centroid(t_data["bbox"])
                dist = np.linalg.norm(det_center - track_center)
                sim = compute_cosine_similarity(det["embedding"], t_data["embedding"])

                max_allowed_dist = max(160.0, det_diag * 1.5)
                
                # Match if identity aligns or spatial continuity is high
                if sim > 0.50 or dist < max_allowed_dist:
                    score = (sim * 0.75) + ((1.0 - min(dist / max_allowed_dist, 1.0)) * 0.25)
                    if score > best_score:
                        best_score = score
                        best_id = t_id

            if best_id is not None:
                active_tracks[best_id]["bbox"] = det["bbox"]
                # Running average updates face features smoothly as head turns
                active_tracks[best_id]["embedding"] = normalize(
                    0.85 * active_tracks[best_id]["embedding"] + 0.15 * det["embedding"]
                )
                active_tracks[best_id]["lost_frames"] = 0
                if det["whitelisted"]:
                    active_tracks[best_id]["whitelisted"] = True

                matched_tracks.add(best_id)
                det["track_id"] = best_id
            else:
                active_tracks[next_track_id] = {
                    "bbox": det["bbox"],
                    "embedding": det["embedding"],
                    "whitelisted": det["whitelisted"],
                    "lost_frames": 0
                }
                det["track_id"] = next_track_id
                next_track_id += 1

        # Handle Click: Toggle between Verified (Clear) and Removed (Blurred)
        if clicked_coords is not None:
            cx, cy = clicked_coords
            for det in current_detections:
                x1, y1, x2, y2 = det["bbox"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    t_id = det["track_id"]
                    is_currently_whitelisted = active_tracks[t_id]["whitelisted"]

                    if not is_currently_whitelisted:
                        # Whitelist & store feature
                        active_tracks[t_id]["whitelisted"] = True
                        whitelist_embeddings.append(det["embedding"])
                        print(f"Face ID {t_id} VERIFIED (Unblurred).")
                    else:
                        # Remove from whitelist & re-blur
                        active_tracks[t_id]["whitelisted"] = False
                        whitelist_embeddings = [
                            saved for saved in whitelist_embeddings
                            if compute_cosine_similarity(saved, det["embedding"]) < SIMILARITY_THRESHOLD
                        ]
                        print(f"Face ID {t_id} REMOVED (Re-blurred).")
                    break
            clicked_coords = None

        # Age out tracks
        to_delete = []
        for t_id, t_data in active_tracks.items():
            if t_id not in matched_tracks:
                t_data["lost_frames"] += 1
                if t_data["lost_frames"] > PERSISTENCE_FRAMES:
                    to_delete.append(t_id)

        for t_id in to_delete:
            del active_tracks[t_id]

        output_frame = frame.copy()

        # Apply smooth blur to unverified faces (No bounding boxes or text drawn)
        for t_id, t_data in active_tracks.items():
            if not t_data["whitelisted"]:
                apply_smooth_blur(output_frame, t_data["bbox"])

        vcam.send(output_frame)
        vcam.sleep_until_next_frame()
        cv2.imshow("Privacy Stream Pro", output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            whitelist_embeddings.clear()
            for t in active_tracks.values():
                t["whitelisted"] = False
            print("Cleared all permissions.")

cap.release()
cv2.destroyAllWindows()