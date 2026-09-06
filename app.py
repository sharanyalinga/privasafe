import cv2
import numpy as np
import pyvirtualcam
from insightface.app import FaceAnalysis

# 1. Initialize detector
app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Settings
SIMILARITY_THRESHOLD = 0.44  # Adjusted for better profile tolerance
PERSISTENCE_FRAMES = 25     # Frames to remember a face after it turns/disappears

whitelist_embeddings = []
active_tracks = {}  # track_id -> {"bbox": [...], "whitelisted": bool, "lost_frames": int}
next_track_id = 0
clicked_coords = None

def on_mouse_click(event, x, y, flags, param):
    global clicked_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords = (x, y)

def compute_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)

def is_whitelisted(face_embedding):
    for saved in whitelist_embeddings:
        if compute_cosine_similarity(face_embedding, saved) >= SIMILARITY_THRESHOLD:
            return True
    return False

def get_centroid(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

def apply_smooth_blur(frame, bbox):
    """Applies an elliptical feathered blur instead of a harsh square box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] < 5 or roi.shape[1] < 5:
        return

    # Create elliptical mask
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    axes = (roi.shape[1] // 2, roi.shape[0] // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 11) / 255.0

    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
    for c in range(3):
        roi[:, :, c] = (roi[:, :, c] * (1 - mask) + blurred_roi[:, :, c] * mask).astype(np.uint8)
    frame[y1:y2, x1:x2] = roi

cap = cv2.VideoCapture(0)
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

        faces = app.get(frame)
        current_detections = []

        # Process detected faces
        for face in faces:
            bbox = face.bbox
            emb = face.embedding
            whitelisted = is_whitelisted(emb)
            current_detections.append({"bbox": bbox, "embedding": emb, "whitelisted": whitelisted})

        # Simple spatial association (Track linking across frames)
        matched_tracks = set()
        for det in current_detections:
            det_center = get_centroid(det["bbox"])
            best_id = None
            min_dist = 120.0  # Pixel threshold for tracking across frames

            for t_id, t_data in active_tracks.items():
                if t_id in matched_tracks:
                    continue
                track_center = get_centroid(t_data["bbox"])
                dist = np.linalg.norm(det_center - track_center)
                if dist < min_dist:
                    min_dist = dist
                    best_id = t_id

            if best_id is not None:
                # Update existing track
                active_tracks[best_id]["bbox"] = det["bbox"]
                active_tracks[best_id]["lost_frames"] = 0
                if det["whitelisted"]:
                    active_tracks[best_id]["whitelisted"] = True
                matched_tracks.add(best_id)
                det_track_id = best_id
            else:
                # New track
                active_tracks[next_track_id] = {
                    "bbox": det["bbox"],
                    "whitelisted": det["whitelisted"],
                    "lost_frames": 0
                }
                det_track_id = next_track_id
                next_track_id += 1

            # Check if user clicked on this face
            if clicked_coords is not None:
                cx, cy = clicked_coords
                x1, y1, x2, y2 = det["bbox"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    is_now_white = not active_tracks[det_track_id]["whitelisted"]
                    active_tracks[det_track_id]["whitelisted"] = is_now_white
                    if is_now_white:
                        whitelist_embeddings.append(det["embedding"])
                        print(f"Track {det_track_id} Whitelisted (Embedding registered).")
                    else:
                        print(f"Track {det_track_id} Re-blurred.")
                    clicked_coords = None

        # Age out missing tracks (handles head turns where detection fails temporarily)
        to_delete = []
        for t_id, t_data in active_tracks.items():
            if t_id not in matched_tracks:
                t_data["lost_frames"] += 1
                if t_data["lost_frames"] > PERSISTENCE_FRAMES:
                    to_delete.append(t_id)

        for t_id in to_delete:
            del active_tracks[t_id]

        output_frame = frame.copy()
        display_frame = frame.copy()

        # Apply blurs based on persistent track states
        for t_id, t_data in active_tracks.items():
            bbox = t_data["bbox"]
            is_clear = t_data["whitelisted"]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            if not is_clear:
                apply_smooth_blur(output_frame, bbox)
                apply_smooth_blur(display_frame, bbox)
                color = (0, 0, 255)
                status = "BLURRED"
            else:
                color = (0, 255, 0)
                status = "CLEARED"

            # Overlay visual indicator in the local control window
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"ID:{t_id} [{status}]", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        vcam.send(output_frame)
        vcam.sleep_until_next_frame()
        cv2.imshow("Privacy Stream Pro", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            whitelist_embeddings.clear()
            for t in active_tracks.values():
                t["whitelisted"] = False
            print("Reset all permissions.")

cap.release()
cv2.destroyAllWindows()