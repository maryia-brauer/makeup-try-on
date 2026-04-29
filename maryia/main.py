import cv2
import numpy as np
import sys
import os

# ─────────────────────────────────────────────
# Load Haar Cascade classifiers
# ─────────────────────────────────────────────
def load_cascades():
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    mouth_path = cv2.data.haarcascades + "haarcascade_smile.xml"

    for p in [face_path, mouth_path]:
        if not os.path.exists(p):
            print(f"ERROR: Cascade not found: {p}")
            print("Fix: pip install --upgrade opencv-python")
            sys.exit(1)

    return (
        cv2.CascadeClassifier(face_path),
        cv2.CascadeClassifier(mouth_path)
    )


# ─────────────────────────────────────────────
# STEP 2 – Algorithm 1: Detect the face
# ─────────────────────────────────────────────
def detect_face(gray_frame, face_cascade):
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,   # how much the image is scaled down each step
        minNeighbors=5,    # higher = fewer false detections
        minSize=(80, 80)   # ignore tiny detections
    )

    if len(faces) == 0:
        return None

    # Pick the largest face (most likely the real one)
    largest = max(faces, key=lambda f: f[2] * f[3])
    return largest


# ─────────────────────────────────────────────
# STEP 3 – Algorithm 2: Feature Detection (mouth ROI)
# ─────────────────────────────────────────────
def detect_mouth(gray_frame, face_rect, mouth_cascade):
    fx, fy, fw, fh = face_rect

    # Lower 45 % of the face contains the mouth
    roi_y_start = fy + int(fh * 0.55)
    roi_y_end   = fy + fh
    roi_x_start = fx
    roi_x_end   = fx + fw

    mouth_roi = gray_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    mouths = mouth_cascade.detectMultiScale(
        mouth_roi,
        scaleFactor=1.07,
        minNeighbors=6,    # lower to 4 if mouth is rarely detected
        minSize=(30, 15)
    )

    if len(mouths) == 0:
        return None

    # Pick the LOWEST mouth rectangle (most likely the actual lips, not teeth)
    best = max(mouths, key=lambda m: m[1])  # largest y = lowest in ROI
    mx, my, mw, mh = best

    # Convert back to full-frame coordinates
    return (roi_x_start + mx,
            roi_y_start + my,
            mw, mh)


# ─────────────────────────────────────────────
# STEP 4 – Algorithm 3a: YCrCb Thresholding
# ─────────────────────────────────────────────
def threshold_lips(mouth_region_bgr):

    ycrcb = cv2.cvtColor(mouth_region_bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)

    # Pixels where Cr > 138 are likely lip-colored skin
    # Raise threshold (e.g. 150) if color bleeds onto surrounding skin
    _, mask = cv2.threshold(cr, 138, 255, cv2.THRESH_BINARY)

    # Clean the mask: remove specks, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill holes

    return mask


# ─────────────────────────────────────────────
# STEP 5 – Algorithm 3b: Contour Detection
# ─────────────────────────────────────────────
def get_lip_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return []

    # Sort by area, keep top-2
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[:2]


# ─────────────────────────────────────────────
# STEP 6 – Apply lipstick color
# ─────────────────────────────────────────────
def apply_lipstick(frame, mouth_rect, color_bgr):
    mx, my, mw, mh = mouth_rect

    # Safety clamp so we never go outside the frame
    h, w = frame.shape[:2]
    mx = max(0, mx);  my = max(0, my)
    mw = min(mw, w - mx);  mh = min(mh, h - my)

    if mw <= 0 or mh <= 0:
        return frame

    mouth_crop = frame[my:my+mh, mx:mx+mw]

    # --- Thresholding (Algorithm 3a) ---
    mask = threshold_lips(mouth_crop)

    # --- Contour Detection (Algorithm 3b) ---
    contours = get_lip_contours(mask)
    if not contours:
        return frame

    # Draw filled contours onto a blank canvas the size of the mouth crop
    lip_canvas = np.zeros_like(mouth_crop)
    cv2.drawContours(lip_canvas, contours, -1, color_bgr, thickness=cv2.FILLED)

    # Build an alpha mask from the contour fill (1 where lips are, 0 elsewhere)
    lip_gray  = cv2.cvtColor(lip_canvas, cv2.COLOR_BGR2GRAY)
    _, alpha  = cv2.threshold(lip_gray, 1, 255, cv2.THRESH_BINARY)
    alpha_3ch = cv2.merge([alpha, alpha, alpha]).astype(np.float32) / 255.0

    # Alpha blend: result = color * alpha * opacity + original * (1 - alpha * opacity)
    opacity      = 0.45
    mouth_float  = mouth_crop.astype(np.float32)
    color_float  = np.full_like(mouth_float, color_bgr, dtype=np.float32)

    blended = mouth_float * (1.0 - alpha_3ch * opacity) + color_float * (alpha_3ch * opacity)

    # Write blended region back into a copy of the frame
    result = frame.copy()
    result[my:my+mh, mx:mx+mw] = blended.astype(np.uint8)
    return result


# ─────────────────────────────────────────────
# STEP 7 – Draw HUD / UI overlay
# ─────────────────────────────────────────────
def draw_ui(frame, current_color_name, face_found, mouth_found):
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at the bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Controls text
    cv2.putText(frame, "R=Red  P=Pink  N=None  Q=Quit",
                (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Status line
    face_status  = "Face: YES" if face_found  else "Face: NO"
    mouth_status = "Mouth: YES" if mouth_found else "Mouth: NO"
    color_text   = f"Color: {current_color_name}"
    status = f"{face_status}   {mouth_status}   {color_text}"
    cv2.putText(frame, status,
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 1)

    return frame


# ─────────────────────────────────────────────
# STEP 8 – Main loop
# ─────────────────────────────────────────────
def main():
    print("Loading cascades...")
    face_cascade, mouth_cascade = load_cascades()
    print("Cascades loaded OK.")

    # Open default webcam.  If you get an error, change 0 → 1 or 2.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try changing VideoCapture(0) to VideoCapture(1).")

    # Lipstick color palette (BGR format)
    COLORS = {
        "Red":  (0,   0,   200),
        "Pink": (130, 105, 220),
        "None": None
    }
    current_color_name = "None"

    # Only run detection every N frames for performance
    detection_every = 3
    frame_count     = 0
    last_face_rect  = None
    last_mouth_rect = None

    print("Press R / P / N to change lipstick color.  Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from webcam. Exiting.")
            break

        # Mirror the frame so it feels like a mirror
        frame = cv2.flip(frame, 1)

        frame_count += 1
        run_detection = (frame_count % detection_every == 0)

        # ── Run detection on every Nth frame ──
        if run_detection:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)   # improves detection in varied lighting

            face_rect = detect_face(gray, face_cascade)
            last_face_rect = face_rect

            if face_rect is not None:
                mouth_rect = detect_mouth(gray, face_rect, mouth_cascade)
                last_mouth_rect = mouth_rect
            else:
                last_mouth_rect = None

        # ── Apply lipstick (every frame for smooth visuals) ──
        color_bgr = COLORS[current_color_name]
        if color_bgr is not None and last_mouth_rect is not None:
            frame = apply_lipstick(frame, last_mouth_rect, color_bgr)

        # ── Draw debug rectangles (optional, helps while developing) ──
        if last_face_rect is not None:
            fx, fy, fw, fh = last_face_rect
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 200, 0), 2)

        if last_mouth_rect is not None:
            mx, my, mw, mh = last_mouth_rect
            cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)

        # ── HUD overlay ──
        frame = draw_ui(frame,
                        current_color_name,
                        face_found=(last_face_rect is not None),
                        mouth_found=(last_mouth_rect is not None))

        cv2.imshow("Virtual Makeup Try-On", frame)

        # ── Keyboard input ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Quitting...")
            break
        elif key == ord('r') or key == ord('R'):
            current_color_name = "Red"
            print("Lipstick: Red")
        elif key == ord('p') or key == ord('P'):
            current_color_name = "Pink"
            print("Lipstick: Pink")
        elif key == ord('n') or key == ord('N'):
            current_color_name = "None"
            print("Lipstick: removed")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()