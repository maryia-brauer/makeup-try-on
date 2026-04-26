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
# STEP 1 — Face detection
# ─────────────────────────────────────────────
def detect_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    return max(faces, key=lambda r: r[2] * r[3])


# ─────────────────────────────────────────────
# STEP 2 — Mouth region detection
# ─────────────────────────────────────────────
def detect_mouth(frame, face_rect, mouth_cascade):
    fx, fy, fw, fh = face_rect

    lower_y = fy + int(fh * 0.60)
    lower_h = int(fh * 0.40)

    roi = frame[lower_y: lower_y + lower_h, fx: fx + fw]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    mouths = mouth_cascade.detectMultiScale(
        gray_roi,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(30, 15)
    )

    if len(mouths) == 0:
        return None

    mx, my, mw, mh = max(mouths, key=lambda r: r[2] * r[3])

    return (fx + mx, lower_y + my, mw, mh)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    global current_color

    print("=" * 50)
    print(" Virtual Makeup Try-On System")
    print(" Controls: R=Red  P=Pink  N=None  Q=Quit")
    print("=" * 50)

    face_cascade, mouth_cascade = load_cascades()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Fix: Try VideoCapture(1) or VideoCapture(2)")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam opened. Press Q to quit.\n")

    frame_count = 0
    detect_every = 3

    last_face = None
    last_mouth = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # ─────────────────────────────
        # Detection every N frames
        # ─────────────────────────────
        if frame_count % detect_every == 0:
            face = detect_face(frame, face_cascade)
            last_face = face

            if face is not None:
                last_mouth = detect_mouth(frame, face, mouth_cascade)
            else:
                last_mouth = None

        # ─────────────────────────────
        # Debug boxes
        # ─────────────────────────────
        if last_face is not None:
            fx, fy, fw, fh = last_face
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 1)

        if last_mouth is not None:
            mx, my, mw, mh = last_mouth
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 100, 0), 1)

        # ─────────────────────────────
        # Display
        # ─────────────────────────────
        cv2.imshow("Virtual Makeup Try-On", frame)

        # ─────────────────────────────
        # Keyboard input
        # ─────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()