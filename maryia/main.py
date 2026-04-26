import cv2
import numpy as np
import sys
import os

# ─────────────────────────────────────────────────────────────
# Load Haar Cascade classifiers
# ─────────────────────────────────────────────────────────────
 
def load_cascades():
    face_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    mouth_path = cv2.data.haarcascades + "haarcascade_smile.xml"
 
    for p in [face_path, mouth_path]:
        if not os.path.exists(p):
            print(f"ERROR: Cascade not found: {p}")
            print("Fix: pip install --upgrade opencv-python")
            sys.exit(1)
 
    return cv2.CascadeClassifier(face_path), cv2.CascadeClassifier(mouth_path)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Face detection (Haar Cascade)
# ─────────────────────────────────────────────────────────────
 
def detect_face(frame, face_cascade):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


# ─────────────────────────────────────────────────────────────
# STEP 8: Main loop
# ─────────────────────────────────────────────────────────────
 
def main():
    global current_color
 
    print("=" * 50)
    print("  Virtual Makeup Try-On System")
    print("  Controls: R=Red  P=Pink  N=None  Q=Quit")
    print("=" * 50)

    face_cascade, mouth_cascade = load_cascades()
 
    # Open webcam (0 = first camera)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
 
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Fix: Try VideoCapture(1) or VideoCapture(2)")
        sys.exit(1)
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
    print("Webcam opened. Press Q to quit.\n")

    frame_count    = 0
    detect_every   = 3       # run Haar every 3 frames for performance
    last_face      = None
    last_mouth     = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)   # mirror
        frame_count += 1

        # ── Detection (every N frames) ───────────────────────
        if frame_count % detect_every == 0:
            face = detect_face(frame, face_cascade)
            last_face = face

        # ── Debug boxes (comment out for clean demo) ─────────
        if last_face is not None:
            fx, fy, fw, fh = last_face
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 1)

        # # ── Draw UI ───────────────────────────────────────────
        # output_frame = draw_ui(output_frame, current_color)
 
        cv2.imshow("Virtual Makeup Try-On", frame)
 
        # ── Keyboard input ────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
 
        if   key in (ord('q'), ord('Q')): break


    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")

if __name__ == "__main__":
    main() 