import cv2
import numpy as np
import sys
import os


# ─────────────────────────────────────────────────────────────
# STEP 8: Main loop
# ─────────────────────────────────────────────────────────────
 
def main():
    global current_color
 
    print("=" * 50)
    print("  Virtual Makeup Try-On System")
    print("  Controls: R=Red  P=Pink  N=None  Q=Quit")
    print("=" * 50)
 
    # Open webcam (0 = first camera)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
 
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Fix: Try VideoCapture(1) or VideoCapture(2)")
        sys.exit(1)
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
    print("Webcam opened. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

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