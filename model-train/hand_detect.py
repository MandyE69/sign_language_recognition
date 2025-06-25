import cv2
import mediapipe as mp
import time
import numpy as np

# ----- 1.  Initialize MediaPipe Hand model -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,       # streaming mode
    max_num_hands=2,               # track up to 2 hands
    min_detection_confidence=0.5,  # tweak if you get false negatives
    min_tracking_confidence=0.5
)

# ----- 2.  Start webcam -----
cap = cv2.VideoCapture(0)  # change index if you have multiple cameras
prev_time = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame…")
            continue

        # Flip & convert BGR→RGB (MediaPipe requirement)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ----- 3.  Run hand landmark detection -----
        results = hands.process(rgb)

        # Draw and optionally save landmark positions
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 3a. Draw landmarks & connections
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 3b. Extract landmark coordinates
                coords = np.array([[lm.x, lm.y, lm.z]
                                   for lm in hand_landmarks.landmark])
                # Print as one line so you can redirect output if you want
                print("LANDMARKS", coords.flatten().round(3).tolist())

        # ----- 4.  FPS counter -----
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time) if prev_time else 0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ----- 5.  Display -----
        cv2.imshow("Real-Time Hand Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # Esc to exit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
