import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Ask for label
label = input("Enter label for this gesture Hello: ")
save_file = "data.csv"

# Make sure file exists and has header
if not os.path.exists(save_file):
    with open(save_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
        header.append("label")
        writer.writerow(header)

# Start camera
cap = cv2.VideoCapture(0)
print("Press 's' to start capturing data...")
collecting = False
collected = 0
target = 200

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            if collecting and collected < target:
                with open(save_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(np.append(landmarks, label))
                collected += 1

    # Instructions & count overlay
    status = f"Collecting: {collected}/{target}" if collecting else "Waiting to start (press 's')"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print(f"Started collecting '{label}'")
        collecting = True
        collected = 0
        time.sleep(1)  # brief pause to adjust hand

    if key == 27 or collected >= target:  # ESC or done
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"Data collection complete for '{label}'! Saved to {save_file}")
