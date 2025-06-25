import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model
with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

print("🖐️  Show your sign... (ESC to exit)")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = "No hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 landmark points (x, y, z)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]

            break  # Only handle one hand at a time

    # Overlay prediction on frame
    cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show webcam feed
    cv2.imshow("Sign Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
