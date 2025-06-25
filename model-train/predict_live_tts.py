import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speech speed

# Load trained model
with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

prev_prediction = ""
cooldown = 30  # speak only once per ~30 frames

frame_count = 0
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

            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]

            break

    # Text-to-Speech trigger
    frame_count += 1
    if prediction != prev_prediction and prediction != "No hand" and frame_count > cooldown:
        print(f"🗣️  Speaking: {prediction}")
        engine.say(prediction)
        engine.runAndWait()
        prev_prediction = prediction
        frame_count = 0

    # Show prediction
    cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Prediction (with Speech)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
