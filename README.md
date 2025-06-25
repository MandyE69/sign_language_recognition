# 🤟 Real-Time Sign Language Recognition with Voice (Python + MediaPipe)

This project is a real-time hand sign recognition system that uses computer vision and machine learning to identify gestures from hand landmarks.

---

## 📦 Features

- ✋ Real-time hand detection using MediaPipe
- 🏷️ Labeled data collection with webcam
- 🧠 Model training using Random Forest
- 🔮 Live gesture prediction from camera input

---

## 🧰 Tech Stack

- Python 3.9+
- [MediaPipe](https://mediapipe.dev/)
- OpenCV
- scikit-learn
- NumPy / pandas

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
```
### 2. Install Dependencies
Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
Install packages:
```bash
pip install -r requirements.txt
```

### 🎯 Scripts & Usage
### 1️⃣ collect_data.py — Collect Labeled Sign Data
```bash
python collect_data.py
```
what it does!! 
---Enter a label (e.g., Hello)
---Press s to start collecting 200 frames
---Press Esc to stop
---It appends data to data.csv
---Repeat for multiple signs (e.g., Yes, No, Thanks, etc.)

### 2️⃣ train_model.py — Train the Classifier
```bash
python train_model.py
```

what it does!!
---Loads data.csv
---Trains a Random Forest classifier
---Prints evaluation metrics
---Saves model as sign_model.pkl

### 3️⃣ predict_live.py — Real-Time Prediction (No Voice)
```bash
python predict_live.py
```

what it does!!
---Opens webcam
---Detects hand landmarks
---Predicts and displays label on screen

Also there is hand_detect.py for detecting hand through mediapipe just to know it working 
