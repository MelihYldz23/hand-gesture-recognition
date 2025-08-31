import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import pyautogui
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import subprocess
import json

MODEL_PATH = "gesture_model.keras"
DATASET_DIR = "dataset"
PREPARATION_DURATION = 1
COOLDOWN = 2

# 📁 video_path.json dosyasını oku
with open("video_path.json", "r") as f:
    config = json.load(f)
video_path = config.get("path")
vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"

# Kamera aç
cap = cv2.VideoCapture(0)

# Kamera açılana kadar bekle
while not cap.isOpened():
    time.sleep(0.1)

# Kamera açıldıktan sonra video başlat
if os.path.exists(video_path):
    subprocess.Popen([vlc_path, video_path])
else:
    print("❌ Video not found:", video_path)

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Model yükle
model = tf.keras.models.load_model(MODEL_PATH)
INPUT_DIM = model.input_shape[1]

# Gesture isimleri
gesture_names = sorted([
    os.path.splitext(f)[0]
    for f in os.listdir(DATASET_DIR)
    if f.lower().endswith(".csv")
])
print("🎯 Active classes in media mode:", gesture_names)

def get_volume_interface():
    dev = AudioUtilities.GetSpeakers()
    iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return iface.QueryInterface(IAudioEndpointVolume)

def change_volume(delta):
    vol = get_volume_interface()
    current = vol.GetMasterVolumeLevelScalar()
    new_volume = np.clip(current + delta, 0.0, 1.0)
    vol.SetMasterVolumeLevelScalar(new_volume, None)
    print(f"🔊 New volume level: {new_volume:.2f}")

gesture_actions_media = {
    "0": (lambda: pyautogui.press("space"), "Video started"),
    "1": (lambda: pyautogui.press("space"), "Video paused"),
    "2": (lambda: change_volume(+0.2), "Volume increased"),
    "3": (lambda: change_volume(-0.2), "Volume decreased"),
    "5": (lambda: print("📷 Camera shutting down..."), "Camera turned off"),
    "6": (lambda: pyautogui.hotkey("alt", "right"), "Skipped forward 10 seconds"),
    "7": (lambda: pyautogui.hotkey("alt", "left"), "Skipped backward 10 seconds"),
}

def get_feature_vector(landmarks):
    x = np.array([lm.x for lm in landmarks])
    y = np.array([lm.y for lm in landmarks])
    nx = (x - x.min()) / (x.max() - x.min() + 1e-6)
    ny = (y - y.min()) / (y.max() - y.min() + 1e-6)
    inter = np.empty((42,), dtype=np.float32)
    inter[0::2] = nx
    inter[1::2] = ny
    return inter.reshape(1, INPUT_DIM)

cap = cv2.VideoCapture(0)
gesture_start = None
last_trigger = 0
waiting = False
feedback_message = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    now = time.time()

    if waiting and now - last_trigger < COOLDOWN:
        cv2.putText(frame, f"Please wait... {COOLDOWN - (now - last_trigger):.1f}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if feedback_message:
            cv2.putText(frame, feedback_message,
                        (frame.shape[1] - 400, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Media Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    waiting = False
    feedback_message = ""

    if res.multi_hand_landmarks:
        if gesture_start is None:
            gesture_start = now

        elapsed = now - gesture_start
        if elapsed < PREPARATION_DURATION:
            cv2.putText(frame, f"Preparing... {PREPARATION_DURATION - elapsed:.1f}s",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for hand in res.multi_hand_landmarks:
                inp = get_feature_vector(hand.landmark)
                preds = model.predict(inp, verbose=0)
                cid = int(np.argmax(preds))
                name = gesture_names[cid] if cid < len(gesture_names) else "Bilinmeyen"

                cv2.putText(frame, name, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"➡️ Gesture: {name}, ID={cid}, Tahmin={preds}")

                if name in gesture_actions_media:
                    action, feedback_message = gesture_actions_media[name]
                    action()

                    if name == "5":
                        # VLC'yi kapat
                        os.system('taskkill /f /im vlc.exe')

                        # Kamerayı kapat
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                    last_trigger = now
                    waiting = True

            gesture_start = None
    else:
        gesture_start = None
        cv2.putText(frame, "Hand not detected!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "Mod: Media", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Media Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
