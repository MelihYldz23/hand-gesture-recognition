import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import pyautogui

MODEL_PATH = "gesture_model.keras"
DATASET_DIR = "dataset"
PREPARATION_DURATION = 1
COOLDOWN = 2

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
INPUT_DIM = model.input_shape[1]

# Gesture names
gesture_names = sorted([
    os.path.splitext(f)[0]
    for f in os.listdir(DATASET_DIR)
    if f.lower().endswith(".csv")
])
print("🎯 Active classes in presentation mode:", gesture_names)

gesture_actions_presentation = {
    "0": (lambda: pyautogui.press("f5"), "Presentation Started"),
    "1": (lambda: pyautogui.press("esc"), "Presentation Stopped"),
    "6": (lambda: pyautogui.press("right"), "Next Slide"),
    "7": (lambda: pyautogui.press("left"), "Previous Slide"),
    "5": (lambda: pyautogui.press("esc"), "Presentation Stopped and Exited"),
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
        if feedback_message:
            cv2.putText(frame, feedback_message,
                        (50, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"Please wait... {COOLDOWN - (now - last_trigger):.1f}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Presentation Mode", frame)
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
                name = gesture_names[cid] if cid < len(gesture_names) else "Unknown"

                cv2.putText(frame, name, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"➡️ Gesture: {name}, ID={cid}, Prediction={preds}")

                if name in gesture_actions_presentation:
                    action, feedback_message = gesture_actions_presentation[name]
                    action()

                    last_trigger = now
                    waiting = True

                    if name == "5":
                        # Close PowerPoint
                        os.system('taskkill /f /im POWERPNT.EXE')

                        # Close camera
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

            gesture_start = None
    else:
        gesture_start = None
        cv2.putText(frame, "Hand is not detected!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if feedback_message:
        cv2.putText(frame, feedback_message,
                    (50, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.putText(frame, "Mode: Presentation", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Presentation Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

