import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import json
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import pyautogui
import screen_brightness_control as sbc
from datetime import datetime

# ---------- AYARLAR ----------
MODEL_PATH = "gesture_model.keras"
DATASET_DIR = "dataset"
PREPARATION_DURATION = 2  # saniye
COOLDOWN = 3  # saniye (2'den 3'e çıkarıldı)
MOD_CONFIG_PATH = "mod_config.json"
# ------------------------------

# 💾 Ekran görüntüsü klasörü (varsayılan 'Resimler/Ekran Görüleri')
screenshot_dir = r"C:\Users\mlhyl\OneDrive\Resimler\Ekran Görüntüleri"
os.makedirs(screenshot_dir, exist_ok=True)  # klasör yoksa oluştur


# --- Mod Seçimi Okuma ---
def get_active_mode():
    try:
        with open(MOD_CONFIG_PATH, "r") as f:
            return json.load(f).get("mode", "computer")
    except FileNotFoundError:
        return "computer"


# 1) Mediapipe el tanıma
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 2) Modeli yükle ve giriş boyutunu al
model = tf.keras.models.load_model(MODEL_PATH)
INPUT_DIM = model.input_shape[1]  # eğitimde belirlenen 42

# 3) Sınıf adlarını dataset klasöründen oku
gesture_names = sorted([
    os.path.splitext(f)[0]
    for f in os.listdir(DATASET_DIR)
    if f.lower().endswith(".csv")
])
print("🔢 Bulunan gesture sınıfları:", gesture_names)


# 4) Ses kontrol fonksiyonları (Windows)
def set_volume(level):
    dev = AudioUtilities.GetSpeakers()
    iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    vol = iface.QueryInterface(IAudioEndpointVolume)
    vol.SetMasterVolumeLevelScalar(level, None)


def unmute():
    print("🔊 Ses Açıldı!")
    set_volume(1.0)


def mute():
    print("🔇 Ses Kapatıldı!")
    set_volume(0.0)


# --- 5) Modlara Göre Gesture-Komut Sözlükleri ---

# Bilgisayar (PC) modu - feedback mesajları eklendi
gesture_actions_pc = {
    "0": (unmute, "Sound turned up!"),
    "1": (mute, "Sound turned off!"),
    "2": (
        lambda: (sbc.set_brightness(100), print("Brightness increased!")),
        "Brightness increased!"
    ),
    "3": (
        lambda: (sbc.set_brightness(0), print("Brightness decreased!")),
        "Brightness decreased!"
    ),
    "4": (
        lambda: (
            pyautogui.screenshot(
                os.path.join(screenshot_dir,
                             f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            ),
            print("Screenshot saved!")
        ),
        "Screenshot saved!"
    ),
    "5": (
        lambda: (
            print("🛑 Çıkılıyor..."),
            cap.release(),
            cv2.destroyAllWindows(),
            exit()
        ),
        "Exiting..."
    )
}


# Medya modu - feedback mesajları eklendi
gesture_actions_media = {
    "0": (lambda: print("⏯️ Play/Pause"), "⏯️ Play/Pause"),
    "1": (lambda: print("🔇 Mute/Unmute"), "🔇 Mute/Unmute"),
    "2": (lambda: print("⏭️ Next Track"), "⏭️ Next Track"),
    "3": (lambda: print("⏮️ Previous Track"), "⏮️ Previous Track"),
    "4": (lambda: print("🔊 Volume Up / Down Toggle"), "🔊 Volume Up/Down")
}

# Slayt (Sunum) modu - feedback mesajları eklendi
gesture_actions_slide = {
    "0": (lambda: print("▶️ Start Presentation"), "▶️ Presentation Started"),
    "1": (lambda: print("❌ End Presentation"), "❌ Presentation Ended"),
    "2": (lambda: print("➡ Next Slide"), "➡ Next Slide"),
    "3": (lambda: print("⬅ Previous Slide"), "⬅ Previous Slide"),
    "4": (lambda: print("🔒 Lock Slide / Highlight"), "🔒 Slide Locked")
}


# 6) Landmark → 42-boyut özellik vektörü fonksiyonu
def get_feature_vector(landmarks):
    x = np.array([lm.x for lm in landmarks])
    y = np.array([lm.y for lm in landmarks])
    nx = (x - x.min()) / (x.max() - x.min() + 1e-6)
    ny = (y - y.min()) / (y.max() - y.min() + 1e-6)
    inter = np.empty((42,), dtype=np.float32)
    inter[0::2] = nx
    inter[1::2] = ny
    return inter.reshape(1, INPUT_DIM)


# 7) Kamera ve zamanlayıcılar
cap = cv2.VideoCapture(0)
gesture_start = None
last_trigger = 0
waiting = False
feedback_message = ""  # Feedback mesajı için yeni değişken

# ---------- ANA DÖNGÜ ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    now = time.time()

    # cooldown kontrolü
    if waiting and now - last_trigger < COOLDOWN:
        txt = f"Please {COOLDOWN - (now - last_trigger):.1f}s wait"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Sağ alt köşeye feedback mesajı ekleme
        if feedback_message:
            cv2.putText(frame, feedback_message,
                        (frame.shape[1] - 400, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    waiting = False
    feedback_message = ""  # Feedback mesajını sıfırla

    # el algılama ve hazırlık süresi
    if res.multi_hand_landmarks:
        if gesture_start is None:
            gesture_start = now

        elapsed = now - gesture_start
        remaining = max(0, PREPARATION_DURATION - elapsed)

        if elapsed < PREPARATION_DURATION:
            cv2.putText(frame,
                        f"Get ready! {remaining:.1f}s",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # tahmin zamanı
            for hand in res.multi_hand_landmarks:
                inp = get_feature_vector(hand.landmark)
                preds = model.predict(inp, verbose=0)
                cid = int(np.argmax(preds))
                name = gesture_names[cid] if cid < len(gesture_names) else "Bilinmeyen"

                cv2.putText(frame, name, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"→ Tahmin: {name} (id={cid}), olasılıklar: {preds}")

                # aktif moda göre komut çalıştır
                mode = get_active_mode()
                if mode == "computer" and name in gesture_actions_pc:
                    action, feedback_message = gesture_actions_pc[name]
                    action()
                elif mode == "media" and name in gesture_actions_media:
                    action, feedback_message = gesture_actions_media[name]
                    action()
                elif mode == "presentation" and name in gesture_actions_slide:
                    action, feedback_message = gesture_actions_slide[name]
                    action()
                else:
                    # tanınmayan hareket veya geçersiz mod
                    feedback_message = ""

                last_trigger = now
                waiting = True

            gesture_start = None
    else:
        gesture_start = None
        cv2.putText(frame, "Hand is not detected!",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Modu ekrana yazdırma
    mode = get_active_mode()
    cv2.putText(frame, f"Mod: {mode.capitalize()}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()