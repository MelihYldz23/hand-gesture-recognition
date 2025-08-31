import cv2
import mediapipe as mp
import csv
import time
import os

# Mediapipe el modülü
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

# Veri klasörü
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

gesture_name = "0"
csv_writer = None
file = None

print("Veri kaydı başlıyor, ellerini kameraya göster...")
print("Hareket seçmek için 0-9 tuşlarına basın (örn. 1=Yumruk).")

time.sleep(2)

waiting_for_confirmation = False
hand_detected_time: None = None
saved_data = None  # Geçici olarak kaydedilecek veri

# 42-boyut özellik vektörü çıkarma
def get_landmark_data(landmarks):
    data = []
    for lm in landmarks:
        data.extend([lm.x, lm.y])
    return data

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    hand_detected = result.multi_hand_landmarks is not None
    now = time.time()

    # Tuş yakalama
    key = cv2.waitKey(1) & 0xFF

    # 0–9 tuşlarına basıldığında gesture_name seç ve CSV hazırla
    if key in [ord(str(d)) for d in range(10)] and not waiting_for_confirmation:
        gesture_name = chr(key)
        # Dosya yolu
        file_path = os.path.join(dataset_dir, f"{gesture_name}.csv")
        existed = os.path.exists(file_path)
        file = open(file_path, "a", newline="")
        csv_writer = csv.writer(file)
        if not existed:
            csv_writer.writerow([f"point_{i}_{axis}" for i in range(21) for axis in ("x", "y")])
        print(f"✔ Hareket '{gesture_name}' seçildi. Kayıt için hazır.")
        hand_detected_time = None
        continue  # sonraki döngüye geç

    # 'q' ile çık
    if key == ord('q'):
        break

    # El algılama ve sabit kalma süresi
    if hand_detected and csv_writer is not None and not waiting_for_confirmation:
        if hand_detected_time is None:
            hand_detected_time = now
        elapsed = now - hand_detected_time
        if elapsed >= 1.5:
            for hand_landmarks in result.multi_hand_landmarks:
                saved_data = get_landmark_data(hand_landmarks.landmark)
                waiting_for_confirmation = True
                print(f"📸 Hareket '{gesture_name}' algılandı. Kaydetmek için 'e', iptal için 'h'.")
    else:
        hand_detected_time = None

    # Onay / iptal
    if waiting_for_confirmation:
        if key == ord('e'):
            csv_writer.writerow(saved_data)
            print(f"✅ '{gesture_name}' verisi kaydedildi!")
            waiting_for_confirmation = False
            hand_detected_time = None
        elif key == ord('h'):
            print("❌ Kayıt iptal edildi.")
            waiting_for_confirmation = False
            hand_detected_time = None

    # Ekrana yazılar
    cv2.putText(frame, f"Hareket: {gesture_name}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if waiting_for_confirmation:
        cv2.putText(frame, "Kaydet (E) / İptal (H)", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    elif not hand_detected:
        cv2.putText(frame, "El Algılanmadı!", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        remaining = max(0, 1.5 - (now - hand_detected_time))
        cv2.putText(frame, f"Kayıt için bekle: {remaining:.1f}s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Recording", frame)

# Temizlik
if file is not None:
    file.close()
cap.release()
cv2.destroyAllWindows()
