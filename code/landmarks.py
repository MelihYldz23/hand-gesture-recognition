import cv2
import mediapipe as mp

# MediaPipe el modülü
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# El algılayıcı ayarları
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Kamera başlat
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kameradan görüntü alınamıyor.")
        break

    # Görüntüyü çevirmek ve BGR'den RGB'ye dönüştürmek
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El tespiti
    results = hands.process(rgb_frame)

    # Eğer eller tespit edildiyse çiz
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Ekrana göster
    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
