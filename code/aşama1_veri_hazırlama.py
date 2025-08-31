import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset klasöründeki tüm CSV dosyalarını oku
dataset_dir = "dataset"  # dataset klasörünün yolu
all_data = []
all_labels = []

# Her sınıfın verisini ayrı dosyadan oku
for filename in os.listdir(dataset_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(dataset_dir, filename)
        df = pd.read_csv(file_path)

        # Etiket olarak dosya adını kullan (örneğin, 'beş.csv' -> 'beş')
        gesture_name = filename.split('.')[0]

        # Özellikleri ve etiketleri ayır
        X = df.iloc[:, 1:].values  # Landmark verileri
        y = np.full(len(X), gesture_name)  # Etiketler

        all_data.append(X)
        all_labels.append(y)

# Veriyi birleştir
X_all = np.vstack(all_data)
y_all = np.hstack(all_labels)

# Sınıf dağılımını kontrol et
unique, counts = np.unique(y_all, return_counts=True)
class_counts = dict(zip(unique, counts))

# Eksik sınıfları ekrana yaz
for label, count in class_counts.items():
    if count < 2:
        print(f"❌ Hata: '{label}' hareketi için en az 2 veri gerekiyor! Şu an sadece {count} tane var.")
        exit()

# Minimum test veri sayısını belirle
num_classes = len(class_counts)  # Toplam sınıf sayısı
min_test_size = max(num_classes, int(len(y_all) * 0.2))  # Test seti için min sınıf kadar veri ayır

# test_size oranını uygun şekilde belirle
test_size = min_test_size / len(y_all)  # test_size oranını otomatik ayarla

# Veri setini eğitim ve test olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, stratify=y_all, random_state=42)

# Veriyi kaydet
np.savez("gesture_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print(f"✅ Veri başarıyla hazırlandı ve kaydedildi: 'gesture_data.npz' (test_size = {test_size:.2f})")
