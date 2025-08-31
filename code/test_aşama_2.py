import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pickle

# CSV dosyalarını oku
dataset_dir = "dataset"
all_data = []
all_labels = []

for filename in os.listdir(dataset_dir):
    if filename.endswith(".csv"):
        label = os.path.splitext(filename)[0]
        file_path = os.path.join(dataset_dir, filename)
        df = pd.read_csv(file_path)
        all_data.append(df.values)
        all_labels.extend([label] * len(df))

X = np.vstack(all_data)  # shape (num_samples, 42)
y = np.array(all_labels)

# Etiketleri sayıya çevir
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Eğitim ve test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# Modeli oluştur
model = Sequential([
    Dense(64, activation='relu', input_shape=(42,)),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Eğit
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Test doğruluğu
loss, acc = model.evaluate(X_test, y_test)
print(f"Test doğruluğu: {acc:.2f}")

# Modeli ve etiket çeviriciyi kaydet
model.save("gesture_model.keras")
with open("labels.pkl", "wb") as f:
    pickle.dump(le, f)
