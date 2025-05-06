"""
Konfigurasi untuk aplikasi deteksi masker wajah.
"""
import os

# --- Konfigurasi Model ---
MODEL_DIR = "models"
# Hanya menggunakan model Modular Keras
MODEL_FILE = os.path.join(MODEL_DIR, "mask_detector_efficientnetb0_modular.h5")
# Gunakan model terbaik jika tersedia
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_mask_detector_efficientnetb0_modular.h5")
if os.path.exists(BEST_MODEL_FILE):
    MODEL_FILE = BEST_MODEL_FILE

# --- Konfigurasi Input Size ---
# Ukuran input yang direkomendasikan untuk EfficientNetB0
INPUT_SIZE = (224, 224)

# --- Konfigurasi Label ---
CLASS_LABELS = ["Unknown", "Menggunakan Masker", "Tanpa Masker"]

# --- Konfigurasi Deteksi Wajah ---
FACE_DETECTION_MIN_SIZE = (60, 60)
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5

# --- Konfigurasi Tampilan ---
MASK_COLOR = (0, 255, 0)  # Hijau untuk menggunakan masker
NO_MASK_COLOR = (0, 0, 255)  # Merah untuk tanpa masker
UNKNOWN_COLOR = (255, 255, 255)  # Putih untuk unknown
