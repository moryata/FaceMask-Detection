"""
Modul untuk deteksi masker wajah.
"""
import cv2
import numpy as np
import os
import time

class MaskDetector:
    """
    Kelas untuk mendeteksi masker wajah dengan preprocessing yang lebih baik.
    """

    def __init__(self, model, input_size, class_labels):
        """
        Inisialisasi detektor masker.

        Args:
            model: Model deteksi masker Keras
            input_size (tuple): Ukuran input model (width, height)
            class_labels (list): Daftar label kelas
        """
        self.model = model
        self.input_size = input_size
        self.class_labels = class_labels

        # Untuk tracking performa
        self.last_inference_time = 0
        self.avg_inference_time = 0
        self.inference_count = 0

        # Import preprocessing function dari TensorFlow Keras
        from tensorflow.keras.applications.efficientnet import preprocess_input
        self.preprocess_input = preprocess_input

        # Cetak informasi inisialisasi
        print(f"[INFO] Detektor masker diinisialisasi dengan ukuran input {input_size}")
        print(f"[INFO] Label kelas: {class_labels}")

    def detect_mask(self, face_roi):
        """
        Mendeteksi masker pada ROI wajah dengan preprocessing yang lebih baik.

        Args:
            face_roi: Region of interest wajah

        Returns:
            tuple: (pred_idx, confidence) di mana pred_idx adalah indeks prediksi
                  dan confidence adalah nilai kepercayaan
        """
        if face_roi.size == 0:
            return -1, 0.0

        try:
            # Mulai timer untuk mengukur waktu inferensi
            start_time = time.time()

            # Preprocessing yang lebih baik
            # 1. Konversi ke RGB
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # 2. Resize dengan interpolasi yang lebih baik
            face_resized = cv2.resize(face_rgb, self.input_size, interpolation=cv2.INTER_AREA)

            # 3. Normalisasi gambar
            face_array = np.array(face_resized, dtype="float32")

            # 4. Expand dimensions untuk batch
            face_array = np.expand_dims(face_array, axis=0)

            # 5. Preprocessing khusus untuk EfficientNet
            face_preprocessed = self.preprocess_input(face_array)

            # 6. Prediksi dengan model
            preds = self.model.predict(face_preprocessed, verbose=0)

            # 7. Ambil indeks dengan confidence tertinggi
            pred_idx = np.argmax(preds[0])
            confidence = preds[0][pred_idx]

            # Hitung waktu inferensi
            inference_time = time.time() - start_time
            self.last_inference_time = inference_time

            # Update rata-rata waktu inferensi
            self.inference_count += 1
            self.avg_inference_time = ((self.inference_count - 1) * self.avg_inference_time + inference_time) / self.inference_count

            return pred_idx, confidence

        except Exception as e:
            print(f"[WARNING] Error saat memproses ROI: {e}")
            return -1, 0.0

    def get_label_info(self, pred_idx, confidence):
        """
        Mendapatkan informasi label berdasarkan indeks prediksi dengan tampilan yang lebih baik.

        Args:
            pred_idx (int): Indeks prediksi
            confidence (float): Nilai kepercayaan

        Returns:
            tuple: (label, color, label_text) di mana label adalah teks label,
                  color adalah warna untuk visualisasi, dan label_text adalah teks
                  yang akan ditampilkan
        """
        # Tentukan Label dan Warna dengan threshold confidence
        confidence_threshold = 0.7  # Threshold untuk kepercayaan tinggi

        if 0 <= pred_idx < len(self.class_labels):
            label = self.class_labels[pred_idx]

            # Warna berdasarkan label dan confidence
            if label == "Menggunakan Masker":
                if confidence >= confidence_threshold:
                    color = (0, 255, 0)  # Hijau terang untuk kepercayaan tinggi
                else:
                    color = (0, 192, 0)  # Hijau lebih gelap untuk kepercayaan rendah
            elif label == "Tanpa Masker":
                if confidence >= confidence_threshold:
                    color = (0, 0, 255)  # Merah terang untuk kepercayaan tinggi
                else:
                    color = (0, 0, 192)  # Merah lebih gelap untuk kepercayaan rendah
            else:
                color = (255, 255, 0)  # Kuning untuk label lain
        else:
            label = "Unknown"
            color = (128, 128, 128)  # Abu-abu untuk unknown

        # Format teks label dengan confidence dan waktu inferensi
        if self.inference_count > 0:
            label_text = f"{label} ({confidence*100:.1f}%) [{self.last_inference_time*1000:.1f}ms]"
        else:
            label_text = f"{label} ({confidence*100:.1f}%)"

        return label, color, label_text
