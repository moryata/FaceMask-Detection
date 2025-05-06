Siap — aku bikinkan README.md versi profesional buat GitHub project berdasarkan penjelasan sistem deteksi masker wajah tadi. Nanti tinggal kamu sesuaikan nama repo dan upload filenya ke GitHub. Ini dia isi README-nya:

---

````markdown
# 📷 Face Mask Detection System (EfficientNetB0)

Sistem deteksi masker wajah berbasis *real-time video* menggunakan webcam, OpenCV, dan model CNN EfficientNetB0. Sistem ini mampu mendeteksi wajah pada gambar video lalu mengklasifikasikan apakah seseorang:
- Menggunakan masker
- Tidak menggunakan masker
- Tidak diketahui

Sistem bekerja secara real-time dengan kecepatan sekitar 15-30 FPS (tergantung spesifikasi hardware).

---

## 📖 Fitur Utama
- Deteksi wajah menggunakan **Haar Cascade Classifier** dari OpenCV.
- Klasifikasi masker wajah dengan model **EfficientNetB0** yang telah di-*fine-tune*.
- Visualisasi bounding box:
  - 🟩 **Hijau** untuk *With Mask*
  - 🟥 **Merah** untuk *Without Mask*
  - ⬜ **Putih** untuk *Unknown*
- Real-time inferensi dari webcam.
- Confidence score ditampilkan di atas tiap wajah.

---

## 🖥️ Sistem dan Library yang Digunakan
### 📌 Hardware
- Komputer/laptop dengan webcam.
- CPU / GPU (opsional untuk percepatan inferensi).

### 📌 Software Dependencies
- Python 3.8 – 3.11
- TensorFlow / Keras
- OpenCV
- NumPy
- Rich (opsional, untuk console output)

**Install requirement:**
```bash
pip install -r requirements.txt
````

---

## 📊 Dataset

Dataset terdiri atas 3 kelas:

* `with_mask` : gambar orang memakai masker dengan benar.
* `without_mask` : gambar orang tanpa masker.
* `incorrect_mask` : gambar orang memakai masker tidak sesuai.

**Struktur dataset:**

```
dataset/
├── train/
│   ├── with_mask/
│   ├── without_mask/
│   └── incorrect_mask/
└── validation/
    ├── with_mask/
    ├── without_mask/
    └── incorrect_mask/
```

---

## 🧠 Arsitektur Model

Menggunakan arsitektur **EfficientNetB0** dengan modifikasi:

* Global Average Pooling
* Dense 256 neuron + ReLU
* Dropout 0.5
* Dense 128 neuron + ReLU
* Dropout 0.25
* Dense output 3 neuron + Softmax

---

## 📷 Alur Sistem

1. **Capture video frame** dari webcam.
2. Deteksi wajah menggunakan *Haar Cascade*.
3. Ekstraksi ROI wajah.
4. Resize & preprocess gambar.
5. Klasifikasikan masker menggunakan model EfficientNetB0.
6. Tampilkan hasil real-time dengan bounding box & label.

---

## 🚀 Cara Menjalankan

1. Clone repository:

```bash
git clone https://github.com/username/face-mask-detection.git
cd face-mask-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Jalankan program:

```bash
python mask_detection.py
```

4. Tekan `q` untuk keluar dari aplikasi.

---

## 📈 Potensi Pengembangan

* Upgrade model ke **EfficientNetV2** atau **MobileNetV3**.
* Ganti face detector ke **MTCNN** / **RetinaFace**.
* Optimasi performa via **model quantization** dan **GPU acceleration**.
* Fitur **tracking antar frame**.
* Integrasi dengan sistem **akses kontrol**.
* Statistik kepatuhan masker.

---

## 📑 Lisensi

Proyek ini bersifat open-source dan bebas digunakan untuk keperluan edukasi, riset, maupun pengembangan lebih lanjut.

---

## 📩 Kontak

Jika ada pertanyaan, saran, atau ingin berkolaborasi, silakan hubungi:

📧 [moryata@gmail.com](mailto:moryata@gmail.com)
📱 IG: @moryata

---

