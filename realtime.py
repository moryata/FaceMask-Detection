"""
Aplikasi deteksi masker wajah real-time menggunakan webcam.
"""
import sys
import cv2
import time

# Import modul kustom
from utils.dependency_manager import upgrade_pip, check_basic_dependencies, check_tensorflow_dependencies
from utils.model_loader import load_model, load_face_cascade, check_model_file
from detectors.face_detector import FaceDetector
from detectors.mask_detector import MaskDetector
import config

def main():
    """
    Fungsi utama aplikasi deteksi masker wajah real-time.
    """
    # --- Upgrade Pip di Awal ---
    upgrade_pip()

    # --- Periksa Dependensi Dasar ---
    all_deps_ok = check_basic_dependencies()
    if not all_deps_ok:
        print("[ERROR] Tidak semua dependensi dasar dapat diinstal. Keluar.")
        sys.exit(1)

    # --- Import Rich Console Setelah Dipastikan Ada ---
    from rich.console import Console
    console = Console()

    # --- Periksa Dependensi TensorFlow dan Model ---
    if not check_tensorflow_dependencies():
        console.print("[bold red][ERROR][/] Dependensi TensorFlow tidak dapat diinstal. Keluar.")
        sys.exit(1)

    if not check_model_file(config.MODEL_FILE, console):
        sys.exit(1)

    # --- Load Model ---
    model, _ = load_model(config.MODEL_FILE, config.INPUT_SIZE, config.CLASS_LABELS, console)

    # --- Load Haar Cascade ---
    face_cascade = load_face_cascade(console)

    # --- Inisialisasi Detektor ---
    face_detector = FaceDetector(
        face_cascade,
        min_size=config.FACE_DETECTION_MIN_SIZE,
        scale_factor=config.FACE_DETECTION_SCALE_FACTOR,
        min_neighbors=config.FACE_DETECTION_MIN_NEIGHBORS
    )

    mask_detector = MaskDetector(
        model,
        config.INPUT_SIZE,
        config.CLASS_LABELS
    )

    # --- Inisialisasi Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("[bold red][ERROR][/] Webcam tidak dapat diakses. Periksa koneksi atau ID perangkat.")
        sys.exit(1)

    # --- Loop Utama Deteksi ---
    console.print("[bold blue][INFO][/] Memulai deteksi real-time menggunakan EfficientNetB0. Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            console.print("[bold yellow][WARNING][/] Gagal membaca frame dari webcam. Mencoba lagi...")
            continue  # Coba baca frame berikutnya

        if frame is None or frame.size == 0:
            console.print("[bold yellow][WARNING][/] Menerima frame kosong. Melewati.")
            continue

        # Deteksi wajah
        faces = face_detector.detect_faces(frame)

        for face_coords in faces:
            # Dapatkan ROI wajah
            face_roi, (x1, y1, x2, y2) = face_detector.get_face_roi(frame, face_coords)
            if face_roi.size == 0:
                continue

            # Deteksi masker
            pred_idx, confidence = mask_detector.detect_mask(face_roi)

            # Dapatkan informasi label
            _, color, label_text = mask_detector.get_label_info(pred_idx, confidence)

            # Gambar Output
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15  # Sesuaikan posisi teks
            cv2.putText(frame, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Tampilkan Frame
        try:
            cv2.imshow("Face Mask Detection (EfficientNetB0)", frame)
        except Exception as e:
            console.print(f"[bold red]ERROR:[/]: Tidak dapat menampilkan frame: {e}")
            console.print("       Apakah Anda menjalankan di environment dengan dukungan GUI?")
            break  # Keluar dari loop jika tidak bisa display

        # Keluar dengan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            console.print("[INFO] Tombol 'q' ditekan. Keluar...")
            break

    # --- Cleanup ---
    console.print("[bold blue][INFO][/] Menghentikan program dan membersihkan resource...")
    cap.release()
    cv2.destroyAllWindows()
    # Tambahkan jeda sedikit agar window sempat tertutup sebelum pesan selesai
    time.sleep(0.5)
    console.print("[bold green][INFO][/] Selesai.")

if __name__ == "__main__":
    main()

