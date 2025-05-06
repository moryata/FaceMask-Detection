"""
Modul untuk memuat model deteksi masker.
"""
import os
import sys
import cv2

def load_model(model_path, input_size, class_labels, console):
    """
    Memuat model Keras.

    Args:
        model_path (str): Path ke file model
        input_size (tuple): Ukuran input model (width, height)
        class_labels (list): Daftar label kelas
        console: Rich console untuk output

    Returns:
        tuple: (model, None) di mana None adalah placeholder untuk kompatibilitas
    """
    model = None

    try:
        # Set TensorFlow log level before importing (1=INFO, 2=WARNING, 3=ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Load model dengan TensorFlow Keras
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)

        # Tampilkan informasi model
        console.print(f"[bold blue][INFO][/] Model dimuat dari: {model_path}")

        # Tampilkan ringkasan model
        console.print("[blue][INFO][/] Ringkasan model:")
        model.summary(print_fn=lambda x: console.print(f"[dim]{x}[/dim]"))

        # Tampilkan informasi input dan output
        input_shape = model.input_shape
        output_shape = model.output_shape
        console.print(f"[blue][INFO][/] Input shape: {input_shape}")
        console.print(f"[blue][INFO][/] Output shape: {output_shape}")

        # Verifikasi bahwa output shape sesuai dengan jumlah kelas
        if output_shape[-1] != len(class_labels):
            console.print(f"[bold yellow][WARNING][/] Jumlah output model ({output_shape[-1]}) tidak sesuai dengan jumlah kelas ({len(class_labels)})")

    except Exception as e:
        console.print(f"[bold red][ERROR][/] Gagal memuat model Keras. Error: {e}")
        sys.exit(1)

    return model, None

def load_face_cascade(console):
    """
    Memuat Haar Cascade untuk deteksi wajah.

    Args:
        console: Rich console untuk output

    Returns:
        cv2.CascadeClassifier: Classifier untuk deteksi wajah
    """
    haar_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(haar_cascade_path):
        console.print(f"[bold red][ERROR][/] File Haar Cascade tidak ditemukan di: {haar_cascade_path}")
        console.print("       Pastikan OpenCV terinstal dengan benar atau path valid.")
        sys.exit(1)
    try:
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        if face_cascade.empty():
            raise IOError(f"Gagal memuat cascade classifier dari {haar_cascade_path}")
        return face_cascade
    except Exception as e:
        console.print(f"[bold red][ERROR][/] Gagal memuat Haar Cascade: {e}")
        sys.exit(1)

def check_model_file(model_path, console):
    """
    Memeriksa keberadaan file model.

    Args:
        model_path (str): Path ke file model
        console: Rich console untuk output

    Returns:
        bool: True jika file model ada, False jika tidak
    """
    if not os.path.exists(model_path):
        console.print(f"[bold red][ERROR][/] Model tidak ditemukan di: {model_path}")
        console.print("Pastikan Anda telah menjalankan skrip pelatihan terlebih dahulu.")
        return False
    return True
