"""
Modul untuk mengelola dependensi aplikasi.
"""
import importlib
import subprocess
import sys

def upgrade_pip():
    """
    Meng-upgrade pip ke versi terbaru.
    """
    print("[INFO] Mencoba meng-upgrade pip...")
    try:
        # Menggunakan sys.executable memastikan pip di environment yang benar
        command = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[INFO] Pip berhasil di-upgrade (atau sudah terbaru).")
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Gagal meng-upgrade pip. Kode error: {e.returncode}")
        print("         Lanjutkan dengan versi pip saat ini.")
    except Exception as e:
        print(f"[WARNING] Gagal meng-upgrade pip. Error: {e}")
        print("         Lanjutkan dengan versi pip saat ini.")

def check_and_install(package_name, import_name=None, install_name=None):
    """
    Memeriksa dan menginstal paket jika belum terinstal.

    Args:
        package_name (str): Nama paket untuk diimpor
        import_name (str, optional): Nama modul untuk diimpor. Default ke package_name.
        install_name (str, optional): Nama paket untuk diinstal. Default ke package_name.

    Returns:
        bool: True jika paket berhasil diimpor atau diinstal, False jika gagal.
    """
    if import_name is None:
        import_name = package_name
    if install_name is None:
        install_name = package_name  # Nama untuk di-pip install

    try:
        importlib.import_module(import_name)
        print(f"[INFO] Dependensi '{import_name}' sudah terinstal.")
        return True
    except ImportError:
        print(f"[INFO] Dependensi '{import_name}' tidak ditemukan. Mencoba menginstal '{install_name}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # Sembunyikan output pip sukses
            print(f"[INFO] Dependensi '{install_name}' berhasil diinstal.")
            # Coba impor lagi untuk memastikan
            importlib.import_module(import_name)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Gagal menginstal '{install_name}'. Pip error output:")
            print("-------------------------------------------")
            print(e.stderr.decode())
            print("-------------------------------------------")
        except Exception as e:
            print(f"[ERROR] Gagal menginstal '{install_name}'. Error: {e}")

        print(f"[ERROR] Silakan instal '{install_name}' secara manual dan jalankan kembali skrip.")
        return False

def check_basic_dependencies():
    """
    Memeriksa dependensi dasar yang diperlukan aplikasi.

    Returns:
        bool: True jika semua dependensi berhasil diinstal, False jika tidak.
    """
    all_deps_ok = True
    all_deps_ok &= check_and_install("numpy", "numpy")
    all_deps_ok &= check_and_install("opencv-python", "cv2", "opencv-python")
    all_deps_ok &= check_and_install("rich", "rich")

    return all_deps_ok

def check_tensorflow_dependencies():
    """
    Memeriksa dependensi TensorFlow.

    Returns:
        bool: True jika dependensi TensorFlow berhasil diinstal, False jika tidak.
    """
    return check_and_install("tensorflow", "tensorflow")

def check_native_keras_dependencies():
    """
    Memeriksa dependensi native Keras.

    Returns:
        bool: True jika dependensi native Keras berhasil diinstal, False jika tidak.
    """
    return check_and_install("keras", "keras")
