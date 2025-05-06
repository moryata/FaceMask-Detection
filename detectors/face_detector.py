"""
Modul untuk deteksi wajah menggunakan Haar Cascade.
"""
import cv2

class FaceDetector:
    """
    Kelas untuk mendeteksi wajah dalam frame.
    """

    def __init__(self, face_cascade, min_size=(60, 60), scale_factor=1.1, min_neighbors=5):
        """
        Inisialisasi detektor wajah.

        Args:
            face_cascade: Haar Cascade classifier untuk deteksi wajah
            min_size (tuple): Ukuran minimum wajah yang akan dideteksi
            scale_factor (float): Faktor skala untuk deteksi multi-skala
            min_neighbors (int): Jumlah minimum tetangga untuk deteksi
        """
        self.face_cascade = face_cascade
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect_faces(self, frame):
        """
        Mendeteksi wajah dalam frame.

        Args:
            frame: Frame gambar dari kamera

        Returns:
            list: Daftar koordinat wajah yang terdeteksi [(x, y, w, h), ...]
        """
        # Konversi ke grayscale untuk deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        return faces

    def get_face_roi(self, frame, face_coords):
        """
        Mendapatkan region of interest (ROI) wajah dari frame.

        Args:
            frame: Frame gambar dari kamera
            face_coords (tuple): Koordinat wajah (x, y, w, h)

        Returns:
            tuple: (face_roi, (x1, y1, x2, y2)) di mana face_roi adalah gambar wajah
                  dan (x1, y1, x2, y2) adalah koordinat yang divalidasi
        """
        x, y, w, h = face_coords

        # Validasi koordinat (kadang bisa negatif atau di luar batas)
        y1, y2 = max(0, y), min(frame.shape[0], y + h)
        x1, x2 = max(0, x), min(frame.shape[1], x + w)

        face_roi = frame[y1:y2, x1:x2]

        return face_roi, (x1, y1, x2, y2)
