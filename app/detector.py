"""YOLOv8 wrapper untuk Food Calorie Detector.

Konvensi gambar di seluruh app: numpy array RGB uint8 (HxWx3).
ultralytics menerima numpy BGR, jadi konversi dilakukan di sini sebelum predict.
"""
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Model" / "best.pt"

# Warna kotak (RGB) — dipakai untuk anotasi.
_BOX_COLOR = (0, 200, 0)
_TEXT_COLOR = (255, 255, 255)


class FoodDetector:
    """Memuat best.pt sekali, lalu predict() mengembalikan list deteksi rapi."""

    def __init__(self, model_path=None):
        from ultralytics import YOLO  # import lokal supaya import modul ini ringan
        self.model_path = str(model_path or DEFAULT_MODEL_PATH)
        self.model = YOLO(self.model_path)
        # model.names: dict {class_id: name} — sumber kebenaran 256 kelas.
        self.names = self.model.names

    def predict(self, image_rgb, conf=0.25, iou=0.45, max_det=300):
        """Jalankan deteksi pada satu gambar RGB.

        Returns: list[dict] dengan key: class_id, name, conf, xyxy ([x1,y1,x2,y2] int).
        """
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(source=bgr, conf=conf, iou=iou,
                                     max_det=max_det, verbose=False)
        r = results[0]
        detections = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                cls_id = int(b.cls[0].item())
                detections.append({
                    "class_id": cls_id,
                    "name": self.names.get(cls_id, str(cls_id)),
                    "conf": float(b.conf[0].item()),
                    "xyxy": [x1, y1, x2, y2],
                })
        return detections


def draw_detections(image_rgb, detections, label_fn=None):
    """Gambar bounding box + label pada salinan gambar RGB.

    label_fn(det) -> str opsional untuk mengkustom teks label (mis. tambah kkal).
    """
    img = image_rgb.copy()
    h = img.shape[0]
    scale = max(0.5, min(1.2, h / 800.0))
    thickness = max(1, int(round(2 * scale)))
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cv2.rectangle(img, (x1, y1), (x2, y2), _BOX_COLOR, thickness)
        if label_fn is not None:
            text = label_fn(det)
        else:
            text = f"{det['name']} {det['conf']:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        ytop = max(0, y1 - th - bl)
        cv2.rectangle(img, (x1, ytop), (x1 + tw, ytop + th + bl), _BOX_COLOR, -1)
        cv2.putText(img, text, (x1, ytop + th), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, _TEXT_COLOR, thickness, cv2.LINE_AA)
    return img
