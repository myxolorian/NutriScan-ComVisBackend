
import time

import cv2

# COCO class id -> (nama Indonesia, warna kotak BGR)
FRUIT_CLASSES = {
    47: ("Apel", (60, 60, 220)),     # apple  -> merah
    46: ("Pisang", (40, 200, 230)),  # banana -> kuning
    49: ("Jeruk", (40, 140, 240)),   # orange -> oranye
}
FRUIT_IDS = list(FRUIT_CLASSES.keys())

# Gizi per 100 g (USDA FoodData Central, buah mentah) + porsi tipikal 1 buah.
FRUIT_NUTRITION = {
    "Apel":   {"kcal_100g": 52, "protein_g": 0.3, "carbs_g": 14.0, "fat_g": 0.2,
               "serving_g": 182, "source": "USDA FDC 171688"},
    "Pisang": {"kcal_100g": 89, "protein_g": 1.1, "carbs_g": 23.0, "fat_g": 0.3,
               "serving_g": 118, "source": "USDA FDC 173944"},
    "Jeruk":  {"kcal_100g": 47, "protein_g": 0.9, "carbs_g": 12.0, "fat_g": 0.1,
               "serving_g": 131, "source": "USDA FDC 169097"},
}

_MODEL = None

INFER_SIZE = 320

_latest = {"fruits": [], "fps": 0.0}


def _get_model():
    
    global _MODEL
    if _MODEL is None:
        from pathlib import Path
        from ultralytics import YOLO
        ov_dir = Path(__file__).resolve().parent.parent / "yolov8n_openvino_model"
        if ov_dir.exists():
            _MODEL = YOLO(str(ov_dir), task="detect")
        else:
            base = YOLO("yolov8n.pt")
            try:
                base.export(format="openvino", imgsz=INFER_SIZE)  
                _MODEL = YOLO(str(ov_dir), task="detect")
            except Exception:
                _MODEL = base  
    return _MODEL


def fruit_info():
    
    return [{"name": name, **vals, "kcal_serving": round(vals["kcal_100g"] * vals["serving_g"] / 100)} for name, vals in FRUIT_NUTRITION.items()]


def latest_status():
    
    return _latest


def fruit_nutrition_for(name, grams):
    v = FRUIT_NUTRITION[name]
    f = grams / 100.0
    return {
        "name": name, "grams": round(grams, 1),
        "kcal": round(v["kcal_100g"] * f, 1),
        "protein_g": round(v["protein_g"] * f, 1),
        "carbs_g": round(v["carbs_g"] * f, 1),
        "fat_g": round(v["fat_g"] * f, 1),
        "verified": True, "source": v["source"],
        "serving_g": v["serving_g"],
    }


def detect_fruits(rgb, conf=0.35):
   
    model = _get_model()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    res = model.predict(bgr, classes=FRUIT_IDS, conf=conf,
                        imgsz=INFER_SIZE, verbose=False)
    r = res[0]
    out = []
    if r.boxes is not None:
        for b in r.boxes:
            cid = int(b.cls[0])
            name, _ = FRUIT_CLASSES.get(cid, ("?", None))
            x1, y1, x2, y2 = b.xyxy[0].int().tolist()
            out.append({"class_id": 1000 + cid, "name": name,
                        "conf": float(b.conf[0]), "xyxy": [x1, y1, x2, y2]})
    return out


def generate_frames(cam_index=0, conf=0.35):
    
    model = _get_model()
    
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    first = True
    prev = time.time()
    try:
        if not cap.isOpened():
            _latest["fruits"], _latest["fps"] = [], 0.0
            return
        while True:
            ok, frame = cap.read()
            if not ok:
                break
           
            res = model.track(frame, persist=not first, classes=FRUIT_IDS,
                              conf=conf, imgsz=INFER_SIZE, tracker="bytetrack.yaml",
                              verbose=False)
            first = False
            r = res[0]
            fruits = []
            if r.boxes is not None:
                for b in r.boxes:
                    cid = int(b.cls[0])
                    name, color = FRUIT_CLASSES.get(cid, ("?", (200, 200, 200)))
                    tid = int(b.id[0]) if b.id is not None else -1
                    x1, y1, x2, y2 = b.xyxy[0].int().tolist()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label = f"{name} #{tid}" if tid >= 0 else name
                    (tw, th), bl = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - th - bl - 4),
                                  (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    fruits.append({"name": name, "track_id": tid,
                                   "conf": round(float(b.conf[0]), 2)})
            _latest["fruits"] = fruits

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev)
            prev = now
            _latest["fps"] = round(fps, 1)

            ok2, buf = cv2.imencode(".jpg", frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok2:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
    finally:
        cap.release()
        _latest["fruits"], _latest["fps"] = [], 0.0
