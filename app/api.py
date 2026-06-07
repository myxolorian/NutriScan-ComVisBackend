"""FastAPI backend untuk NutriScan — membungkus pipeline CV jadi HTTP API.

Jalankan dari root project:
    uvicorn api:app --app-dir app --port 8000 --reload

Endpoint:
    GET  /api/health  -> status + jumlah kelas
    POST /api/analyze -> multipart(file + params) -> JSON lengkap utk 4 tab
"""
import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import features_demo as feat          # noqa: E402
import fruit as fruitmod              # noqa: E402  (fitur tambahan: tracking buah)
import nms_demo as nmsd               # noqa: E402
import portion as por                 # noqa: E402
import preprocessing as pre           # noqa: E402
import segmentation as seg            # noqa: E402
from detector import FoodDetector, draw_detections   # noqa: E402
from nutrition import NutritionDB                     # noqa: E402

# ----------------------------- load resources sekali -----------------------------
detector = FoodDetector()
db = NutritionDB()

# Map nama kernel di UI -> kunci di preprocessing.KERNELS
KERNEL_MAP = {"Blur": "box blur", "Sharpen": "sharpen",
              "Edge Detect": "laplacian", "Emboss": "emboss"}
DISP_MAX = 800  # ukuran maks untuk gambar overlay (display only)

app = FastAPI(title="NutriScan API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


# ----------------------------- helpers -----------------------------
def _resize_max(rgb, max_dim):
    h, w = rgb.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return rgb


def to_data_url(rgb, max_dim=DISP_MAX, quality=82):
    """numpy RGB -> data URL JPEG base64 (di-downscale utk payload ringan)."""
    rgb = _resize_max(rgb, max_dim)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def preprocess(rgb, denoise, denoise_k, do_sharpen, sharpen_amt):
    proc = rgb
    if denoise == "gaussian":
        proc = pre.gaussian_blur(proc, denoise_k)
    elif denoise == "median":
        proc = pre.median_blur(proc, denoise_k)
    if do_sharpen:
        proc = pre.unsharp_sharpen(proc, sharpen_amt, denoise_k)
    return proc


# ----------------------------- endpoints -----------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "n_classes": len(detector.names)}


# ----------------------------- fitur tambahan: tracking buah (webcam) -----------
@app.get("/api/track-stream")
def track_stream():
    """Stream MJPEG webcam dgn bounding box buah (ByteTrack). Pipeline terpisah."""
    return StreamingResponse(
        fruitmod.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/track-status")
def track_status():
    """Buah yang sedang terlihat + FPS (utk panel live di frontend)."""
    return fruitmod.latest_status()


@app.get("/api/fruit-info")
def fruit_info():
    """Tabel referensi gizi 3 buah (apel/pisang/jeruk)."""
    return {"fruits": fruitmod.fruit_info()}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    denoise: str = Form("none"),
    denoise_k: int = Form(5),
    do_sharpen: bool = Form(False),
    sharpen_amt: float = Form(1.0),
):
    raw = await file.read()
    rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    H, W = rgb.shape[:2]
    proc = preprocess(rgb, denoise, denoise_k, do_sharpen, sharpen_amt)

    # ---- deteksi (multi-plate: semua makanan yang terdeteksi) ----
    dets = detector.predict(proc, conf=conf, iou=iou)
    dets.sort(key=lambda d: d["conf"], reverse=True)
    image_area = H * W

    # ---- segmentasi + estimasi porsi per item ----
    detections_out = []
    seg_masks = []
    totals = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

    for det in dets:
        cid = det["class_id"]
        s = seg.segment_food_in_box(proc, det["xyxy"])
        seg_masks.append(s["mask_full"])
        area_px = int(s["area_px"])

        # Estimasi porsi dari rasio area
        portion = por.estimate_portion_from_area(
            area_px=area_px,
            image_area=image_area,
            food_profile=db.food_profile(cid),
            typical_serving_g=db.default_grams(cid),
        )
        grams = portion["grams"]
        nut = db.nutrition_for(cid, grams)

        detections_out.append({
            "name": nut["name"], "class_id": cid,
            "conf": round(float(det["conf"]), 3), "xyxy": det["xyxy"],
            "grams": round(float(grams)), "kcal": nut["kcal"],
            "protein_g": nut["protein_g"], "carbs_g": nut["carbs_g"],
            "fat_g": nut["fat_g"],
            "area_px": area_px, "area_cm2": None,
            "source": nut["source"], "verified": nut["verified"],
            "portion_source": f"estimasi area ({portion['size_label']})",
            "size_label": portion["size_label"],
            "multiplier": portion["multiplier"],
            "area_ratio": portion["area_ratio"],
        })
        for k in totals:
            totals[k] += nut[k]

    # ---- fitur tambahan: deteksi buah (apel/pisang/jeruk) via COCO ----
    # Membuat upload image & foto webcam juga mengenali buah (model COCO terpisah).
    for fd in fruitmod.detect_fruits(proc, conf=conf):
        s = seg.segment_food_in_box(proc, fd["xyxy"])
        seg_masks.append(s["mask_full"])
        area_px = int(s["area_px"])
        serving = fruitmod.FRUIT_NUTRITION[fd["name"]]["serving_g"]
        portion = por.estimate_portion_from_area(
            area_px=area_px, image_area=image_area,
            food_profile="default", typical_serving_g=serving)
        grams = portion["grams"]
        nut = fruitmod.fruit_nutrition_for(fd["name"], grams)
        detections_out.append({
            "name": nut["name"], "class_id": fd["class_id"],
            "conf": round(float(fd["conf"]), 3), "xyxy": fd["xyxy"],
            "grams": round(float(grams)), "kcal": nut["kcal"],
            "protein_g": nut["protein_g"], "carbs_g": nut["carbs_g"],
            "fat_g": nut["fat_g"],
            "area_px": area_px, "area_cm2": None,
            "source": nut["source"], "verified": nut["verified"],
            "portion_source": f"estimasi area ({portion['size_label']})",
            "size_label": portion["size_label"],
            "multiplier": portion["multiplier"],
            "area_ratio": portion["area_ratio"],
        })
        for k in totals:
            totals[k] += nut[k]

    # ---- gambar overlay (pakai salinan di-downscale supaya cepat & ringan) ----
    disp = _resize_max(rgb, DISP_MAX)
    disp_proc = _resize_max(proc, DISP_MAX)
    images = {
        "preprocessed": to_data_url(proc),
        "sobel": to_data_url(pre.sobel_edges(disp)),
        "canny": to_data_url(pre.canny_edges(disp)),
        "kernels": {ui: to_data_url(pre.apply_kernel(disp, pre.KERNELS[k]))
                    for ui, k in KERNEL_MAP.items()},
    }
    h_img, h_n = feat.harris_corners(disp)
    st_img, st_n = feat.shi_tomasi(disp)
    orb_img, orb_n = feat.orb_keypoints(disp)
    images["harris"], images["shi_tomasi"], images["orb"] = (
        to_data_url(h_img), to_data_url(st_img), to_data_url(orb_img))
    # Invariance / augmentasi: cocokkan citra dgn versi rotasi+skala-nya (ORB).
    match_img, match_n = feat.orb_match_invariance(disp)
    images["orb_match"] = to_data_url(match_img, max_dim=1100)
    # Segmentasi area makanan (overlay merah) — visualisasi materi segmentasi.
    images["segmentation"] = (to_data_url(seg.overlay_masks(rgb, seg_masks))
                              if seg_masks else None)

    # ---- NMS demo ----
    before = detector.predict(disp_proc, conf=max(0.05, conf - 0.1), iou=0.9, max_det=300)
    if len(before) <= 1 and before:
        before = nmsd.make_synthetic_overlaps(before)
    kept, _ = nmsd.nms_on_detections(before, iou)
    images["nms_before"] = to_data_url(draw_detections(disp, before))
    images["nms_after"] = to_data_url(draw_detections(disp, kept))

    counts = {"harris": int(h_n), "shi_tomasi": int(st_n), "orb": int(orb_n),
              "orb_match": int(match_n),
              "nms_before": len(before), "nms_after": len(kept)}

    return JSONResponse({
        "image_w": W, "image_h": H,
        "detections": detections_out,
        "totals": {k: round(v, 1) for k, v in totals.items()},
        "images": images,
        "counts": counts,
    })
