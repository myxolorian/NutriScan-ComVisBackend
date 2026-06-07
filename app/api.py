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
from fastapi.responses import JSONResponse
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import features_demo as feat          # noqa: E402
import nms_demo as nmsd               # noqa: E402
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

    # ---- deteksi (single dish: confidence tertinggi) ----
    dets = detector.predict(proc, conf=conf, iou=iou)
    dets.sort(key=lambda d: d["conf"], reverse=True)
    top = dets[0] if dets else None

    # ---- skala: tidak ada kartu referensi, selalu pakai porsi standar CSV ----
    scale = {"ok": False, "px_per_cm": 0.0, "inliers": 0,
             "reason": "menggunakan porsi standar dari CSV"}

    # ---- deteksi + nutrisi ----
    detection = None
    seg_mask = None
    totals = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    if top is not None:
        s = seg.segment_food_in_box(proc, top["xyxy"])
        seg_mask = s["mask_full"]
        area_px = int(s["area_px"])
        # Gram selalu dari porsi standar CSV (typical_serving_g).
        grams = db.default_grams(top["class_id"])
        portion_source = "porsi standar (CSV)"
        nut = db.nutrition_for(top["class_id"], grams)
        detection = {
            "name": nut["name"], "class_id": top["class_id"],
            "conf": round(float(top["conf"]), 3), "xyxy": top["xyxy"],
            "grams": round(float(grams)), "kcal": nut["kcal"],
            "protein_g": nut["protein_g"], "carbs_g": nut["carbs_g"], "fat_g": nut["fat_g"],
            "area_px": area_px, "area_cm2": None,
            "source": nut["source"], "verified": nut["verified"],
            "portion_source": portion_source,
        }
        totals = {"kcal": nut["kcal"], "protein_g": nut["protein_g"],
                  "carbs_g": nut["carbs_g"], "fat_g": nut["fat_g"]}

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
    images["segmentation"] = (to_data_url(seg.overlay_masks(rgb, [seg_mask]))
                              if seg_mask is not None else None)

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
        "detection": detection,
        "totals": {k: round(v, 1) for k, v in totals.items()},
        "scale": scale,
        "images": images,
        "counts": counts,
    })
