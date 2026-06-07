"""Food Calorie & Nutrition Detector — Streamlit app.

Pipeline: input (upload/webcam) -> preprocessing -> YOLOv8 -> segmentasi area
-> estimasi porsi (ORB+homografi RANSAC, opsional) -> lookup nutrisi -> output.

Jalankan:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Pastikan modul sibling dapat di-import saat dijalankan via `streamlit run`.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import features_demo as feat          # noqa: E402
import nms_demo as nmsd               # noqa: E402
import portion as por                 # noqa: E402
import preprocessing as pre           # noqa: E402
import segmentation as seg            # noqa: E402
from detector import FoodDetector, draw_detections   # noqa: E402
from nutrition import NutritionDB, build_item_rows    # noqa: E402

st.set_page_config(page_title="Food Calorie & Nutrition Detector",
                   page_icon="🍱", layout="wide")


# ----------------------------- Resource loaders -----------------------------
@st.cache_resource(show_spinner="Memuat model YOLOv8 ...")
def load_detector():
    return FoodDetector()


@st.cache_resource(show_spinner="Memuat tabel nutrisi ...")
def load_db():
    return NutritionDB()


detector = load_detector()
db = load_db()


# ----------------------------- Sidebar (kontrol) -----------------------------
st.sidebar.title("⚙️ Pengaturan")

st.sidebar.subheader("Model")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("IoU (NMS internal YOLO)", 0.1, 0.95, 0.45, 0.05)

st.sidebar.subheader("Preprocessing (Materi: Filtering)")
denoise = st.sidebar.selectbox("Noise reduction", ["none", "gaussian", "median"])
denoise_k = st.sidebar.slider("Kernel size denoise", 3, 15, 5, 2)
do_sharpen = st.sidebar.checkbox("Unsharp sharpening")
sharpen_amt = st.sidebar.slider("Sharpen amount", 0.2, 3.0, 1.0, 0.1,
                                disabled=not do_sharpen)

st.sidebar.subheader("Sumber gambar")
source = st.sidebar.radio("Input", ["Upload", "Webcam"], horizontal=True)


# ----------------------------- Ambil gambar -----------------------------
st.title("🍱 Food Calorie & Nutrition Detector")
st.caption("YOLOv8 (UEC-Food-256) = *makanan apa* · Classical CV = *berapa banyak* · "
           "Tabel nutrisi = *berapa kalori*")

if source == "Upload":
    up = st.file_uploader("Unggah foto makanan", type=["jpg", "jpeg", "png", "bmp"])
    raw = up
else:
    raw = st.camera_input("Ambil foto dari webcam")

if raw is None:
    st.info("⬆️ Unggah foto atau ambil dari webcam untuk memulai.")
    st.stop()

image_rgb = np.array(Image.open(raw).convert("RGB"))


# ----------------------------- Preprocessing -----------------------------
proc = image_rgb
if denoise == "gaussian":
    proc = pre.gaussian_blur(proc, denoise_k)
elif denoise == "median":
    proc = pre.median_blur(proc, denoise_k)
if do_sharpen:
    proc = pre.unsharp_sharpen(proc, sharpen_amt, denoise_k)


# ----------------------------- Deteksi -----------------------------
detections = detector.predict(proc, conf=conf, iou=iou)


# ----------------------------- Estimasi porsi -----------------------------
masks = []
grams_by_index = {}

image_area = proc.shape[0] * proc.shape[1]

for i, det in enumerate(detections):
    s = seg.segment_food_in_box(proc, det["xyxy"])
    masks.append(s["mask_full"])
    det["area_px"] = s["area_px"]

    # Area-ratio estimation
    portion = por.estimate_portion_from_area(
        area_px=s["area_px"],
        image_area=image_area,
        food_profile=db.food_profile(det["class_id"]),
        typical_serving_g=db.default_grams(det["class_id"]),
    )
    grams_by_index[i] = portion["grams"]
    det["size_label"] = portion["size_label"]
    det["multiplier"] = portion["multiplier"]
    det["area_ratio"] = portion["area_ratio"]

# Lampirkan kkal & gram ke tiap deteksi (untuk label gambar).
for i, det in enumerate(detections):
    grams = grams_by_index.get(i) or db.default_grams(det["class_id"])
    nut = db.nutrition_for(det["class_id"], grams)
    det["grams"] = round(grams, 0)
    det["kcal"] = nut["kcal"] if nut else 0.0

rows, totals = build_item_rows(detections, db, grams_by_index)


# ----------------------------- Output (Tabs) -----------------------------
tab_main, tab_filter, tab_feat, tab_portion = st.tabs(
    ["🍽️ Deteksi & Nutrisi", "🧪 Filtering & Edge", "🔑 Fitur & NMS",
     "📐 Estimasi Porsi (Area)"])

with tab_main:
    c1, c2 = st.columns([3, 2])
    with c1:
        annotated = draw_detections(
            image_rgb, detections,
            label_fn=lambda d: f"{d['name']} {d['conf']:.2f} | {d['kcal']:.0f} kkal")
        st.image(annotated, caption=f"{len(detections)} makanan terdeteksi",
                 use_container_width=True)
    with c2:
        st.info("Porsi diestimasi dari rasio area piksel segmentasi terhadap gambar keseluruhan.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Kalori", f"{totals['kcal']:.0f} kkal")
        m2.metric("Protein", f"{totals['protein_g']:.0f} g")
        m3.metric("Karbo", f"{totals['carbs_g']:.0f} g")
        m4.metric("Lemak", f"{totals['fat_g']:.0f} g")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        n_verified = sum(db.is_verified(d["class_id"]) for d in detections)
        if n_verified:
            st.caption(f"✔️ {n_verified}/{len(detections)} makanan memakai **nilai gizi "
                       f"terverifikasi** (USDA/MEXT/TKPI — lihat kolom *Sumber gizi*); "
                       "sisanya draft estimasi.")
    else:
        st.info("Tidak ada makanan terdeteksi. Turunkan confidence threshold.")
    st.caption("⚠️ Estimasi kalori bersifat aproksimasi (lihat keterbatasan di README).")

with tab_filter:
    st.subheader("Materi: Convolution, Noise reduction, Smoothing/Sharpening, Edge")
    c1, c2 = st.columns(2)
    c1.image(image_rgb, caption="Asli", use_container_width=True)
    c2.image(proc, caption="Setelah preprocessing (dipakai untuk deteksi)",
             use_container_width=True)
    st.markdown("**Konvolusi manual (cv2.filter2D)** — pilih kernel:")
    kname = st.selectbox("Kernel 3×3", list(pre.KERNELS.keys()), index=2)
    c3, c4 = st.columns(2)
    c3.code(np.array2string(pre.KERNELS[kname]), language="text")
    c4.image(pre.apply_kernel(image_rgb, pre.KERNELS[kname]),
             caption=f"Hasil konvolusi: {kname}", use_container_width=True)
    st.markdown("**Deteksi tepi**")
    c5, c6 = st.columns(2)
    c5.image(pre.sobel_edges(image_rgb), caption="Sobel (magnitudo gradien)",
             use_container_width=True)
    t1, t2 = st.slider("Threshold Canny", 0, 400, (100, 200))
    c6.image(pre.canny_edges(image_rgb, t1, t2), caption="Canny",
             use_container_width=True)
    st.markdown("**Segmentasi area makanan** (threshold Otsu + morfologi) — dasar estimasi porsi")
    if masks:
        st.image(seg.overlay_masks(image_rgb, masks),
                 caption="Area makanan tersegmentasi (merah)", use_container_width=True)
    else:
        st.info("Belum ada deteksi untuk disegmentasi.")

with tab_feat:
    st.subheader("Materi: Harris, Shi-Tomasi, ORB, invariance, Non-Max Suppression")
    c1, c2, c3 = st.columns(3)
    h_img, h_n = feat.harris_corners(image_rgb)
    c1.image(h_img, caption=f"Harris corners (~{h_n} px)", use_container_width=True)
    st_img, st_n = feat.shi_tomasi(image_rgb)
    c2.image(st_img, caption=f"Shi-Tomasi ({st_n} sudut)", use_container_width=True)
    orb_img, orb_n = feat.orb_keypoints(image_rgb)
    c3.image(orb_img, caption=f"ORB keypoints ({orb_n})", use_container_width=True)
    st.markdown("**Invariance** — ORB tetap cocok meski gambar dirotasi & diskala:")
    mv, nmatch = feat.orb_match_invariance(image_rgb)
    st.image(mv, caption=f"Matching kiri↔kanan (rotasi+skala): {nmatch} match",
             use_container_width=True)

    st.divider()
    st.subheader("Non-Maximum Suppression (implementasi from-scratch)")
    nms_iou = st.slider("IoU threshold NMS (demo)", 0.1, 0.9, 0.45, 0.05)
    before = detector.predict(proc, conf=max(0.05, conf - 0.1), iou=0.9, max_det=300)
    if len(before) <= 1 and detections:
        before = nmsd.make_synthetic_overlaps(detections)
        st.caption("ℹ️ Gambar bersih — ditampilkan kotak tumpang-tindih sintetis untuk demonstrasi.")
    kept, removed = nmsd.nms_on_detections(before, nms_iou)
    cc1, cc2 = st.columns(2)
    cc1.image(draw_detections(image_rgb, before),
              caption=f"Sebelum NMS: {len(before)} kotak", use_container_width=True)
    cc2.image(draw_detections(image_rgb, kept),
              caption=f"Sesudah NMS: {len(kept)} kotak ({len(removed)} dibuang)",
              use_container_width=True)

with tab_portion:
    st.subheader("Estimasi Porsi (Area Ratio)")
    st.info("Porsi diestimasi dari rasio area segmentasi terhadap gambar.")
    
    area_rows = []
    for det in detections:
        area_rows.append({
            "Makanan": det["name"], 
            "Area (px)": f"{det['area_px']:,}",
            "Area Ratio": f"{det['area_ratio']*100:.1f}%",
            "Ukuran": det["size_label"], 
            "Multiplier": f"×{det['multiplier']:.2f}",
            "Massa est. (g)": det["grams"],
            "Kalori (kkal)": round(det["kcal"], 0),
        })
    if area_rows:
        st.dataframe(pd.DataFrame(area_rows), use_container_width=True, hide_index=True)
