# 🍱 NutriScan — Food Calorie & Nutrition Detector

Aplikasi web yang mendeteksi makanan dari foto lalu mengestimasi
**kalori & nutrisi**-nya. Menggabungkan **deep learning** (YOLOv8) dengan
**Computer Vision klasik** (materi kuliah) dalam satu pipeline yang utuh.

**Arsitektur:** frontend **React (Vite + Tailwind)** di folder `../NutriScanFrontendUI`
+ backend **FastAPI** (`app/api.py`) yang membungkus pipeline CV. (Versi Streamlit lama
`app/app.py` masih ada sebagai arsip, tidak dipakai lagi.)

> **Ide besar project ini:**
> | Pertanyaan | Dijawab oleh |
> |---|---|
> | Makanan **apa**? | YOLOv8 (dilatih pada UEC-Food-256, 256 kelas) |
> | **Berapa banyak**? | Classical CV: segmentasi area + objek referensi + homografi |
> | Jadi **berapa kalori**? | Tabel nutrisi (`data/nutrition.csv`) |

---

## 1. Cara menjalankan (2 server)
**Terminal A — backend (FastAPI, port 8000):**
```bash
pip install -r requirements.txt
uvicorn api:app --app-dir app --port 8000 --reload
```
**Terminal B — frontend (React/Vite, port 5173):**
```bash
cd ../NutriScanFrontendUI
npm install
npm run dev
```
Lalu buka **http://localhost:5173**. Frontend mem-proxy `/api/*` ke backend `:8000`
(diatur di `vite.config.ts`), jadi tidak ada masalah CORS.

Model: `Model/best.pt` (YOLOv8s, sudah dilatih — mAP50 ≈ 0.78, mAP50-95 ≈ 0.62).

---

## 2. Cara pakai
1. Di sidebar pilih sumber gambar: **Upload** atau **Webcam**.
2. (Opsional) atur **preprocessing** (denoise / sharpening) & **confidence/IoU**.
3. (Opsional, untuk kalori ter-skala) centang **"Foto memuat kartu referensi"**:
   - Klik **Unduh kartu referensi**, **cetak pada ukuran 8.56 × 5.40 cm** (seukuran KTP/kartu kredit).
   - Letakkan kartu **di samping makanan**, lalu foto dari atas. Kartu harus tampak
     **utuh, besar, tajam, dan tidak terlalu miring**.
4. Lihat hasil di 4 tab:
   - **🍽️ Deteksi & Nutrisi** — kotak + label, tabel per-item, total kalori/makro.
   - **🧪 Filtering & Edge** — demo konvolusi, Sobel/Canny, segmentasi area.
   - **🔑 Fitur & NMS** — Harris/Shi-Tomasi/ORB, invariance, NMS before/after.
   - **📐 Estimasi Porsi (Homografi)** — kartu terdeteksi, bird's-eye rectified, luas cm² → gram.

---

## 3. Struktur project
```
app/
  app.py            # UI Streamlit + orkestrasi pipeline
  detector.py       # wrapper YOLOv8 (load best.pt, predict, gambar box)
  nutrition.py      # load CSV, map label→kalori/makro, agregasi total
  preprocessing.py  # konvolusi, denoise, sharpen, Sobel, Canny
  segmentation.py   # Otsu + morfologi → mask area makanan (luas piksel)
  portion.py        # ORB + homografi(RANSAC) → px/cm → luas cm² → gram
  features_demo.py  # Harris, Shi-Tomasi, ORB, demo invariance
  nms_demo.py       # NMS from-scratch + visualisasi
data/
  nutrition.csv            # 256 baris (BASE, draft heuristik): kcal/protein/karbo/lemak + koef porsi
  nutrition_overrides.csv  # ~40 makanan demo TERVERIFIKASI + kolom `source` (sitasi USDA/MEXT/TKPI)
  reference_card.png       # kartu referensi bertekstur (untuk dicetak)
tools/
  generate_nutrition_csv.py    # generator draft nutrition.csv (BASE)
  build_nutrition_overrides.py # bangun overrides terverifikasi + validasi makro
  sanity_check.py              # cek model load + alignment 256 kelas
Model/best.pt       # model YOLOv8s hasil training (Kaggle)
```

---

## 4. Pemetaan materi kuliah → implementasi
| Sub-topik kelas | File | Fungsi / API OpenCV |
|---|---|---|
| Convolution & correlation | `preprocessing.py` | `cv2.filter2D` + kernel manual (`KERNELS`) |
| Noise reduction | `preprocessing.py` | `cv2.GaussianBlur`, `cv2.medianBlur` |
| Smoothing & sharpening | `preprocessing.py` | unsharp masking |
| Edge detection | `preprocessing.py`, `segmentation.py` | `cv2.Sobel`, `cv2.Canny` |
| Feature concept & invariance | `features_demo.py` | matching ORB pada citra rotasi+skala |
| Harris & Shi–Tomasi detectors | `features_demo.py` | `cv2.cornerHarris`, `cv2.goodFeaturesToTrack` |
| SIFT/SURF/**ORB** descriptors | `portion.py`, `features_demo.py` | `cv2.ORB_create`, `BFMatcher` + Lowe ratio |
| Non-max suppression | `nms_demo.py` + YOLO | NMS from-scratch (greedy IoU) + NMS internal YOLO |
| Affine & projective transforms | `portion.py` | `cv2.warpPerspective` (bird's-eye rectify) |
| Homography estimation with **RANSAC** | `portion.py` | `cv2.findHomography(src, dst, cv2.RANSAC)` |

---

## 5. Bagaimana kalori dihitung
```
foto → preprocessing → YOLOv8 → (label, box, conf)
                                    │
        ┌───────────────────────────┼────────────────────────────┐
        ▼                           ▼                            ▼
 segmentasi area            skala px→cm (objek referensi:     lookup nutrisi
 (Otsu+morfologi)           ORB + homografi RANSAC)           (CSV per 100 g)
 → luas piksel              → px_per_cm                       → kcal/100g, makro
        └────────► luas cm² = luas_px / px_per_cm² ───► gram = luas_cm² × koef
                                                         kkal = gram × kcal_per_100g / 100
```
- **Dengan kartu referensi** → kalori **ter-skala** dengan ukuran porsi nyata.
- **Tanpa kartu** → fallback ke **porsi standar** (`typical_serving_g` di CSV).

### Dari mana nilai gizi (per 100 g)?
Dua lapis (di-merge di `nutrition.py` berdasarkan `class_id`):
1. **BASE — `data/nutrition.csv`**: draft otomatis (14 profil per kata kunci, `tools/generate_nutrition_csv.py`).
   Cakupan 256 kelas tapi kasar (semua makanan satu profil dapat angka sama).
2. **OVERRIDE — `data/nutrition_overrides.csv`**: ~40 makanan demo dengan nilai **terverifikasi dari
   sumber resmi** (USDA FoodData Central, MEXT Japan, TKPI Kemenkes, FatSecret) + **sitasi** di kolom
   `source`. Override **menimpa** baris base.

Di aplikasi, kolom **"Sumber gizi"** menandai tiap makanan: **✔️ + sitasi** (terverifikasi) atau
**"draft (estimasi)"**. Jadi saat demo jelas mana yang bersumber resmi.

---

## 6. Keterbatasan (jujurkan di laporan)
- Estimasi kalori dari **satu foto** bersifat **aproksimasi**: metode area mengasumsikan
  makanan relatif datar (tidak menghitung tinggi/volume).
- Skala metrik akurat **membutuhkan kartu referensi** yang tampak besar, tajam, dan
  cukup datar; jika tidak terdeteksi, aplikasi memakai porsi standar.
- Konversi **luas → massa** memakai koefisien per-kategori (`area_to_gram_coeff`) yang
  masih kasar — perlu kalibrasi lebih lanjut.
- Nilai gizi **~40 makanan demo sudah terverifikasi** dari sumber resmi (`data/nutrition_overrides.csv`,
  kolom `source`). **Sisa ~216 kelas** masih **draft otomatis** (`data/nutrition.csv`) — koreksi bila perlu.
- Nilai per-100 g adalah **rata-rata** "as-served"; porsi & resep nyata bisa berbeda.

## 7. Pengembangan lanjut (opsional / bonus)
- **Volume via stereo** (2 foto): epipolar, 8-point, fundamental/essential matrix,
  disparity→depth, triangulasi → kalori lebih akurat (kelas "two-view geometry"/"stereo").
- **Kalibrasi kamera** (Zhang's method) + koreksi distorsi lensa sebelum homografi.
- **Optical flow** (Lucas–Kanade) untuk input video multi-frame.
