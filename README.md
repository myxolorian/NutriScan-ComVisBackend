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

