"""Sanity check Fase 0: model load, 256 kelas, alignment CSV, inferensi 1 gambar."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from detector import FoodDetector  # noqa: E402
from nutrition import NutritionDB  # noqa: E402

print("Loading model ...")
det = FoodDetector()
print(f"  model.names count = {len(det.names)} (harus 256)")
print(f"  contoh: 0={det.names[0]!r}, 255={det.names[255]!r}")

db = NutritionDB()
print(f"  nutrition.csv rows = {len(db.df)}")

# Cek alignment nama model vs CSV
mismatch = [cid for cid in range(len(det.names))
            if db.lookup(cid) and db.lookup(cid)["name"] != det.names[cid]]
print(f"  mismatch nama model vs CSV = {len(mismatch)}")
if mismatch[:5]:
    for cid in mismatch[:5]:
        print(f"    cid {cid}: model={det.names[cid]!r} csv={db.lookup(cid)['name']!r}")

# Ambil 1 gambar test mana saja
test_dir = ROOT / "yolo_uec_food256" / "images" / "test"
imgs = sorted(test_dir.glob("*.jpg"))[:1] or sorted(test_dir.glob("*.*"))[:1]
if not imgs:
    print("  (tidak ada gambar test ditemukan)")
    sys.exit(0)
img_path = imgs[0]
print(f"Inferensi pada: {img_path.name}")
rgb = np.array(Image.open(img_path).convert("RGB"))
dets = det.predict(rgb, conf=0.25)
print(f"  deteksi: {len(dets)} objek")
for d in dets[:10]:
    nut = db.nutrition_for(d["class_id"], db.default_grams(d["class_id"]))
    kcal = nut["kcal"] if nut else "?"
    print(f"    {d['name']:<28} conf={d['conf']:.2f}  kcal(porsi std)={kcal}")
print("OK.")
