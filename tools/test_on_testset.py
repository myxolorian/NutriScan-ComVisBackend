"""Test model pada test set — visualisasi deteksi beberapa gambar."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from detector import FoodDetector, draw_detections  # noqa: E402
from nutrition import NutritionDB, build_item_rows  # noqa: E402

det = FoodDetector()
db = NutritionDB()
test_dir = ROOT / "yolo_uec_food256" / "images" / "test"
images = sorted(test_dir.glob("*.jpg"))[:20]  # 20 gambar pertama

print(f"Testing on {len(images)} images from {test_dir}")
print("-" * 80)

total_dets = 0
for i, img_path in enumerate(images):
    rgb = np.array(Image.open(img_path).convert("RGB"))
    dets = det.predict(rgb, conf=0.25, iou=0.45)
    total_dets += len(dets)

    rows, totals = build_item_rows(dets, db)

    print(f"\n{i+1}. {img_path.name}")
    print(f"   Deteksi: {len(dets)} objek | Total kalori: {totals['kcal']:.0f} kkal")
    for j, row in enumerate(rows[:3]):  # tampilkan 3 top items
        print(f"     {j+1}) {row['Makanan']:<30} conf={row['Conf']:.2f} "
              f"porsi={row['Porsi (g)']:.0f}g kcal={row['Kalori (kkal)']:.0f}")
    if len(rows) > 3:
        print(f"     ... (+{len(rows)-3} lainnya)")

print("\n" + "=" * 80)
print(f"Summary: {len(images)} gambar, rata-rata {total_dets/len(images):.1f} deteksi/gambar")
print(f"Total deteksi: {total_dets}")
