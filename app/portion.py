"""Estimasi Porsi berdasarkan rasio area.

Estimasi porsi menggunakan rasio luas segmentasi makanan terhadap luas gambar keseluruhan.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Area-Ratio Portion Estimation
# ---------------------------------------------------------------------------
# Rasio luas segmentasi makanan terhadap luas gambar → size bucket → multiplier.
# Threshold per food profile karena proporsi visual bervariasi antar jenis.

_PROFILE_THRESHOLDS: dict[str, list[float]] = {
    # Format: [batas_xs, batas_s, batas_m, batas_l] — di atas batas_l = XL.
    "rice":     [0.02, 0.05, 0.12, 0.22],
    "noodle":   [0.02, 0.05, 0.12, 0.22],
    "meat":     [0.015, 0.04, 0.10, 0.18],
    "seafood":  [0.015, 0.04, 0.10, 0.18],
    "fried":    [0.015, 0.04, 0.10, 0.18],
    "soup":     [0.03, 0.06, 0.14, 0.25],
    "veg":      [0.01, 0.03, 0.08, 0.15],
    "egg_tofu": [0.01, 0.03, 0.08, 0.15],
    "handheld": [0.01, 0.03, 0.08, 0.15],
    "dessert":  [0.01, 0.03, 0.08, 0.15],
    "dumpling": [0.01, 0.03, 0.08, 0.15],
    "pizza":    [0.02, 0.05, 0.12, 0.22],
    "curry":    [0.02, 0.05, 0.12, 0.22],
    "default":  [0.02, 0.05, 0.12, 0.20],
}

_SIZE_BUCKETS: list[tuple[str, float]] = [
    ("XS", 0.50),
    ("S",  0.75),
    ("M",  1.00),
    ("L",  1.35),
    ("XL", 1.70),
]


def estimate_portion_from_area(area_px, image_area, food_profile,
                               typical_serving_g):
    """Estimasi porsi berdasarkan rasio area makanan terhadap luas gambar.

    Args:
        area_px:            luas piksel makanan (dari segmentasi Otsu).
        image_area:         luas total gambar (width × height).
        food_profile:       profil makanan dari CSV (rice, meat, …).
        typical_serving_g:  porsi standar dari CSV.

    Returns dict:
        grams       : estimasi berat porsi (g).
        size_label  : label ukuran (XS / S / M / L / XL).
        multiplier  : faktor pengali terhadap porsi standar.
        area_ratio  : rasio area makanan terhadap gambar.
    """
    if image_area <= 0 or area_px <= 0:
        return {"grams": float(typical_serving_g), "size_label": "M",
                "multiplier": 1.0, "area_ratio": 0.0}

    area_ratio = area_px / image_area
    thresholds = _PROFILE_THRESHOLDS.get(
        food_profile, _PROFILE_THRESHOLDS["default"])

    bucket_idx = len(thresholds)          # default ke XL (terakhir)
    for i, t in enumerate(thresholds):
        if area_ratio <= t:
            bucket_idx = i
            break

    label, multiplier = _SIZE_BUCKETS[bucket_idx]
    grams = typical_serving_g * multiplier

    return {"grams": round(float(grams), 1), "size_label": label,
            "multiplier": multiplier, "area_ratio": round(float(area_ratio), 4)}
