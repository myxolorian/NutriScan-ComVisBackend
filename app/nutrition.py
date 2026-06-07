"""Lookup nutrisi: petakan deteksi YOLO -> kkal & makro, lalu agregasi 1 piring.

Sumber data:
- BASE  : data/nutrition.csv          (draft heuristik, 256 baris; lihat tools/generate_nutrition_csv.py)
- OVERRIDE: data/nutrition_overrides.csv (nilai TERVERIFIKASI utk makanan demo + sitasi `source`;
            lihat tools/build_nutrition_overrides.py). Menimpa baris base by class_id.
Pemetaan utama via class_id (0-based), cocok dgn model.names.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "nutrition.csv"
DEFAULT_OVERRIDES = PROJECT_ROOT / "data" / "nutrition_overrides.csv"

# Kolom gizi yang ditimpa oleh override (area_to_gram_coeff TETAP dari base).
_OVERRIDE_COLS = ["kcal_per_100g", "protein_g", "carbs_g", "fat_g",
                  "typical_serving_g", "density_g_per_cm3", "source"]


class NutritionDB:
    def __init__(self, csv_path=None, overrides_path=None):
        df = pd.read_csv(csv_path or DEFAULT_CSV)
        # cast kolom numerik ke float supaya update override (float) tidak bentrok dtype
        for c in ["kcal_per_100g", "protein_g", "carbs_g", "fat_g", "typical_serving_g"]:
            if c in df.columns:
                df[c] = df[c].astype(float)
        # kolom default untuk semua baris
        df["source"] = "draft auto-estimate"
        df["verified"] = False
        if "density_g_per_cm3" not in df.columns:
            df["density_g_per_cm3"] = pd.NA
        df.set_index("class_id", inplace=True, drop=False)

        # Merge override (menimpa baris yg cocok by class_id).
        ov_path = Path(overrides_path or DEFAULT_OVERRIDES)
        if ov_path.exists():
            ov = pd.read_csv(ov_path).set_index("class_id", drop=False)
            cols = [c for c in _OVERRIDE_COLS if c in ov.columns]
            df.update(ov[cols])                       # update nilai gizi
            df.loc[df.index.isin(ov.index), "verified"] = True
        self.df = df

    def lookup(self, class_id):
        """Return dict baris nutrisi untuk class_id, atau None bila tak ada."""
        if class_id in self.df.index:
            return self.df.loc[class_id].to_dict()
        return None

    def is_verified(self, class_id):
        row = self.lookup(class_id)
        return bool(row["verified"]) if row else False

    def source_of(self, class_id):
        row = self.lookup(class_id)
        return str(row["source"]) if row else ""

    def nutrition_for(self, class_id, grams):
        """Hitung gizi untuk `grams` gram makanan kategori class_id.

        Returns dict: name, grams, kcal, protein_g, carbs_g, fat_g, verified, source.
        """
        row = self.lookup(class_id)
        if row is None:
            return None
        factor = grams / 100.0
        return {
            "name": row["name"],
            "grams": round(grams, 1),
            "kcal": round(row["kcal_per_100g"] * factor, 1),
            "protein_g": round(row["protein_g"] * factor, 1),
            "carbs_g": round(row["carbs_g"] * factor, 1),
            "fat_g": round(row["fat_g"] * factor, 1),
            "verified": bool(row["verified"]),
            "source": str(row["source"]),
        }

    def default_grams(self, class_id):
        row = self.lookup(class_id)
        return float(row["typical_serving_g"]) if row else 0.0

    def area_coeff(self, class_id):
        """Gram per cm^2 (top-down) untuk konversi luas -> massa."""
        row = self.lookup(class_id)
        return float(row["area_to_gram_coeff"]) if row else 0.0

    def food_profile(self, class_id):
        """Return profil makanan (rice, meat, soup, …) untuk class_id."""
        row = self.lookup(class_id)
        return str(row["profile"]) if row and "profile" in row else "default"


def build_item_rows(detections, db: NutritionDB, grams_by_index=None):
    """Bangun baris tabel per-item.

    grams_by_index: dict {i: grams} hasil estimasi porsi (Tier 2). Bila None atau
    indeks tak ada, pakai typical_serving_g (fallback porsi standar).
    Returns: (rows: list[dict], totals: dict).
    """
    rows = []
    totals = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    for i, det in enumerate(detections):
        cid = det["class_id"]
        if grams_by_index and i in grams_by_index and grams_by_index[i] is not None:
            grams = grams_by_index[i]
            portion_src = "estimasi area"
        else:
            grams = db.default_grams(cid)
            portion_src = "porsi standar"
        nut = db.nutrition_for(cid, grams)
        if nut is None:
            continue
        gizi_src = f"✔️ {nut['source']}" if nut["verified"] else "draft (estimasi)"
        rows.append({
            "Makanan": nut["name"],
            "Conf": round(det["conf"], 2),
            "Porsi (g)": nut["grams"],
            "Sumber porsi": portion_src,
            "Kalori (kkal)": nut["kcal"],
            "Protein (g)": nut["protein_g"],
            "Karbo (g)": nut["carbs_g"],
            "Lemak (g)": nut["fat_g"],
            "Sumber gizi": gizi_src,
        })
        for k in totals:
            totals[k] += nut[k]
    totals = {k: round(v, 1) for k, v in totals.items()}
    return rows, totals
