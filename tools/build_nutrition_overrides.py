"""Bangun data/nutrition_overrides.csv — nilai gizi TERVERIFIKASI utk makanan demo.

Nilai per-100g dirangkum dari sumber resmi/terpercaya (USDA FoodData Central,
MEXT Japan Food Composition, TKPI Kemenkes, FatSecret) — dicatat di kolom `source`.
Hanya berisi baris yang diverifikasi; sisanya tetap pakai draft heuristik.

Skrip ini memvalidasi konsistensi makro (4*P + 4*C + 9*F ≈ kcal, toleransi ±15%)
lalu menulis CSV. Jalankan: python tools/build_nutrition_overrides.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_CSV = ROOT / "data" / "nutrition.csv"
OUT_CSV = ROOT / "data" / "nutrition_overrides.csv"

# name -> (kcal/100g, protein, carbs, fat, serving_g, density g/cm3, source)
# Nilai = per 100 g makanan sebagaimana disajikan (cooked/as-served).
VERIFIED = {
    # ---- Dari test image (terbukti terdeteksi) ----
    "rice": (130, 2.7, 28.0, 0.3, 150, 0.85, "USDA FDC #168878 (Rice, white, cooked)"),
    "stir-fried beef and peppers": (149, 12.0, 6.0, 8.5, 200, 0.90, "USDA FNDDS (Beef with peppers, stir-fried)"),
    "Caesar salad": (110, 4.0, 6.0, 8.0, 150, 0.40, "USDA FNDDS (Caesar salad)"),
    "Deep Fried Chicken Wing": (290, 27.0, 9.0, 18.0, 100, 0.90, "USDA FDC (Chicken wing, fried, batter)"),
    "french bread": (274, 9.0, 52.0, 3.0, 80, 0.30, "USDA FDC #172731 (Bread, French/Vienna)"),
    "thinly sliced raw horsemeat": (110, 20.1, 0.3, 2.5, 80, 1.05, "MEXT 11109 (Horse, raw / basashi)"),
    "inarizushi": (200, 5.0, 38.0, 3.5, 120, 0.90, "MEXT (Inarizushi)"),
    "mozuku": (14, 0.5, 3.0, 0.1, 100, 1.00, "MEXT (Mozuku seaweed)"),
    "chow mein": (150, 6.0, 20.0, 5.0, 250, 0.70, "USDA FNDDS (Chow mein, with meat)"),
    "parfait": (230, 4.0, 30.0, 11.0, 150, 0.60, "USDA FNDDS (Parfait, dessert)"),
    "pho": (72, 5.0, 9.0, 1.5, 350, 1.00, "USDA FNDDS (Pho, Vietnamese noodle soup)"),
    "pork belly": (320, 15.0, 1.0, 28.0, 120, 0.95, "USDA FDC (Pork belly, cooked)"),
    "okinawa soba": (130, 6.0, 20.0, 3.0, 350, 0.80, "MEXT (Okinawa soba)"),
    "braised pork meat ball with napa cabbage": (180, 11.0, 6.0, 12.0, 200, 0.90, "USDA FNDDS (Pork meatball, braised)"),
    "shrimp with chill source": (130, 14.0, 6.0, 5.0, 150, 0.90, "USDA FNDDS (Shrimp in chili sauce)"),
    "hot pot": (100, 8.0, 5.0, 5.0, 350, 0.95, "USDA FNDDS (Hot pot, mixed)"),
    "steamed meat dumpling": (220, 9.0, 26.0, 9.0, 120, 0.80, "USDA FNDDS (Dumpling, steamed, meat)"),
    "curry puff": (290, 6.0, 30.0, 16.0, 80, 0.60, "Sing/MY Food Comp (Curry puff)"),
    "pork loin cutlet": (290, 18.0, 15.0, 18.0, 150, 0.85, "MEXT (Tonkatsu, pork loin cutlet)"),
    # ---- Subset Indonesia (TKPI / FatSecret ID) ----
    "nasi goreng": (247, 9.4, 31.5, 9.0, 250, 0.85, "FatSecret ID (Nasi goreng ayam)"),
    "ayam goreng": (250, 19.0, 8.0, 15.0, 120, 0.90, "TKPI Kemenkes (Ayam goreng)"),
    "ayam bakar": (185, 22.0, 3.0, 9.0, 120, 0.95, "TKPI Kemenkes (Ayam bakar)"),
    "bubur ayam": (155, 11.0, 15.0, 5.0, 250, 1.00, "FatSecret ID (Bubur ayam)"),
    "gulai": (180, 9.0, 6.0, 13.0, 150, 0.95, "TKPI Kemenkes (Gulai)"),
    "mie goreng": (180, 6.0, 24.0, 7.0, 250, 0.70, "FatSecret ID (Mie goreng)"),
    "mie ayam": (150, 7.0, 20.0, 5.0, 250, 0.80, "FatSecret ID (Mie ayam)"),
    "nasi campur": (190, 8.0, 27.0, 6.0, 300, 0.90, "FatSecret ID (Nasi campur)"),
    "nasi padang": (210, 9.0, 24.0, 9.0, 300, 0.90, "TKPI/Kompas (Nasi Padang, rata-rata)"),
    "nasi uduk": (163, 2.5, 20.5, 7.8, 200, 0.90, "FatSecret ID (Nasi uduk)"),
    "laksa": (110, 5.0, 12.0, 5.0, 350, 0.95, "Sing/MY Food Comp (Laksa)"),
    "babi guling": (290, 20.0, 2.0, 23.0, 150, 0.95, "FatSecret ID (Babi guling)"),
    # ---- Ikonik global (USDA) ----
    "sushi": (145, 6.0, 28.0, 2.0, 150, 1.00, "USDA FNDDS (Sushi roll)"),
    "ramen noodle": (110, 5.0, 14.0, 4.0, 400, 0.85, "USDA FNDDS (Ramen, cooked, with broth)"),
    "pizza": (266, 11.0, 33.0, 10.0, 150, 0.70, "USDA FDC #170096 (Pizza, cheese)"),
    "hamburger": (250, 15.0, 29.0, 9.0, 110, 0.80, "USDA FNDDS (Hamburger, plain, on bun)"),
    "spaghetti": (158, 5.8, 30.9, 0.9, 250, 0.90, "USDA FDC #168928 (Pasta, cooked)"),
    "fried rice": (170, 5.0, 24.0, 6.0, 250, 0.85, "USDA FNDDS (Fried rice)"),
    "french fries": (280, 3.4, 36.0, 14.0, 120, 0.50, "USDA FDC #170437 (French fries)"),
    "hot dog": (290, 11.0, 22.0, 17.0, 100, 0.70, "USDA FNDDS (Hot dog, with bun)"),
    "sandwiches": (250, 11.0, 28.0, 10.0, 150, 0.50, "USDA FNDDS (Sandwich, generic)"),
}


def main():
    # name -> class_id dari base CSV (sumber kebenaran urutan kelas)
    name2id = {}
    with open(BASE_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name2id[r["name"]] = int(r["class_id"])

    rows, warnings = [], []
    for name, (kcal, p, c, fat, serv, dens, src) in VERIFIED.items():
        if name not in name2id:
            warnings.append(f"NAME NOT FOUND: {name!r}")
            continue
        # validasi konsistensi makro
        kcal_calc = 4 * p + 4 * c + 9 * fat
        if kcal_calc <= 0 or abs(kcal_calc - kcal) / kcal > 0.15:
            warnings.append(f"MACRO OFF {name!r}: tertulis {kcal} vs hitung {kcal_calc:.0f}")
        rows.append((name2id[name], name, kcal, p, c, fat, serv, dens, src))

    rows.sort(key=lambda r: r[0])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "kcal_per_100g", "protein_g", "carbs_g",
                    "fat_g", "typical_serving_g", "density_g_per_cm3", "source"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} verified rows -> {OUT_CSV}")
    if warnings:
        print(f"\n{len(warnings)} WARNING:")
        for w_ in warnings:
            print("  -", w_)
    else:
        print("Semua baris lolos validasi makro (±15%).")


if __name__ == "__main__":
    main()
