"""
Generate a DRAFT nutrition table for the 256 UEC-Food-256 categories.

Cara kerja: tiap nama makanan dicocokkan ke sebuah "profil" gizi berdasarkan
kata kunci (rice / noodle / soup / fried / dessert / dst). Nilai per-100g adalah
ESTIMASI kasar yang masuk akal, bukan angka resmi — gunakan sebagai titik awal
lalu koreksi manual baris yang penting. Didokumentasikan sbg future work di laporan.

Output: data/nutrition.csv dengan kolom:
  class_id, name, kcal_per_100g, protein_g, carbs_g, fat_g,
  typical_serving_g, area_to_gram_coeff, profile, notes

- class_id : 0-based, urut sama dengan model.names / data.yaml.
- area_to_gram_coeff : gram per cm^2 luas top-down (untuk estimasi porsi Tier 2).

Jalankan:  python tools/generate_nutrition_csv.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATEGORY_FILE = ROOT / "yolo_uec_food256" / "FoodCategory.txt"
OUT_CSV = ROOT / "data" / "nutrition.csv"

# Profil gizi: (kcal_per_100g, protein_g, carbs_g, fat_g, typical_serving_g, coeff g/cm^2)
PROFILES = {
    "dessert":  (350,  5, 50, 15, 120, 1.2),
    "soup":     ( 55,  3,  5, 2.5, 300, 1.0),
    "noodle":   (145,  6, 22,  4, 250, 1.8),
    "rice":     (150,  4, 30, 2.5, 200, 2.0),
    "fried":    (280, 14, 18, 18, 150, 1.6),
    "pizza":    (266, 11, 33, 10, 150, 1.5),
    "handheld": (280, 10, 35, 11, 150, 1.4),  # burger/sandwich/bread/taco
    "meat":     (230, 20,  3, 16, 150, 2.0),
    "seafood":  (150, 18,  2,  7, 150, 1.7),
    "egg_tofu": (145, 11,  3, 10, 130, 1.6),
    "veg":      ( 75,  3,  9, 3.5, 150, 1.0),
    "dumpling": (230,  9, 26, 10, 150, 1.5),  # dumpling/spring roll/snack
    "curry":    (130,  6, 14,  6, 250, 2.0),
    "default":  (180,  8, 20,  8, 180, 1.6),
}

# (profile, [keywords]) dicek BERURUTAN -> yang cocok pertama dipakai.
# Urutan penting: yang lebih spesifik/dominan ditaruh lebih dulu.
RULES = [
    ("dessert",  ["cake", "tiramisu", "waffle", "pancake", "pudding", "jelly",
                  "crepe", "crape", "muffin", "scone", "churro", "brownie",
                  "doughnut", "donut", "pie", "parfait", "tart", "cream puff",
                  "moon cake", "mooncake", "oshiruko", "malasada", "haupia",
                  "custard", "glutinous rice balls", "fish-shaped pancake"]),
    ("soup",     ["soup", "miso", "chowder", "broth", "stew", "jjigae", "oden",
                  "pot au feu", "minestrone", "potage", "hot pot", "ragout",
                  "sukiyaki", "champon", "tanmen"]),
    ("noodle",   ["noodle", "ramen", "udon", "soba", "spaghetti", "pasta", "pho",
                  "laksa", "mie ", "vermicelli", "macaroni", "lasagna",
                  "chow mein", "mian", "khao soi", "crispy noodles",
                  "fine white noodles", "dipping noodles"]),
    ("rice",     ["rice", "pilaf", "risotto", "paella", "bibimbap", "sushi",
                  "kamameshi", "gruel", "congee", "jambalaya", "nasi", "bowl",
                  "musubi", "inarizushi", "zoni", "samul"]),
    ("fried",    ["fried", "tempura", "katsu", "cutlet", "croquette", "nugget",
                  "karaage", "kushikatu", "fries", "tonkatsu", "crispy",
                  "deep fried", "crullers"]),
    ("pizza",    ["pizza"]),
    ("handheld", ["hamburger", "burger", "hot dog", "sandwich", "taco", "nachos",
                  "bagel", "toast", "bread", "croissant", "tortilla"]),
    ("meat",     ["steak", "beef", "pork", "chicken", "duck", "lamb", "satay",
                  "yakitori", "teriyaki", "kebab", "adobo", "gulai", "meat",
                  "sausage", "belly", "spareribs", "kung pao", "galbi",
                  "loco moco", "ham ", "horsemeat", "babi"]),
    ("seafood",  ["fish", "salmon", "sashimi", "eel", "shrimp", "prawn", "oyster",
                  "mussel", "saury", "namero", "snails", "winter melon"]),
    ("egg_tofu", ["egg", "omelet", "tofu", "natto", "ganmodoki", "yudofu",
                  "scrambled"]),
    ("veg",      ["salad", "vegetable", "spinach", "eggplant", "sauteed",
                  "papaya", "mozuku", "burdock", "cabbage"]),
    ("dumpling", ["jiaozi", "gyoza", "dumpling", "xiao long bao", "wonton",
                  "lumpia", "spring roll", "curry puff", "takoyaki", "bun",
                  "baozi", "siu mai", "popcorn", "spam"]),
    ("curry",    ["curry"]),
]


def pick_profile(name: str) -> str:
    low = name.lower()
    for profile, keywords in RULES:
        for kw in keywords:
            if kw in low:
                return profile
    return "default"


def read_categories():
    """Return list of (class_id_0based, name) following file order."""
    names = []
    with open(CATEGORY_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i == 0 and line.lower().startswith("id"):
                continue  # header
            if not line.strip():
                continue
            # format: "<id>\t<name>"
            parts = line.split("\t", 1)
            name = parts[1].strip() if len(parts) == 2 else parts[0].strip()
            names.append(name)
    return list(enumerate(names))  # class_id 0-based


def main():
    cats = read_categories()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "kcal_per_100g", "protein_g", "carbs_g",
                    "fat_g", "typical_serving_g", "area_to_gram_coeff",
                    "profile", "notes"])
        for class_id, name in cats:
            prof = pick_profile(name)
            kcal, p, c, fat, serv, coeff = PROFILES[prof]
            w.writerow([class_id, name, kcal, p, c, fat, serv, coeff, prof,
                        "draft auto-estimate"])
    print(f"Wrote {len(cats)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
