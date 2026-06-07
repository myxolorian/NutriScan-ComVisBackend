"""MATERI: SIFT/SURF/ORB descriptors + Affine/Projective transforms + Homography(RANSAC).

Estimasi skala dunia-nyata (piksel -> cm) memakai OBJEK REFERENSI berukuran diketahui.
Alur:
  1. ORB detect+compute pada template kartu referensi dan pada foto (scene).
  2. BFMatcher + Lowe ratio test -> good matches.
  3. cv2.findHomography(..., cv2.RANSAC) -> H (template_px -> scene_px).
  4. Proyeksikan 4 sudut kartu -> hitung panjang sisi (px) / ukuran nyata (cm) = px_per_cm.
  5. Luas makanan cm^2 = area_px / px_per_cm^2 ; gram = luas_cm2 * area_to_gram_coeff.

Kartu referensi default: seukuran kartu kredit/KTP = 8.56 x 5.40 cm.
"""
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / "data" / "reference_card.png"

# Ukuran fisik kartu referensi (cm) — ISO/IEC 7810 ID-1 (kartu kredit/KTP).
CARD_W_CM = 8.56
CARD_H_CM = 5.40


def make_reference_card(out_path=DEFAULT_TEMPLATE, w_px=856, h_px=540, seed=42):
    """Buat template kartu bertekstur tinggi (banyak sudut) agar ORB mudah cocok.

    Cetak file ini pada ukuran fisik 8.56 x 5.40 cm lalu letakkan di samping makanan.
    """
    rng = np.random.default_rng(seed)
    img = np.full((h_px, w_px, 3), 255, np.uint8)
    # Pola kotak-kotak + bentuk acak -> banyak tepi/sudut untuk ORB.
    step = 40
    for y in range(0, h_px, step):
        for x in range(0, w_px, step):
            if ((x // step) + (y // step)) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + step, y + step), (30, 30, 30), -1)
    for _ in range(60):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p1 = (int(rng.integers(0, w_px)), int(rng.integers(0, h_px)))
        if rng.random() < 0.5:
            r = int(rng.integers(8, 30))
            cv2.circle(img, p1, r, c, -1)
        else:
            p2 = (int(rng.integers(0, w_px)), int(rng.integers(0, h_px)))
            cv2.rectangle(img, p1, p2, c, 2)
    cv2.rectangle(img, (4, 4), (w_px - 5, h_px - 5), (0, 0, 0), 6)  # border tegas
    cv2.putText(img, "REF 8.56 x 5.40 cm", (30, h_px - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 3, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_path


def estimate_scale_orb(scene_rgb, template_rgb, card_w_cm=CARD_W_CM,
                       card_h_cm=CARD_H_CM, min_matches=15, n_features=1500):
    """Estimasi px_per_cm via ORB matching + homografi RANSAC.

    Returns dict:
        ok          : bool
        px_per_cm   : float (0 jika gagal)
        n_inliers   : int
        H           : homografi 3x3 (template_px -> scene_px) atau None
        quad        : 4 sudut kartu di scene (px) atau None
        reason      : str penjelasan bila gagal
    """
    fail = {"ok": False, "px_per_cm": 0.0, "n_inliers": 0, "H": None,
            "quad": None, "reason": ""}
    g_t = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
    g_s = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    k1, d1 = orb.detectAndCompute(g_t, None)
    k2, d2 = orb.detectAndCompute(g_s, None)
    if d1 is None or d2 is None:
        return {**fail, "reason": "ORB tidak menemukan fitur"}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in (pair for pair in knn if len(pair) == 2)
            if m.distance < 0.75 * n.distance]  # Lowe ratio test
    if len(good) < min_matches:
        return {**fail, "reason": f"good matches {len(good)} < {min_matches}"}

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return {**fail, "reason": "findHomography gagal"}
    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < min_matches:
        return {**fail, "reason": f"inliers {n_inliers} < {min_matches}"}

    h_t, w_t = g_t.shape
    corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # --- Validasi anti false-positive (tanpa kartu, ORB bisa "asal cocok") ---
    # 1) kuadran harus konveks (kartu planar -> proyeksi konveks).
    if not cv2.isContourConvex(quad.astype(np.float32)):
        return {**fail, "reason": "kuadran tidak konveks (kemungkinan salah cocok)"}
    # 2) luas kuadran wajar terhadap gambar.
    scene_area = scene_rgb.shape[0] * scene_rgb.shape[1]
    quad_area = abs(cv2.contourArea(quad.astype(np.float32)))
    if not (0.003 * scene_area < quad_area < 0.85 * scene_area):
        return {**fail, "reason": "luas kartu di luar batas wajar"}

    # Panjang sisi atas/bawah -> lebar (cm); sisi kiri/kanan -> tinggi (cm).
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    ppc_w = (top + bottom) / 2 / card_w_cm
    ppc_h = (left + right) / 2 / card_h_cm
    # 3) skala dari lebar & tinggi harus konsisten (kartu nyata -> mirip).
    if min(ppc_w, ppc_h) <= 0 or max(ppc_w, ppc_h) / min(ppc_w, ppc_h) > 2.2:
        return {**fail, "reason": "skala lebar vs tinggi tidak konsisten"}

    px_per_cm = float((ppc_w + ppc_h) / 2)
    return {"ok": True, "px_per_cm": px_per_cm, "n_inliers": n_inliers,
            "H": H, "quad": quad, "reason": "ok"}


def draw_reference_quad(scene_rgb, quad):
    """Gambar kotak kartu referensi yang terdeteksi pada scene."""
    out = scene_rgb.copy()
    if quad is not None:
        cv2.polylines(out, [quad.astype(int)], True, (255, 0, 255), 3, cv2.LINE_AA)
    return out


def rectify_topdown(scene_rgb, H, template_shape, card_w_cm=CARD_W_CM,
                    card_h_cm=CARD_H_CM, margin=2.5):
    """Warp scene ke tampilan bird's-eye metrik (Materi: projective transform).

    H memetakan template_px -> scene. Kita balik (scene -> template-frame) lalu
    skala ke kanvas cm. `margin` memperluas kanvas agar makanan di sekitar kartu
    ikut terlihat. template_shape = (h_t, w_t) dari template yang dipakai.
    """
    if H is None:
        return scene_rgb
    h_t, w_t = template_shape[:2]
    px_per_cm = 40
    out_w = int(card_w_cm * px_per_cm * margin)
    out_h = int(card_h_cm * px_per_cm * margin)
    # template_px -> metric (kartu di tengah kanvas).
    S = np.array([[card_w_cm * px_per_cm / w_t, 0, out_w / 2 - card_w_cm * px_per_cm / 2],
                  [0, card_h_cm * px_per_cm / h_t, out_h / 2 - card_h_cm * px_per_cm / 2],
                  [0, 0, 1]], np.float32)
    warp = S @ np.linalg.inv(H)
    return cv2.warpPerspective(scene_rgb, warp, (out_w, out_h))


def grams_from_area(area_px, px_per_cm, coeff, clamp=(10.0, 1500.0)):
    """Konversi luas piksel -> cm^2 -> gram (massa)."""
    if px_per_cm <= 0:
        return None
    area_cm2 = area_px / (px_per_cm ** 2)
    grams = area_cm2 * coeff
    return float(np.clip(grams, *clamp))
