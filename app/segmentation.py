"""MATERI: Edge detection + thresholding + morfologi -> mask area makanan.

Tujuan: dari bounding box YOLO, segmentasi piksel makanan sesungguhnya (bukan
seluruh kotak) untuk mendapat LUAS PIKSEL -> dipakai estimasi porsi (Tier 2).

Heuristik: makanan biasanya lebih ber-saturasi daripada piring putih. Kita pakai
kanal saturasi (HSV) + Otsu threshold + operasi morfologi (open/close) + ambil
komponen terbesar. Ini aproksimasi, didokumentasikan sebagai keterbatasan.
"""
import cv2
import numpy as np


def segment_food_in_box(img_rgb, box):
    """Segmentasi makanan di dalam satu bounding box.

    Args:
        img_rgb: gambar penuh (RGB uint8)
        box: [x1, y1, x2, y2]
    Returns dict:
        area_px      : luas piksel makanan
        box_px       : luas kotak
        fill_ratio   : area_px / box_px
        mask_full    : mask bool seukuran gambar penuh (True = makanan)
    """
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = img_rgb[y1:y2, x1:x2]
    mask_full = np.zeros((h, w), dtype=bool)
    if crop.size == 0:
        return {"area_px": 0, "box_px": 0, "fill_ratio": 0.0, "mask_full": mask_full}

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    sat = cv2.GaussianBlur(sat, (5, 5), 0)  # noise reduction sebelum threshold
    # Otsu: piksel saturasi tinggi -> makanan.
    _, th = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfologi: tutup lubang lalu buang bintik kecil.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # Ambil komponen terhubung terbesar (makanan utama).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    crop_mask = np.zeros(th.shape, dtype=bool)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        crop_mask = labels == largest
    else:
        crop_mask = th > 0

    area_px = int(crop_mask.sum())
    box_px = crop.shape[0] * crop.shape[1]
    mask_full[y1:y2, x1:x2] = crop_mask
    fill = area_px / box_px if box_px else 0.0
    # Jika segmentasi gagal (terlalu kecil/besar), fallback ke 70% kotak.
    if fill < 0.05 or fill > 0.98:
        area_px = int(0.70 * box_px)
        fill = 0.70
    return {"area_px": area_px, "box_px": box_px, "fill_ratio": round(fill, 3),
            "mask_full": mask_full}


def overlay_masks(img_rgb, masks, color=(255, 0, 0), alpha=0.45):
    """Tumpuk beberapa mask (list of bool array) di atas gambar untuk visualisasi."""
    out = img_rgb.copy().astype(np.float32)
    combined = np.zeros(img_rgb.shape[:2], dtype=bool)
    for m in masks:
        combined |= m
    color_arr = np.array(color, np.float32)
    out[combined] = (1 - alpha) * out[combined] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)
