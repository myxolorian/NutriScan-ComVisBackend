"""MATERI: Non-Maximum Suppression (NMS) — implementasi from-scratch.

YOLO sudah melakukan NMS internal (parameter `iou`). Modul ini meng-implementasi
NMS sendiri untuk demonstrasi/penjelasan: ambil banyak kotak tumpang-tindih
(dengan iou tinggi saat predict) lalu tunjukkan efek NMS before/after.
"""
import numpy as np


def iou(box_a, box_b):
    """Intersection-over-Union dua kotak [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thresh=0.45):
    """Greedy NMS from-scratch. Return indeks kotak yang dipertahankan.

    boxes: list/array [N,4] format [x1,y1,x2,y2]; scores: list/array [N].
    """
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = scores.argsort()[::-1].tolist()  # skor tertinggi dulu
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep


def make_synthetic_overlaps(detections, per_box=5, jitter=0.12, seed=0):
    """Buat kotak tumpang-tindih sintetis di sekitar tiap deteksi.

    Berguna untuk demo NMS saat gambar bersih (YOLO sudah membuang duplikat),
    sehingga efek NMS tetap terlihat jelas. Ditandai sebagai contoh demonstrasi.
    """
    rng = np.random.default_rng(seed)
    out = []
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        w, h = x2 - x1, y2 - y1
        out.append(dict(d))  # kotak asli (skor tertinggi)
        for _ in range(per_box):
            dx = rng.uniform(-jitter, jitter) * w
            dy = rng.uniform(-jitter, jitter) * h
            out.append({
                "class_id": d["class_id"], "name": d["name"],
                "conf": max(0.05, d["conf"] - rng.uniform(0.05, 0.3)),
                "xyxy": [int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)],
            })
    return out


def nms_on_detections(detections, iou_thresh=0.45):
    """Terapkan NMS (class-agnostic) pada list deteksi dict {xyxy, conf}.

    Return (kept_detections, removed_detections).
    """
    if not detections:
        return [], []
    boxes = [d["xyxy"] for d in detections]
    scores = [d["conf"] for d in detections]
    keep_idx = set(nms(boxes, scores, iou_thresh))
    kept = [d for i, d in enumerate(detections) if i in keep_idx]
    removed = [d for i, d in enumerate(detections) if i not in keep_idx]
    return kept, removed
