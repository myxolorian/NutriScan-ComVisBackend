"""MATERI: Feature concept & invariance, Harris & Shi-Tomasi, SIFT/SURF/ORB.

Visualizer untuk tab "CV Explainer": tampilkan titik fitur yang dideteksi, dan
demo invariance (fitur ORB tetap cocok meski gambar dirotasi/diskala).
"""
import cv2
import numpy as np


def harris_corners(img_rgb, block=2, ksize=3, k=0.04, thresh=0.01):
    """Deteksi sudut Harris. Return RGB dgn sudut ditandai merah."""
    out = img_rgb.copy()
    gray = np.float32(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    dst = cv2.cornerHarris(gray, block, ksize, k)
    dst = cv2.dilate(dst, None)
    out[dst > thresh * dst.max()] = (255, 0, 0)
    return out, int((dst > thresh * dst.max()).sum())


def shi_tomasi(img_rgb, max_corners=200, quality=0.01, min_dist=10):
    """Shi-Tomasi (goodFeaturesToTrack). Return RGB dgn lingkaran hijau."""
    out = img_rgb.copy()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_dist)
    n = 0
    if corners is not None:
        for c in corners.astype(int):
            x, y = c.ravel()
            cv2.circle(out, (x, y), 4, (0, 255, 0), -1)
        n = len(corners)
    return out, n


def orb_keypoints(img_rgb, n_features=500):
    """Deteksi keypoint ORB + gambar (ukuran & orientasi). Return RGB."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    kps = orb.detect(gray, None)
    out = cv2.drawKeypoints(img_rgb, kps, None, color=(255, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out, len(kps)


def orb_match_invariance(img_rgb, angle=30, scale=0.7, n_features=800):
    """Demo invariance: cocokkan gambar dgn versi rotasi+skala-nya sendiri.

    Menunjukkan deskriptor ORB robust terhadap rotasi & skala (Materi).
    Return (match_image RGB, jumlah good matches).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    warped = cv2.warpAffine(gray, M, (w, h))

    orb = cv2.ORB_create(nfeatures=n_features)
    k1, d1 = orb.detectAndCompute(gray, None)
    k2, d2 = orb.detectAndCompute(warped, None)
    if d1 is None or d2 is None:
        return img_rgb, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:40]
    vis = cv2.drawMatches(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), k1,
        cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB), k2,
        matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return vis, len(matches)
