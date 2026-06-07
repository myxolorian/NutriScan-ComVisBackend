"""MATERI: Convolution & correlation, Noise reduction, Smoothing/Sharpening, Edge detection.

Semua fungsi menerima & mengembalikan numpy RGB uint8 (kecuali yg jelas grayscale).
Dipakai sebagai tahap preprocessing sebelum YOLO + tab "CV Explainer".
"""
import cv2
import numpy as np

# Kernel konvolusi 3x3 untuk demonstrasi filter2D (Materi: convolution & correlation).
KERNELS = {
    "identity":   np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32),
    "box blur":   np.ones((3, 3), np.float32) / 9.0,
    "sharpen":    np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
    "laplacian":  np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32),
    "emboss":     np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32),
    "sobel x":    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
    "sobel y":    np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
}


def apply_kernel(img_rgb, kernel):
    """Konvolusi 2D manual via cv2.filter2D (Materi: convolution & correlation)."""
    return cv2.filter2D(img_rgb, ddepth=-1, kernel=kernel)


def gaussian_blur(img_rgb, ksize=5):
    """Noise reduction + smoothing (low-pass Gaussian)."""
    k = int(ksize) | 1  # paksa ganjil
    return cv2.GaussianBlur(img_rgb, (k, k), 0)


def median_blur(img_rgb, ksize=5):
    """Noise reduction (efektif untuk salt-and-pepper noise)."""
    k = int(ksize) | 1
    return cv2.medianBlur(img_rgb, k)


def unsharp_sharpen(img_rgb, amount=1.0, ksize=5):
    """Sharpening via unsharp masking: img + amount*(img - blur)."""
    blur = gaussian_blur(img_rgb, ksize)
    sharp = cv2.addWeighted(img_rgb, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def sobel_edges(img_rgb):
    """Magnitudo gradien Sobel (Materi: edge detection). Return RGB grayscale."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)


def canny_edges(img_rgb, t1=100, t2=200):
    """Deteksi tepi Canny (Materi: edge detection). Return RGB grayscale."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(t1), int(t2))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def add_gaussian_noise(img_rgb, sigma=20):
    """Tambah noise (untuk demo bahwa denoise bekerja)."""
    noise = np.random.normal(0, sigma, img_rgb.shape).astype(np.float32)
    out = np.clip(img_rgb.astype(np.float32) + noise, 0, 255)
    return out.astype(np.uint8)
