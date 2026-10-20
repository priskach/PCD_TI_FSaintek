import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

try:
    # --- SOAL 1: OpenCV & Matplotlib ---
    # 1. Baca gambar 'orange.jpg'
    img_bgr = cv2.imread('orange.jpg')
    # 2. Konversi BGR ke RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Tampilkan Soal 1
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.title("Soal 1: Gambar Asli (RGB)")
    plt.axis('off')
    plt.show()

    # --- SOAL 2: NumPy (Cropping) ---
    # Lakukan slicing array pada img_rgb [y_start:y_end, x_start:x_end]
    # Gunakan y: 200 s.d 600, dan x: 400 s.d 800
    img_crop = img_rgb[200:600, 400:800]
    
    # Tampilkan Soal 2
    plt.figure(figsize=(5, 5))
    plt.imshow(img_crop)
    plt.title("Soal 2: Hasil Crop")
    plt.axis('off')
    plt.show()

    # --- SOAL 3: Scikit-Image (Filtering) ---
    # 1. Konversi img_crop ke grayscale terlebih dahulu
    img_crop_gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
    # 2. Terapkan filter Sobel dari skimage
    edges = filters.sobel(img_crop_gray)
    
    # Tampilkan Soal 3
    plt.figure(figsize=(5, 5))
    plt.imshow(edges, cmap='gray')
    plt.title("Soal 3: Deteksi Tepi Sobel")
    plt.axis('off')
    plt.show()

except Exception as e:
    print("Terjadi kesalahan. Pastikan file 'orange.jpg' telah diunggah!", e)