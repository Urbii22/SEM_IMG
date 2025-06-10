#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flujo mínimo para relacionar espectro HSI (banda 728 nm) con count de Cu-Kα1
Autor: <tu nombre>
Revisión: implementación de selección de ROI con relación de aspecto fija usando callback de OpenCV
"""

import cv2
import numpy as np
import pandas as pd
import spectral as spy
from skimage import measure, morphology, color
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import filedialog, Tk
import sys
from PIL import Image

# ----------------------------------------------------------------------
# 0. Selección de ficheros ------------------------------------------------
# ----------------------------------------------------------------------
root = Tk()
root.withdraw()                         # Ocultamos la ventana principal

hsi_path = filedialog.askopenfilename(
    title="Selecciona el archivo .hdr (o .bil) de la imagen hiperespectral",
    filetypes=[("Archivos ENVI", "*.hdr *.bil"), ("Todos", "*.*")]
)
if not hsi_path:
    sys.exit("No se eligió imagen hiperespectral. Abortando.")
hsi_path = Path(hsi_path)

# Si se ha elegido el .bil, buscamos el .hdr del mismo nombre
if hsi_path.suffix.lower() == ".bil":
    HSI_HDR = hsi_path.with_suffix(".hdr")
else:
    HSI_HDR = hsi_path

# Imágenes SEM y mapa de Cu
sem_path = filedialog.askopenfilename(title="Imagen SEM (gris)")
if not sem_path:
    sys.exit("No se eligió imagen SEM. Abortando.")
cu_path = filedialog.askopenfilename(title="Imagen mapa de Cu Kα1 (rojo)")
if not cu_path:
    sys.exit("No se eligió imagen Cu. Abortando.")

SEM_IMG = Path(sem_path)
CU_IMG  = Path(cu_path)

# ----------------------------------------------------------------------
# 1. Cargar cubo HSI y elegir ROI con relación de aspecto fija --------
# ----------------------------------------------------------------------
print("-> Cargando cubo hiperespectral…")
img_hsi = spy.open_image(str(HSI_HDR))          # SpyFile
cube = img_hsi.load()                           # np.ndarray (r, c, b)

TARGET_WL = 728  # nm – cámbialo si tu banda umbral es otra

a_wl = img_hsi.metadata.get('wavelength', [])
wl = np.asarray(a_wl, dtype=float) if a_wl else np.array([])
if wl.size:
    band_idx = int(np.argmin(np.abs(wl - TARGET_WL)))
    print(f"Banda objetivo: índice {band_idx} (≈ {wl[band_idx]:.1f} nm)")
else:
    band_idx = int(TARGET_WL)
    print(f"No se encontraron longitudes de onda; usando banda {band_idx}")

band_img = cube[:, :, band_idx]
band_norm = cv2.normalize(band_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

orig_h, orig_w = band_norm.shape
aspect_ratio = orig_w / orig_h  # ancho/alto deseada para ROI

# Variables para selección interactiva
drawing = False
ix, iy = -1, -1
roi = None

# Callback del ratón para dibujar rectángulo con relación de aspecto fija
def draw_aspect_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_disp = band_norm.copy()
        # Calculamos ancho actual
        w_sel = x - ix
        # Determinamos signo y abuso de signo si se arrastra hacia arriba/izquierda
        sign_x = 1 if w_sel >= 0 else -1
        w_abs = abs(w_sel)
        # Ajustamos alto según relación de aspecto: h = w / aspect_ratio
        h_abs = int(w_abs / aspect_ratio)
        sign_y = 1 if y - iy >= 0 else -1
        # Definimos esquina inferior derecha teniendo en cuenta signo
        ex = ix + sign_x * w_abs
        ey = iy + sign_y * h_abs
        # Dibujamos rectángulo provisional
        cv2.rectangle(img_disp, (ix, iy), (ex, ey), (255), 2)
        cv2.imshow("Banda 728 nm - ROI fija", img_disp)
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        w_sel = x - ix
        sign_x = 1 if w_sel >= 0 else -1
        w_abs = abs(w_sel)
        h_abs = int(w_abs / aspect_ratio)
        sign_y = 1 if y - iy >= 0 else -1
        ex = ix + sign_x * w_abs
        ey = iy + sign_y * h_abs
        # Normalizamos coordenadas a valores positivos mínimos
        x0, y0 = min(ix, ex), min(iy, ey)
        w0, h0 = w_abs, h_abs
        # Aseguramos que la ROI no salga de la imagen
        x0 = max(0, min(x0, orig_w - w0))
        y0 = max(0, min(y0, orig_h - h0))
        roi = (int(x0), int(y0), int(w0), int(h0))
        # Dibujar rect final
        img_final = band_norm.copy()
        cv2.rectangle(img_final, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (255), 2)
        cv2.imshow("Banda 728 nm - ROI fija", img_final)

cv2.namedWindow("Banda 728 nm - ROI fija", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Banda 728 nm - ROI fija", draw_aspect_rectangle)
cv2.imshow("Banda 728 nm - ROI fija", band_norm)
print("Pincha y arrastra para seleccionar ROI con relación de aspecto. Pulsa ESC para confirmar.")
while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC finaliza
        break
cv2.destroyAllWindows()

if roi is None:
    sys.exit("No se seleccionó ninguna ROI válida. Abortando.")

x, y, w, h = roi
print(f"ROI seleccionada (fija aspecto): x={x}, y={y}, w={w}, h={h}")

hsi_crop   = cube[y:y+h, x:x+w, :]     # cubo recortado
band_crop  = band_norm[y:y+h, x:x+w]   # para registro y mostrar

# ----------------------------------------------------------------------
# 2. Cargar imágenes SEM y mapa de Cu usando PIL -------------------------
# ----------------------------------------------------------------------
print("-> Leyendo imágenes de microscopio con PIL…")
try:
    pil_sem = Image.open(SEM_IMG).convert("L")
    img_sem = np.array(pil_sem)
    pil_cu = Image.open(CU_IMG).convert("RGB")
    img_cu = cv2.cvtColor(np.array(pil_cu), cv2.COLOR_RGB2BGR)
except Exception as e:
    sys.exit(f"Error leyendo imágenes: {e}")

img_sem_u8 = cv2.normalize(img_sem, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
band_crop_u8 = band_crop.copy()

# ----------------------------------------------------------------------
# 3. Registro automático SEM → HSIcrop con ORB (o AKAZE si ORB falla) ------
# ----------------------------------------------------------------------
print("-> Registrando automáticamente con ORB…")
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(img_sem_u8, None)
kp2, des2 = orb.detectAndCompute(band_crop_u8, None)

if des1 is None or des2 is None:
    print("  ORB no encontró suficientes descriptores. Intentando con AKAZE…")
    try:
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(img_sem_u8, None)
        kp2, des2 = akaze.detectAndCompute(band_crop_u8, None)
        if des1 is None or des2 is None:
            sys.exit("Error: AKAZE tampoco encontró descriptores suficientes. Revisa las imágenes.")
        print("  AKAZE descriptors obtenidos.")
    except Exception:
        sys.exit("Error: no se pudo usar AKAZE para extracción de descriptores.")
else:
    print("  ORB descriptors obtenidos.")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
if not matches:
    sys.exit("Error: no se encontraron matches entre descriptores.")
good = sorted(matches, key=lambda x: x.distance)[:50]

pts_sem = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
pts_crop = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

H, mask = cv2.findHomography(pts_sem, pts_crop, cv2.RANSAC, 5.0)
if H is None:
    sys.exit("Error: no se pudo estimar homografía automática.")

# ----------------------------------------------------------------------
# 4. Warp de la máscara roja y silueta SEM --------------------------------
# ----------------------------------------------------------------------
hsv = cv2.cvtColor(img_cu, cv2.COLOR_BGR2HSV)
mask_red = cv2.inRange(hsv, (0,  50, 50), (10, 255, 255)) | \
           cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
mask_red = morphology.remove_small_objects(mask_red.astype(bool), 8)

mask_warp = cv2.warpPerspective(mask_red.astype(np.uint8)*255, H, (w, h)) > 0

sem_bin  = img_sem_u8 > img_sem_u8.mean()
sem_warp = cv2.warpPerspective(sem_bin.astype(np.uint8)*255, H, (w, h)) > 0

# ----------------------------------------------------------------------
# 5. Conteo de puntitos y fracción de área ----------------------------
# ----------------------------------------------------------------------
print("-> Etiquetando puntitos rojos…")
labels = measure.label(mask_red)
props  = measure.regionprops(labels)

centroids = np.array([p.centroid[::-1] for p in props])  # (x,y)
centroids_h = np.c_[centroids, np.ones(len(centroids))]
proj = (H @ centroids_h.T).T
proj /= proj[:, 2:3]
cx, cy = proj[:, 0], proj[:, 1]

valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
cx, cy = cx[valid].astype(int), cy[valid].astype(int)

counts = np.zeros((h, w), int)
for xi, yi in zip(cx, cy):
    counts[yi, xi] += 1

area_fraction = sem_warp.astype(float)

# ----------------------------------------------------------------------
# 6. Exportar CSV -------------------------------------------------------
# ----------------------------------------------------------------------
rows, cols = np.nonzero(area_fraction > 0)
records = [dict(
    row=r+y, col=c+x,
    refl_728=float(band_img[r+y, c+x]),
    area_frac=float(area_fraction[r, c]),
    n_puntos=int(counts[r, c])
) for r, c in zip(rows, cols)]

df = pd.DataFrame.from_records(records)
out_csv = Path("analisis_gota.csv")
df.to_csv(out_csv, index=False)
print(f"CSV guardado en {out_csv.resolve()} ({len(df)} filas)")

# ----------------------------------------------------------------------
# 7. Overlay de control con ZOOM ---------------------------------------
# ----------------------------------------------------------------------
zoom_factor = max(1, int(500 / max(w, h)))
band_zoom = cv2.resize(band_crop, (w*zoom_factor, h*zoom_factor), interpolation=cv2.INTER_NEAREST)
mask_zoom = cv2.resize((mask_warp.astype(np.uint8)*255).reshape(h, w),
                       (w*zoom_factor, h*zoom_factor), interpolation=cv2.INTER_NEAREST) > 0

overlay_rgb = np.stack([band_zoom]*3, axis=-1)
overlay_rgb = cv2.cvtColor(overlay_rgb, cv2.COLOR_GRAY2BGR)
overlay_rgb[mask_zoom, 0] = 255
overlay_rgb[mask_zoom, 1] = 0
overlay_rgb[mask_zoom, 2] = 0
alpha = 0.4
mask3 = np.stack([mask_zoom]*3, axis=-1)
overlay_final = overlay_rgb.copy()
overlay_final[mask3] = (alpha * overlay_rgb[mask3] + (1 - alpha) * np.stack([band_zoom]*3, axis=-1)[mask3]).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(overlay_final)
plt.title("Overlay con zoom x{}".format(zoom_factor))
plt.axis("off")
plt.tight_layout()
plt.show()
