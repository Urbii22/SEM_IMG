import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ----------------------------------------------------------
# 1. Seleccionar manualmente las imágenes mediante diálogo
# ----------------------------------------------------------
# Ocultamos la ventana principal de tkinter
root = Tk()
root.withdraw()

# Pedimos la primera imagen (óptica recortada)
print("Selecciona la imagen Óptica (recorte):")
opt_path_str = askopenfilename(
    title="Selecciona la imagen Óptica (recorte)",
    filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("Todos los archivos", "*.*")]
)
if not opt_path_str:
    raise RuntimeError("No se seleccionó ninguna imagen óptica. Saliendo.")

# Pedimos la segunda imagen (SEM)
print("Selecciona la imagen SEM (microscopía electrónica):")
sem_path_str = askopenfilename(
    title="Selecciona la imagen SEM (microscopía electrónica)",
    filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("Todos los archivos", "*.*")]
)
if not sem_path_str:
    raise RuntimeError("No se seleccionó ninguna imagen SEM. Saliendo.")

opt_path = Path(opt_path_str)
sem_path = Path(sem_path_str)

# ----------------------------------------------------------
# 2. Cargar las imágenes seleccionadas
# ----------------------------------------------------------
opt = cv2.imread(str(opt_path))
sem = cv2.imread(str(sem_path))

if opt is None or sem is None:
    raise FileNotFoundError("Alguna de las imágenes no se pudo leer. ¿Has seleccionado un formato compatible?")

# ----------------------------------------------------------
# 3. Función para detectar la barra de escala (1 mm) en SEM
# ----------------------------------------------------------
def detectar_barra_escala(sem_img, crop_frac=0.15, thresh_val=200):
    """
    Busca la barra de escala (asumida blanca) en la parte inferior de la SEM,
    y devuelve su anchura en píxeles. Ajusta 'crop_frac' (fracción de altura inferior)
    y 'thresh_val' (umbral de binarización) si no la detecta bien.
    """
    h, w = sem_img.shape[:2]
    # Recortamos solo la franja inferior (crop_frac) donde suele estar la barra
    crop = sem_img[int(h * (1 - crop_frac)) : h, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binar = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    # Cerramos huecos horizontales para unir la barra completa
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, kernel)
    conts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        raise RuntimeError("No se detectó la barra de escala (ajusta 'thresh_val' o 'crop_frac').")
    # Tomamos el contorno más ancho (la barra)
    cnt = max(conts, key=lambda c: cv2.boundingRect(c)[2])
    _, _, w_bar, _ = cv2.boundingRect(cnt)
    return w_bar

# Detectamos la barra y calculamos píxeles por milímetro
px_bar = detectar_barra_escala(sem)
px_per_mm = px_bar / 1.0

# ----------------------------------------------------------
# 4. Función para detectar la gota y calcular círculo mínimo
# ----------------------------------------------------------
def detectar_gota_min_circle(img, invert=False, blur_size=5, thresh_val=80):
    """
    Detecta la gota en 'img' devolviendo:
     - centro (x, y),
     - radio del círculo mínimo que la encierra,
     - contorno (lista de puntos).
     
    Parámetros:
    - invert: True si la gota aparece oscura sobre fondo claro (THRESH_BINARY_INV),
              False si aparece clara sobre fondo más oscuro (THRESH_BINARY).
    - blur_size: tamaño del blur antes de umbralizar.
    - thresh_val: valor de umbral; ajusta si la segmentación es incorrecta.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binar = cv2.threshold(gray_blur, thresh_val, 255, flag)
    # Eliminamos ruido pequeño con apertura morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binar = cv2.morphologyEx(binar, cv2.MORPH_OPEN, kernel, iterations=2)
    conts, _ = cv2.findContours(binar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        raise RuntimeError("No se encontró ningún contorno de gota (ajusta 'thresh_val').")
    # Tomamos el contorno de mayor área
    cnt = max(conts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(cnt)
    return (x, y), r, cnt

# Detectamos en la imagen óptica (gota oscura → invert=True)
(opt_c, opt_r, cnt_opt) = detectar_gota_min_circle(opt,  invert=True,  thresh_val=60)

# Detectamos en la imagen SEM (gota clara → invert=False)
(sem_c, sem_r, cnt_sem) = detectar_gota_min_circle(sem,  invert=False, thresh_val=100)

# ----------------------------------------------------------
# 5. Calcular factor de escala y rotación necesaria
# ----------------------------------------------------------
scale = (2 * opt_r) / (2 * sem_r)  # diámetro óptica / diámetro SEM

# Intentamos calcular diferencia de ángulo con fitEllipse; si falla, asumimos 0º
try:
    ang_opt = cv2.fitEllipse(cnt_opt)[2]
    ang_sem = cv2.fitEllipse(cnt_sem)[2]
    d_angle = ang_opt - ang_sem
except:
    d_angle = 0.0

# ----------------------------------------------------------
# 6. Transformar la imagen SEM: escala, rotación y traslación
# ----------------------------------------------------------
# 6.1 Escalar SEM
sem_scaled = cv2.resize(sem, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

# Actualizar coordenadas del centro tras escalar
xs, ys = sem_c
xs_scaled, ys_scaled = xs * scale, ys * scale

# 6.2 Rotar SEM alrededor de su centro escalado
M = cv2.getRotationMatrix2D((xs_scaled, ys_scaled), d_angle, 1.0)
h_s, w_s = sem_scaled.shape[:2]
sem_rot = cv2.warpAffine(sem_scaled, M, (w_s, h_s), flags=cv2.INTER_CUBIC)

# 6.3 Trasladar SEM para alinear el centro con el de la óptica
xo, yo = opt_c
tx, ty = int(xo - xs_scaled), int(yo - ys_scaled)

# Creamos un lienzo (canvas) del mismo tamaño que la imagen óptica
canvas = opt.copy()
h_r, w_r = sem_rot.shape[:2]

# Definimos región de interés en el canvas y en la SEM rotada (respetando límites)
y1, y2 = max(0, ty), min(canvas.shape[0], ty + h_r)
x1, x2 = max(0, tx), min(canvas.shape[1], tx + w_r)
y1_s, x1_s = y1 - ty, x1 - tx
y2_s, x2_s = y1_s + (y2 - y1), x1_s + (x2 - x1)

# Mezclamos con transparencia α = 0.4
alpha = 0.4
roi_canvas = canvas[y1:y2, x1:x2]
roi_sem = sem_rot[y1_s:y2_s, x1_s:x2_s]
if roi_canvas.shape[:2] == roi_sem.shape[:2]:
    blended = cv2.addWeighted(roi_canvas, 1 - alpha, roi_sem, alpha, 0)
    canvas[y1:y2, x1:x2] = blended
else:
    print("Aviso: las dimensiones de la región a mezclar no coinciden. No se superpone SEM.")

# ----------------------------------------------------------
# 7. Guardar y mostrar resultado final
# ----------------------------------------------------------
overlay_path = Path.cwd() / "overlay.png"
cv2.imwrite(str(overlay_path), canvas)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Superposición Óptica + SEM")
plt.show()

print(f"¡Listo! La imagen superpuesta se ha guardado en:")
