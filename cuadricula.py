import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog

# Seleccionar la imagen con un cuadro de diálogo
root = Tk()
root.withdraw()
imagen_path = filedialog.askopenfilename(
    title="Selecciona la imagen",
    filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos los archivos", "*.*")]
)
root.destroy()
if not imagen_path:
    raise SystemExit("No se seleccionó ninguna imagen.")

# 1. Abrir la imagen con Pillow
img = Image.open(imagen_path)
ancho, alto = img.size
arr = np.array(img)

# 2. Determinar mapa de color
if arr.ndim == 2:
    cmap = 'gray'
else:
    cmap = None

# 3. Crear figura y dibujar imagen + cuadrícula
plt.figure(figsize=(ancho / 30, alto / 30))
plt.imshow(arr, cmap=cmap, interpolation='nearest')
for x in range(ancho + 1):
    plt.axvline(x - 0.5, color='red', linewidth=0.5)
for y in range(alto + 1):
    plt.axhline(y - 0.5, color='red', linewidth=0.5)
plt.xlim(-0.5, ancho - 0.5)
plt.ylim(alto - 0.5, -0.5)
plt.axis('off')

# 4. Abrir diálogo para guardar la imagen en local
root = Tk()
root.withdraw()  # Oculta la ventana principal
ruta_guardado = filedialog.asksaveasfilename(
    defaultextension=".png",
    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")],
    title="Guardar imagen con cuadrícula"
)
if ruta_guardado:
    plt.savefig(ruta_guardado, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Imagen guardada en: {ruta_guardado}")
else:
    print("Guardado cancelado por el usuario.")

# 5. Cerrar la figura
plt.close()