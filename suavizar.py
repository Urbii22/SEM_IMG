
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np  # Necesario para la manipulación de datos de la imagen

def seleccionar_imagen():
    """
    Abre un diálogo para seleccionar un archivo de imagen.
    Devuelve la ruta seleccionada o None si se canceló.
    """
    root = tk.Tk()
    root.withdraw()

    ruta = filedialog.askopenfilename(
        title="Selecciona la imagen",
        filetypes=[
            ("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"),
            ("Todos los archivos", "*.*")
        ]
    )

    if not ruta:
        return None
    return ruta

def guardar_imagen_suavizada(imagen_suavizada, ruta_original):
    """
    Abre un diálogo para guardar la imagen suavizada, aplicándole el mapa de color 'viridis' de matplotlib.
    """
    carpeta, nombre_fichero = os.path.split(ruta_original)
    nombre_base, ext = os.path.splitext(nombre_fichero)
    nombre_defecto = f"{nombre_base}_suavizada_colormap{ext}"

    ruta_destino = filedialog.asksaveasfilename(
        title="Guardar imagen con mapa de color como...",
        initialdir=carpeta,
        initialfile=nombre_defecto,
        defaultextension=".png", # Se recomienda PNG para guardar con transparencia (si la hubiera)
        filetypes=[
            ("PNG", "*.png"),
            ("JPEG", "*.jpg;*.jpeg"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tiff"),
        ]
    )

    if not ruta_destino:
        return None

    # --- INICIO DE LA LÓGICA PARA APLICAR EL MAPA DE COLOR ---
    try:
        # 1. Asegurarse de que la imagen suavizada esté en escala de grises ('L') para poder aplicarle un colormap.
        imagen_gris = imagen_suavizada.convert('L')

        # 2. Convertir la imagen de Pillow a un array de NumPy.
        array_gris = np.array(imagen_gris)

        # 3. Obtener el colormap 'viridis' de matplotlib (el de tonos amarillos y morados).
        colormap = plt.get_cmap('viridis')

        # 4. Aplicar el colormap al array. Matplotlib necesita que los valores estén normalizados (entre 0 y 1).
        array_coloreado_rgba = colormap(array_gris / 255.0)

        # 5. Convertir los valores de vuelta a un rango de 8 bits (0-255) que Pillow puede manejar.
        array_coloreado_8bit = (array_coloreado_rgba * 255).astype(np.uint8)

        # 6. Crear una nueva imagen de Pillow a partir del array coloreado.
        imagen_final_a_guardar = Image.fromarray(array_coloreado_8bit, 'RGBA')
        
        # 7. Si el usuario quiere guardar como JPG (que no soporta transparencia), convertir a RGB.
        if ruta_destino.lower().endswith(('.jpg', '.jpeg')):
            imagen_final_a_guardar = imagen_final_a_guardar.convert('RGB')

        # 8. Guardar la imagen final.
        imagen_final_a_guardar.save(ruta_destino)
        return ruta_destino
        
    except Exception as e:
        messagebox.showerror("Error al guardar", f"No se pudo aplicar el colormap y guardar la imagen:\n{e}")
        return None


def mostrar_comparacion(original, suavizada):
    """
    Muestra por pantalla, lado a lado, la imagen original y la suavizada.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original (Color Real)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    # Usamos el mismo colormap aquí para que la vista previa y el guardado coincidan
    plt.imshow(suavizada, cmap='viridis')
    plt.title("Suavizada (Vista con Colormap)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    ruta = seleccionar_imagen()
    if ruta is None:
        print("Operación cancelada: No se seleccionó ninguna imagen.")
        return

    try:
        imagen_original = Image.open(ruta)
    except Exception as e:
        messagebox.showerror("Error al abrir la imagen", f"No se pudo abrir el archivo:\n{e}")
        return

    root_dialog = tk.Tk()
    root_dialog.withdraw()

    texto_radio = simpledialog.askstring(
        "Radio de suavizado",
        "Introduce el radio de desenfoque (deja en blanco o pulsa enter para usar 2):",
        parent=root_dialog
    )
    
    root_dialog.destroy()

    if texto_radio is None:
        print("Operación cancelada por el usuario.")
        return

    try:
        radio = float(texto_radio) if texto_radio.strip() else 2.0
    except ValueError:
        messagebox.showwarning("Valor no válido", "El radio introducido no es un número. Se usará 2.0 por defecto.")
        radio = 2.0

    # Aplicamos el filtro Gaussian Blur. El resultado será usado para guardado y para mostrar.
    imagen_suavizada_pil = imagen_original.filter(ImageFilter.GaussianBlur(radius=radio))
    
    ruta_guardado = guardar_imagen_suavizada(imagen_suavizada_pil, ruta)
    
    if ruta_guardado:
        messagebox.showinfo("Guardado correcto", f"Imagen con mapa de color guardada en:\n{ruta_guardado}")
        
        respuesta = messagebox.askyesno("Mostrar comparación", "¿Quieres ver la comparación entre la original y la suavizada?")
        if respuesta:
            # Para mostrar, usamos la imagen original y la suavizada (matplotlib aplicará el colormap en la vista previa)
            mostrar_comparacion(imagen_original, imagen_suavizada_pil)

if __name__ == "__main__":
    main()
