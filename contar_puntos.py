import cv2
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog
import os

def analizar_imagen_matriz(ruta_imagen, dimensiones_grid, ruta_csv_salida):
    """
    Analiza una imagen, la divide en una cuadrícula y guarda los porcentajes de rojo
    en un CSV con formato de matriz, reflejando la estructura de la cuadrícula.
    """
    print(f"Cargando la imagen desde: {ruta_imagen}")

    # --- CAMBIO IMPORTANTE: Método de lectura a prueba de caracteres especiales ---
    # En lugar de cv2.imread(ruta_imagen), leemos el fichero en memoria primero.
    # Esto evita problemas de codificación con rutas que contienen caracteres no ASCII.
    try:
        with open(ruta_imagen, 'rb') as f:
            # Leemos el fichero como un array de bytes
            buffer_bytes = np.fromfile(f, dtype=np.uint8)
        # Decodificamos la imagen desde el buffer de memoria
        # cv2.IMREAD_COLOR asegura que se cargue en color, incluso si tiene canal alfa.
        imagen = cv2.imdecode(buffer_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error crítico al intentar leer el fichero desde el buffer: {e}")
        imagen = None
    # --- FIN DEL CAMBIO ---

    if imagen is None:
        print(f"Error: No se pudo cargar la imagen. Comprueba la ruta y la integridad del fichero: {ruta_imagen}")
        return

    # --- 1. Definición del color rojo en el espacio de color HSV ---
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    rojo_bajo1 = np.array([0, 100, 100], np.uint8)
    rojo_alto1 = np.array([10, 255, 255], np.uint8)
    rojo_bajo2 = np.array([170, 100, 100], np.uint8)
    rojo_alto2 = np.array([179, 255, 255], np.uint8)
    mascara1 = cv2.inRange(imagen_hsv, rojo_bajo1, rojo_alto1)
    mascara2 = cv2.inRange(imagen_hsv, rojo_bajo2, rojo_alto2)
    mascara_rojo = cv2.add(mascara1, mascara2)

    # --- 2. Preparación de la cuadrícula ---
    altura_img, anchura_img, _ = imagen.shape
    filas_grid, columnas_grid = dimensiones_grid
    altura_celda = altura_img // filas_grid
    anchura_celda = anchura_img // columnas_grid

    print(f"Dimensiones de la imagen: {anchura_img}x{altura_img} píxeles")
    print(f"Dimensiones de la cuadrícula: {columnas_grid}x{filas_grid} celdas")
    print(f"Tamaño de cada celda: {anchura_celda}x{altura_celda} píxeles")
    
    resultados_matriz = []

    # --- 3. Análisis de cada celda ---
    for fila in range(filas_grid):
        fila_de_porcentajes = []
        for col in range(columnas_grid):
            y1, y2 = fila * altura_celda, (fila + 1) * altura_celda
            x1, x2 = col * anchura_celda, (col + 1) * anchura_celda

            roi_mascara = mascara_rojo[y1:y2, x1:x2]
            pixeles_rojos = cv2.countNonZero(roi_mascara)
            total_pixeles_celda = roi_mascara.size
            porcentaje_rojo = (pixeles_rojos / total_pixeles_celda) * 100 if total_pixeles_celda > 0 else 0
            
            fila_de_porcentajes.append(round(porcentaje_rojo, 4))
        
        resultados_matriz.append(fila_de_porcentajes)

    # --- 4. Guardado en fichero CSV (Formato Matriz) ---
    try:
        with open(ruta_csv_salida, 'w', newline='', encoding='utf-8') as f:
            escritor = csv.writer(f)
            escritor.writerows(resultados_matriz)
        print(f"✅ ¡Análisis completado! Resultados guardados en: {ruta_csv_salida}")
    except IOError as e:
        print(f"Error: No se pudo escribir en el fichero CSV. {e}")


if __name__ == '__main__':
    DIMENSIONES = (20, 30)

    root = tk.Tk()
    root.withdraw()

    ruta_imagen = filedialog.askopenfilename(
        title="Selecciona una imagen para analizar",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("Todos los ficheros", "*.*")]
    )

    if not ruta_imagen:
        print("Operación cancelada. No se seleccionó ninguna imagen.")
    else:
        nombre_base, _ = os.path.splitext(os.path.basename(ruta_imagen))
        ruta_salida = f"{nombre_base}_analisis_matriz_{DIMENSIONES[0]}x{DIMENSIONES[1]}.csv"
        
        analizar_imagen_matriz(
            ruta_imagen=ruta_imagen,
            dimensiones_grid=DIMENSIONES,
            ruta_csv_salida=ruta_salida
        )