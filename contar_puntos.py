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
    # Umbrales originales:
    # rojo_bajo1 = np.array([0, 100, 100], np.uint8)
    # rojo_bajo2 = np.array([170, 100, 100], np.uint8)
    

    #INERVALOS PARA GOTA VITIPEC WEST (WUPVIT) dividida en SUP e INF
    
    # Intervalo que captura el 100 % de los puntos azules de la imagen WUPVIT_SUP (AL) azul
    #azul_bajo  = np.array([110, 150,  40], np.uint8)   # H≈110-130, S≥150, V≥40
    #azul_alto  = np.array([130, 255, 255], np.uint8)
    #mascara_azul = cv2.inRange(imagen_hsv, azul_bajo, azul_alto)
    
    # Intervalo que captura el 100 % de los puntos azules de la imagen WUPVIT_INF (AL) amarillo
    #amarillo_bajo = np.array([25, 150, 40],  np.uint8)   # H: 25-35  (≈50-70°)
    #amarillo_alto = np.array([35, 255, 255], np.uint8)

    #mascara_amarillo = cv2.inRange(imagen_hsv, amarillo_bajo, amarillo_alto)
    
    
    
    # Intervalo que captura el 100 % de los puntos azules de la imagen WUPVIT_SUP (S) amarillo
    #amarillo_bajo = np.array([28, 200, 40],  np.uint8)   # H: 28-32, S≥200, V≥40
    #amarillo_alto = np.array([32, 255, 255], np.uint8)

    #mascara_amarillo = cv2.inRange(imagen_hsv, amarillo_bajo, amarillo_alto)
    
    
    # Intervalo que captura el 100 % de los puntos azules de la imagen WUPVIT_INF (S) morado
    #morado_bajo = np.array([130, 150,  25], np.uint8)   # H: 130-140,  S≥150, V≥25
    #morado_alto = np.array([140, 255, 255], np.uint8)

    #mascara_morado = cv2.inRange(imagen_hsv, morado_bajo, morado_alto)
    

    
 
    #INERVALOS PARA GOTA VITIPEC EAST (WUPVIT)
    
    
    
    #Intervalo que captura el 100 % de los puntos azules de la imagen EUPVIT (S) violeta
    violeta_bajo = np.array([130, 180, 15], np.uint8)   # H 130-140, S≥180, V≥15
    violeta_alto = np.array([140, 255, 255], np.uint8)

    mascara = cv2.inRange(imagen_hsv, violeta_bajo, violeta_alto)
    
    
    #Intervalo que captura el 100 % de los puntos azules de la imagen EUPVIT (AL) cian
    #cian_bajo = np.array([90, 150, 15],  np.uint8)   # H 90-105, S≥150, V≥15
    #cian_alto = np.array([105, 255, 255], np.uint8)

    #mascara = cv2.inRange(imagen_hsv, cian_bajo, cian_alto)
    
 
   
    
    
    

    # Umbrales modificados para mayor sensibilidad a rojos claros:
    # Se reduce el mínimo de Saturación (S) y Valor (V)
    #rojo_bajo1 = np.array([0, 70, 70], np.uint8)  # S y V reducidos desde 100
    #rojo_alto1 = np.array([10, 255, 255], np.uint8)
    #rojo_bajo2 = np.array([170, 70, 70], np.uint8) # S y V reducidos desde 100
    #rojo_alto2 = np.array([179, 255, 255], np.uint8)
    #mascara1 = cv2.inRange(imagen_hsv, rojo_bajo1, rojo_alto1)
    #mascara2 = cv2.inRange(imagen_hsv, rojo_bajo2, rojo_alto2)
    #mascara_rojo = cv2.add(mascara1, mascara2)

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

            roi_mascara = mascara[y1:y2, x1:x2]
            pixeles_rojos = cv2.countNonZero(roi_mascara)
            total_pixeles_celda = roi_mascara.size
            porcentaje = (pixeles_rojos / total_pixeles_celda) * 100 if total_pixeles_celda > 0 else 0
            
            fila_de_porcentajes.append("{:.4f}".format(porcentaje).replace('.', ','))
        
        resultados_matriz.append(fila_de_porcentajes)

    # --- 4. Guardado en fichero CSV (Formato Matriz) ---
    try:
        with open(ruta_csv_salida, 'w', newline='', encoding='utf-8') as f:
            escritor = csv.writer(f, delimiter=';')
            escritor.writerows(resultados_matriz)
        print(f"✅ ¡Análisis completado! Resultados guardados en: {ruta_csv_salida}")
    except IOError as e:
        print(f"Error: No se pudo escribir en el fichero CSV. {e}")


if __name__ == '__main__':
    DIMENSIONES = (14, 18)

    root = tk.Tk()
    root.withdraw()

    ruta_imagen = filedialog.askopenfilename(
        title="Selecciona una imagen para analizar",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("Todos los ficheros", "*.*")]
    )

    if not ruta_imagen:
        print("Operación cancelada. No se seleccionó ninguna imagen.")
    else:
        nombre_base_imagen, _ = os.path.splitext(os.path.basename(ruta_imagen))
        nombre_csv_sugerido = f"{nombre_base_imagen}_analisis_matriz_{DIMENSIONES[0]}x{DIMENSIONES[1]}.csv"
        
        # Abrir diálogo para seleccionar dónde guardar el CSV
        ruta_csv_salida = filedialog.asksaveasfilename(
            title="Guardar archivo CSV como...",
            initialfile=nombre_csv_sugerido,
            defaultextension=".csv",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los ficheros", "*.*")]
        )

        if not ruta_csv_salida:
            print("Operación cancelada. No se seleccionó una ubicación para guardar el CSV.")
        else:
            analizar_imagen_matriz(
                ruta_imagen=ruta_imagen,
                dimensiones_grid=DIMENSIONES,
                ruta_csv_salida=ruta_csv_salida
            )