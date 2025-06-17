#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación independiente para recortar una ROI de un cubo hiperespectral ENVI
y guardar:
  1) El cubo recortado en nuevos archivos .bil y .bil.hdr.
  2) Una imagen JPEG de la misma ROI (tomada de la banda visualizada).
  3) Un archivo CSV con los valores de los píxeles de la ROI para la banda visualizada.

Requisitos:
    pip install spectral opencv-python numpy tkinter pillow

Funcionamiento:
    1. Abre un diálogo para seleccionar el archivo .hdr (o .bil).
    2. Carga el cubo y extrae la banda más cercana a 728 nm.
    3. Permite al usuario dibujar un rectángulo para definir la ROI.
    4. Al pulsar ENTER, recorta el cubo según la ROI.
    5. Pide una carpeta de salida.
    6. Guarda los archivos recortados:
       - <nombre_original>_recorte.bil
       - <nombre_original>_recorte.hdr
       - <nombre_original>_recorte.jpg
       - <nombre_original>_recorte_banda_XXXnm.csv
"""

import cv2
import numpy as np
import spectral as spy
from spectral import envi
from pathlib import Path
from tkinter import Tk, filedialog
import sys
from PIL import Image

# Dimensiones fijas para la ROI (ancho, alto en píxeles originales)
FIXED_ROI_WIDTH_ORIG = 40
FIXED_ROI_HEIGHT_ORIG = 32


# Variables globales para la selección interactiva de ROI
_select_roi_params = {
    "current_roi_display_coords": None,  # (x, y, w, h) en coordenadas de la imagen zoomeada completa
    "drawing": False,
    "start_point_display_coords": (-1, -1), # Coordenadas en el canvas de visualización
    "current_mouse_pos_display": (-1,-1), # Posición actual del ratón en el canvas
    "zoom_factor": 1.0,
    "original_image_ref": None,          # Imagen original (banda_uint8)
    "temp_display_image": None,          # Imagen para mostrar con dibujos temporales (el canvas)
    "window_name": "Selecciona ROI",
    "pan_offset_display": (0.0, 0.0),    # (pan_x, pan_y) de la imagen zoomeada en el canvas
    "panning": False,                    # True si se está haciendo paneo con click derecho
    "pan_start_mouse_pos_canvas": (-1,-1),# Posición del ratón en canvas al iniciar paneo
    "pan_start_offset_display": (0.0, 0.0),# pan_offset_display al iniciar paneo
    "display_view_size": (None, None)    # (width, height) del canvas de visualización
}

def limit_pan_offset(pan_offset, zoom_factor, original_img_size, display_view_size):
    """Limita el pan_offset para que la imagen zoomeada no se salga demasiado del display_view."""
    pan_x, pan_y = pan_offset
    orig_w, orig_h = original_img_size # Dimensiones de la imagen original sin zoom
    view_w, view_h = display_view_size # Dimensiones del canvas

    scaled_w = orig_w * zoom_factor
    scaled_h = orig_h * zoom_factor

    # pan_x es la coordenada X de la esquina superior izquierda de la imagen zoomeada,
    # relativa a la esquina superior izquierda del canvas.
    # Si scaled_w < view_w, la imagen es más pequeña que la vista. pan_x puede ir de 0 a view_w - scaled_w.
    # Si scaled_w > view_w, la imagen es más grande. pan_x puede ir de view_w - scaled_w a 0.
    
    min_pan_x = min(0.0, view_w - scaled_w)
    max_pan_x = max(0.0, view_w - scaled_w)
    pan_x = np.clip(pan_x, min_pan_x, max_pan_x)

    min_pan_y = min(0.0, view_h - scaled_h)
    max_pan_y = max(0.0, view_h - scaled_h)
    pan_y = np.clip(pan_y, min_pan_y, max_pan_y)
    
    return pan_x, pan_y

def mouse_callback_interactive(event, x, y, flags, param):
    """Callback del ratón para la selección interactiva de ROI."""
    p = _select_roi_params
    prev_zoom_factor = p["zoom_factor"]
    orig_w, orig_h = p["original_image_ref"].shape[1], p["original_image_ref"].shape[0]

    # Actualizar siempre la posición del ratón en el canvas
    p["current_mouse_pos_display"] = (x,y)

    if event == cv2.EVENT_LBUTTONDOWN:
        p["drawing"] = True
        p["start_point_display_coords"] = (x, y) # Coords en canvas
        p["current_roi_display_coords"] = None  # Resetear ROI al empezar nuevo dibujo

    elif event == cv2.EVENT_MOUSEMOVE:
        if p["drawing"]:
            pass # La lógica de dibujo y texto se maneja en el bucle principal
        elif p["panning"]:
            start_mouse_x_canvas, start_mouse_y_canvas = p["pan_start_mouse_pos_canvas"]
            start_pan_x, start_pan_y = p["pan_start_offset_display"]
            
            dx_canvas = x - start_mouse_x_canvas
            dy_canvas = y - start_mouse_y_canvas
            
            new_pan_x = start_pan_x + dx_canvas
            new_pan_y = start_pan_y + dy_canvas
            
            p["pan_offset_display"] = limit_pan_offset(
                (new_pan_x, new_pan_y), 
                p["zoom_factor"],
                (orig_w, orig_h),
                p["display_view_size"]
            )

    elif event == cv2.EVENT_LBUTTONUP:
        if p["drawing"]: # Finalizar dibujo de ROI
            p["drawing"] = False
            end_point_canvas_coords = (x, y)
            x1_can, y1_can = p["start_point_display_coords"]
            x2_can, y2_can = end_point_canvas_coords
            
            roi_x_canvas = min(x1_can, x2_can)
            roi_y_canvas = min(y1_can, y2_can)
            # Usar dimensiones fijas para la ROI, escaladas por el zoom
            roi_w_canvas = int(FIXED_ROI_WIDTH_ORIG * p["zoom_factor"])
            roi_h_canvas = int(FIXED_ROI_HEIGHT_ORIG * p["zoom_factor"])

            if roi_w_canvas > 0 and roi_h_canvas > 0: # Siempre será true con valores fijos > 0
                # Convertir ROI de coordenadas de canvas a coordenadas de imagen zoomeada
                current_pan_x, current_pan_y = p["pan_offset_display"]
                roi_x_zoomed = roi_x_canvas - current_pan_x
                roi_y_zoomed = roi_y_canvas - current_pan_y
                p["current_roi_display_coords"] = tuple(map(int,[roi_x_zoomed, roi_y_zoomed, roi_w_canvas, roi_h_canvas]))
            else:
                p["current_roi_display_coords"] = None
        elif p["panning"]: # Finalizar paneo
            p["panning"] = False
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        p["panning"] = True
        p["pan_start_mouse_pos_canvas"] = (x,y)
        p["pan_start_offset_display"] = p["pan_offset_display"]

    elif event == cv2.EVENT_RBUTTONUP: # También puede ocurrir si se suelta fuera de la ventana
        p["panning"] = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        cursor_x_canvas, cursor_y_canvas = x, y
        pan_x_old, pan_y_old = p["pan_offset_display"]
        
        # Punto en imagen original bajo el cursor
        orig_point_x = (cursor_x_canvas - pan_x_old) / prev_zoom_factor
        orig_point_y = (cursor_y_canvas - pan_y_old) / prev_zoom_factor

        if flags > 0:  # Rueda hacia arriba (Zoom In)
            p["zoom_factor"] *= 1.1
        else:  # Rueda hacia abajo (Zoom Out)
            p["zoom_factor"] /= 1.1
        p["zoom_factor"] = max(0.1, min(p["zoom_factor"], 20)) # Limitar zoom
        new_zoom_factor = p["zoom_factor"]

        # Nuevo pan_offset para mantener orig_point bajo el cursor
        new_pan_x = cursor_x_canvas - orig_point_x * new_zoom_factor
        new_pan_y = cursor_y_canvas - orig_point_y * new_zoom_factor
        p["pan_offset_display"] = limit_pan_offset(
            (new_pan_x, new_pan_y), new_zoom_factor, (orig_w, orig_h), p["display_view_size"]
        )

        if p["current_roi_display_coords"]:
            rx_z, ry_z, rw_z, rh_z = p["current_roi_display_coords"]
            factor_cambio = new_zoom_factor / prev_zoom_factor
            
            # Centro del ROI en coordenadas de imagen original
            center_x_orig = (rx_z + rw_z / 2) / prev_zoom_factor
            center_y_orig = (ry_z + rh_z / 2) / prev_zoom_factor
            
            # Nuevas dimensiones y centro del ROI en la imagen zoomeada con new_zoom_factor
            new_rw_z = int(FIXED_ROI_WIDTH_ORIG * new_zoom_factor)
            new_rh_z = int(FIXED_ROI_HEIGHT_ORIG * new_zoom_factor)
            new_center_x_z = center_x_orig * new_zoom_factor
            new_center_y_z = center_y_orig * new_zoom_factor
            
            new_rx_z = new_center_x_z - new_rw_z / 2
            new_ry_z = new_center_y_z - new_rh_z / 2
            
            p["current_roi_display_coords"] = tuple(map(int, [new_rx_z, new_ry_z, new_rw_z, new_rh_z]))


def select_roi_interactive_custom(image_to_select_on):
    """Permite al usuario seleccionar una ROI de forma interactiva con zoom y visualización de dimensiones."""
    p = _select_roi_params
    p["original_image_ref"] = image_to_select_on.copy()
    
    h_orig, w_orig = p["original_image_ref"].shape[:2]
    
    # Limitar el tamaño de la ventana de visualización para mejorar la usabilidad en pantallas grandes/pequeñas
    # y asegurar que el paneo funcione correctamente.
    MAX_VIEW_W, MAX_VIEW_H = 1280, 720 # Puedes ajustar estos valores si es necesario
    
    display_view_w = min(w_orig, MAX_VIEW_W)
    display_view_h = min(h_orig, MAX_VIEW_H)
    p["display_view_size"] = (display_view_w, display_view_h)

    p["window_name"] = "ROI (Z/X/Rueda: Zoom, ClickDer+Arrastrar: Mover, R: Reset, ENTER: OK, ESC: Cancelar)"
    p["zoom_factor"] = 1.0
    p["pan_offset_display"] = (0.0, 0.0)
    p["current_roi_display_coords"] = None
    p["drawing"] = False
    p["panning"] = False
    p["start_point_display_coords"] = (-1,-1)
    p["current_mouse_pos_display"] = (-1,-1)

    cv2.namedWindow(p["window_name"]) # WINDOW_AUTOSIZE por defecto, se ajustará a display_view_size
    cv2.setMouseCallback(p["window_name"], mouse_callback_interactive)

    display_view_w, display_view_h = p["display_view_size"]

    while True:
        # Imagen original escalada al zoom_factor actual
        scaled_img_w = int(w_orig * p["zoom_factor"])
        scaled_img_h = int(h_orig * p["zoom_factor"])
        
        if scaled_img_w < 1: scaled_img_w = 1
        if scaled_img_h < 1: scaled_img_h = 1
        
        scaled_image = cv2.resize(p["original_image_ref"], (scaled_img_w, scaled_img_h), interpolation=cv2.INTER_LINEAR)

        # Crear canvas del tamaño de display_view_size
        # Asumimos que original_image_ref es monocromática (banda_uint8)
        canvas = np.zeros((display_view_h, display_view_w), dtype=p["original_image_ref"].dtype)
        
        current_pan_x, current_pan_y = p["pan_offset_display"]

        # Calcular la región de scaled_image (src) a copiar y dónde en canvas (dst)
        src_x = int(max(0, -current_pan_x))
        src_y = int(max(0, -current_pan_y))
        dst_x = int(max(0, current_pan_x))
        dst_y = int(max(0, current_pan_y))

        copy_w = min(scaled_img_w - src_x, display_view_w - dst_x)
        copy_h = min(scaled_img_h - src_y, display_view_h - dst_y)

        if copy_w > 0 and copy_h > 0:
            canvas[dst_y:dst_y+copy_h, dst_x:dst_x+copy_w] = \
                scaled_image[src_y:src_y+copy_h, src_x:src_x+copy_w]
        
        # Convertir canvas a BGR para dibujar en color
        p["temp_display_image"] = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Dibujar rectángulo mientras se arrastra (coordenadas de canvas)
        if p["drawing"] and p["start_point_display_coords"] != (-1,-1):
            start_x_canvas, start_y_canvas = p["start_point_display_coords"]
            rect_w_canvas = int(FIXED_ROI_WIDTH_ORIG * p["zoom_factor"])
            rect_h_canvas = int(FIXED_ROI_HEIGHT_ORIG * p["zoom_factor"])
            
            end_x_canvas = start_x_canvas + rect_w_canvas
            end_y_canvas = start_y_canvas + rect_h_canvas
            
            cv2.rectangle(p["temp_display_image"], 
                          (start_x_canvas, start_y_canvas), 
                          (end_x_canvas, end_y_canvas), 
                          (0, 255, 0), 1)
            text_rt = f"Placing: W:{FIXED_ROI_WIDTH_ORIG} H:{FIXED_ROI_HEIGHT_ORIG}"
            cv2.putText(p["temp_display_image"], text_rt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Dibujar ROI seleccionada (convertir de zoomeado a canvas)
        elif p["current_roi_display_coords"]:
            rx_z, ry_z, rw_z, rh_z = map(int,p["current_roi_display_coords"])
            # Coordenadas para dibujar en el canvas
            draw_rx = int(rx_z + current_pan_x)
            draw_ry = int(ry_z + current_pan_y)
            cv2.rectangle(p["temp_display_image"], (draw_rx, draw_ry), (draw_rx + rw_z, draw_ry + rh_z), (0, 0, 255), 2)
            
            # Las dimensiones originales son fijas
            text_final = f"Selected: W:{FIXED_ROI_WIDTH_ORIG} H:{FIXED_ROI_HEIGHT_ORIG}"
            cv2.putText(p["temp_display_image"], text_final, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

        info_text = f"Zoom:{p['zoom_factor']:.2f}x (Z/X/Rueda). Pan:ClickDer+Arr. R:Reset. ENTER:OK. ESC:Cancel."
        cv2.putText(p["temp_display_image"], info_text, (10, p["temp_display_image"].shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(p["window_name"], p["temp_display_image"])
        k = cv2.waitKey(20) & 0xFF

        if k == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        elif k == 13:  # Enter
            if p["current_roi_display_coords"]:
                rx_z, ry_z, rw_z, rh_z = p["current_roi_display_coords"]
                # Convertir ROI de coordenadas de imagen zoomeada a coordenadas de imagen original
                orig_x = int(rx_z / p["zoom_factor"])
                orig_y = int(ry_z / p["zoom_factor"])
                # Usar dimensiones fijas para la ROI original
                orig_w = FIXED_ROI_WIDTH_ORIG
                orig_h = FIXED_ROI_HEIGHT_ORIG

                # Asegurar que las coordenadas originales están dentro de los límites de la imagen original
                # Y que la ROI de tamaño fijo cabe completamente.
                orig_x = max(0, orig_x)
                orig_y = max(0, orig_y)
                
                if orig_x + orig_w > w_orig or orig_y + orig_h > h_orig:
                    print(f"Advertencia: La ROI de {FIXED_ROI_WIDTH_ORIG}x{FIXED_ROI_HEIGHT_ORIG} no cabe completamente en la imagen desde esta posición ({orig_x},{orig_y}).")
                    print(f"Máxima posición para esquina superior izquierda: x={w_orig-orig_w}, y={h_orig-orig_h}")
                    print("Intenta de nuevo seleccionando una posición válida.")
                    p["current_roi_display_coords"] = None # Invalidar ROI para forzar nueva selección
                
                elif orig_w > 0 and orig_h > 0: # Siempre true para valores fijos > 0
                    cv2.destroyAllWindows()
                    return (orig_x, orig_y, orig_w, orig_h)
                else:
                    # Esto no debería ocurrir con valores fijos positivos
                    print("Advertencia: ROI inválida (tamaño cero o negativo). Intenta de nuevo.")
                    p["current_roi_display_coords"] = None 
            else:
                print("Ninguna ROI seleccionada.")
        
        elif k == ord('z') or k == ord('x'): 
            prev_zoom_factor = p["zoom_factor"]
            pan_x_old, pan_y_old = p["pan_offset_display"]

            # Determinar el punto de centrado para el zoom (centro del ROI o centro de la ventana)
            cursor_x_canvas, cursor_y_canvas = 0,0
            if p["current_roi_display_coords"]:
                rx_z_roi, ry_z_roi, rw_z_roi, rh_z_roi = p["current_roi_display_coords"]
                center_roi_x_z = rx_z_roi + rw_z_roi / 2
                center_roi_y_z = ry_z_roi + rh_z_roi / 2
                cursor_x_canvas = center_roi_x_z + pan_x_old
                cursor_y_canvas = center_roi_y_z + pan_y_old
            else:
                cursor_x_canvas = display_view_w / 2
                cursor_y_canvas = display_view_h / 2
            
            orig_point_x = (cursor_x_canvas - pan_x_old) / prev_zoom_factor
            orig_point_y = (cursor_y_canvas - pan_y_old) / prev_zoom_factor

            if k == ord('z'):
                p["zoom_factor"] *= 1.25
            else: # k == ord('x')
                p["zoom_factor"] /= 1.25
            p["zoom_factor"] = max(0.1, min(p["zoom_factor"], 20))
            new_zoom_factor = p["zoom_factor"]

            new_pan_x = cursor_x_canvas - orig_point_x * new_zoom_factor
            new_pan_y = cursor_y_canvas - orig_point_y * new_zoom_factor
            p["pan_offset_display"] = limit_pan_offset(
                (new_pan_x, new_pan_y), new_zoom_factor, (w_orig, h_orig), p["display_view_size"]
            )
            
            if p["current_roi_display_coords"]:
                rx_z, ry_z, rw_z, rh_z = p["current_roi_display_coords"] # rw_z, rh_z no se usan para calcular el nuevo tamaño
                # factor_cambio = new_zoom_factor / prev_zoom_factor # No necesario para dimensiones fijas
                
                center_x_orig_roi = (rx_z + (FIXED_ROI_WIDTH_ORIG * prev_zoom_factor) / 2) / prev_zoom_factor
                center_y_orig_roi = (ry_z + (FIXED_ROI_HEIGHT_ORIG * prev_zoom_factor) / 2) / prev_zoom_factor
                
                new_rw_z = int(FIXED_ROI_WIDTH_ORIG * new_zoom_factor)
                new_rh_z = int(FIXED_ROI_HEIGHT_ORIG * new_zoom_factor)
                new_center_x_z = center_x_orig_roi * new_zoom_factor
                new_center_y_z = center_y_orig_roi * new_zoom_factor
                
                new_rx_z = new_center_x_z - new_rw_z / 2
                new_ry_z = new_center_y_z - new_rh_z / 2
                p["current_roi_display_coords"] = tuple(map(int, [new_rx_z, new_ry_z, new_rw_z, new_rh_z]))

        elif k == ord('r'): 
            p["zoom_factor"] = 1.0
            p["pan_offset_display"] = (0.0, 0.0)
            p["current_roi_display_coords"] = None
            p["drawing"] = False
            p["panning"] = False
            p["start_point_display_coords"] = (-1,-1)

    cv2.destroyAllWindows()
    return None

def seleccionar_archivo(titulo: str, tipos: list) -> Path:
    """
    Abre un file dialog y retorna la ruta seleccionada como Path.
    Si el usuario cancela, termina el programa.
    """
    root = Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(title=titulo, filetypes=tipos)
    root.destroy()
    if not ruta:
        sys.exit("No se seleccionó ningún archivo. Saliendo.")
    return Path(ruta)

def seleccionar_carpeta(titulo: str) -> Path:
    """
    Abre un diálogo para seleccionar carpeta. Si el usuario cancela, sale.
    """
    root = Tk()
    root.withdraw()
    carpeta = filedialog.askdirectory(title=titulo)
    root.destroy()
    if not carpeta:
        sys.exit("No se seleccionó ninguna carpeta. Saliendo.")
    return Path(carpeta)

def main():
    # 1. Seleccionar archivo HSI (.hdr o .bil)
    hsi_path = seleccionar_archivo(
        "Selecciona el archivo .hdr (o .bil) del cubo hiperespectral",
        [("Archivos ENVI (HDR/BIL)", "*.hdr *.bil"), ("Todos", "*.*")]
    )
    hdr_path = hsi_path.with_suffix(".hdr") if hsi_path.suffix.lower() == ".bil" else hsi_path

    if not hdr_path.exists():
        sys.exit(f"No se encontró el archivo asociado: {hdr_path}")

    # 2. Cargar cubo HSI y extraer banda ~728 nm para visualización (una sola vez)
    print("-> Cargando cubo hiperespectral…")
    try:
        cube_hsi = spy.open_image(str(hdr_path))
        data = cube_hsi.load()

        # --- INICIO DE LA CORRECCIÓN DE ESCALA ---
        # La librería 'spectral' aplica automáticamente factores de escala.
        # Aquí, revertimos esa operación para trabajar con los valores DN originales.
        scale_factor_str = cube_hsi.metadata.get('reflectance scale factor')
        if scale_factor_str:
            try:
                scale_factor = float(scale_factor_str)
                if data.max() < 10.0: # Heurística: si los valores son pequeños, probablemente estén escalados
                    print(f"Factor de escala encontrado: {scale_factor}. Revirtiendo a valores DN originales...")
                    data *= scale_factor
            except (ValueError, TypeError):
                print("Aviso: no se pudo interpretar el factor de escala.")
        # --- FIN DE LA CORRECCIÓN DE ESCALA ---

    except Exception as e:
        sys.exit(f"Error cargando el cubo hiperespectral: {e}")

    TARGET_WL = 728.24
    band_idx = 0
    wl_str_original = "" # Para el nombre del archivo CSV
    meta_wl = cube_hsi.metadata.get("wavelength", [])
    if meta_wl:
        try:
            wl = np.array(meta_wl, dtype=float)
            band_idx = int(np.argmin(np.abs(wl - TARGET_WL)))
            wl_str_original = f"_{int(round(wl[band_idx]))}nm"
            print(f"Banda objetivo para visualización y CSV: índice {band_idx} (≈ {wl[band_idx]:.1f} nm)")
        except Exception:
            print("No se pudo interpretar metadata de longitudes de onda. Se usará banda 0 para visualización y CSV.")
    else:
        print("No hay metadata de longitudes de onda. Se usará la primera banda (0) para visualización y CSV.")

    banda_para_visualizar = data[:, :, band_idx]
    banda_uint8 = cv2.normalize(banda_para_visualizar, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Seleccionar carpeta de salida (una sola vez)
    carpeta_salida = seleccionar_carpeta("Selecciona carpeta donde guardar los recortes")
    base_nombre_original = hdr_path.stem
    
    recorte_count = 0

    while True:
        recorte_count += 1
        print(f"\n--- Iniciando recorte #{recorte_count} ---")

        # 4. Seleccionar ROI con el ratón de forma interactiva
        roi = select_roi_interactive_custom(banda_uint8)

        if not roi or roi[2] == 0 or roi[3] == 0:
            print("No se definió ninguna ROI válida o se canceló la selección.")
            if recorte_count == 1: # Si es el primer intento y se cancela, salir.
                 sys.exit("Saliendo de la aplicación.")
            else: # Si ya se hizo al menos un recorte, preguntar si quiere salir.
                respuesta_salir = input("¿Desea salir de la aplicación? (s/N): ").strip().lower()
                if respuesta_salir == 's':
                    sys.exit("Saliendo de la aplicación.")
                else:
                    recorte_count -=1 # No se completó este recorte, no incrementar contador
                    print("Intentando nueva selección de ROI...")
                    continue # Volver al inicio del bucle para seleccionar otra ROI

        x, y, w, h = roi
        print(f"ROI seleccionada: x={x}, y={y}, ancho={w}, alto={h}")

        # 5. Recortar el cubo
        recorte_datos = data[y : y + h, x : x + w, :].copy()
        rec_rows, rec_cols, rec_bands = recorte_datos.shape
        print(f"Cubo recortado con forma: {recorte_datos.shape} (filas, cols, bandas)")

        # 6. Generar nombre base para este recorte
        nombre_recorte_actual = f"{base_nombre_original}_recorte_{recorte_count}"
        
        # 7. Guardar el cubo recortado en formato ENVI (.bil y .hdr)
        print("-> Guardando cubo recortado en formato ENVI...")
        out_hdr_path = carpeta_salida / f"{nombre_recorte_actual}.hdr"
        out_bil_path = carpeta_salida / f"{nombre_recorte_actual}.bil"
        
        try:
            metanew = dict(cube_hsi.metadata)
            metanew["lines"] = rec_rows
            metanew["samples"] = rec_cols
            metanew["bands"] = rec_bands
            metanew.pop('reflectance scale factor', None)
            metanew.pop('data gain values', None)
            metanew.pop('data offset values', None)

            dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64, 12: np.uint16}
            original_dtype_code = int(cube_hsi.metadata.get('data type', 12))
            save_dtype = dtype_map.get(original_dtype_code, np.uint16)

            envi.save_image(str(out_hdr_path), recorte_datos.astype(save_dtype), force=True, metadata=metanew, interleave='bil')
            
            out_img_path = out_hdr_path.with_suffix('.img') # spectralpy puede crear .img en lugar de .bil
            if out_img_path.exists() and not out_bil_path.exists():
                out_img_path.rename(out_bil_path)
                # Actualizar el nombre del archivo de datos en el .hdr si es necesario
                hdr_content = out_hdr_path.read_text()
                if out_img_path.name in hdr_content:
                    hdr_content = hdr_content.replace(out_img_path.name, out_bil_path.name)
                    out_hdr_path.write_text(hdr_content)

            if out_hdr_path.exists() and out_bil_path.exists():
                print("Guardado exitoso del cubo hiperespectral:")
                print(f"  - Archivo de cabecera: {out_hdr_path}")
                print(f"  - Archivo de datos:    {out_bil_path}")
            else:
                print(f"Error: No se pudieron generar ambos archivos (.hdr y .bil) en: {out_hdr_path.parent}")
                # Considerar si continuar o salir si el guardado falla
        except Exception as e:
            print(f"Error guardando el cubo recortado: {e}")
            # Considerar si continuar o salir

        # 8. Guardar un JPEG de la ROI
        try:
            roi_uint8_recortada = banda_uint8[y : y + h, x : x + w]
            img_roi = Image.fromarray(roi_uint8_recortada)
            ruta_jpg = carpeta_salida / f"{nombre_recorte_actual}.jpg"
            img_roi.save(str(ruta_jpg), format="JPEG", quality=95)
            print(f"JPEG de la ROI guardado en: {ruta_jpg}")
        except Exception as e:
            print(f"¡Aviso! No se pudo guardar la imagen JPEG: {e}")

        # 9. Guardar la matriz de píxeles en un archivo CSV
        print("-> Guardando matriz de píxeles en CSV...")
        try:
            # Usar el band_idx determinado al inicio para el CSV
            roi_banda_data_para_csv = recorte_datos[:, :, band_idx] 
            nombre_csv = f"{nombre_recorte_actual}_banda{wl_str_original}.csv"
            ruta_csv = carpeta_salida / nombre_csv
            np.savetxt(str(ruta_csv), roi_banda_data_para_csv, delimiter=";", fmt="%d") # Asumiendo valores enteros para CSV
            print(f"Matriz de píxeles guardada en: {ruta_csv}")
        except Exception as e:
            print(f"¡Aviso! No se pudo guardar el archivo CSV: {e}")

        print(f"\n--- Recorte #{recorte_count} completado ---")
        
        respuesta = input("¿Desea realizar otro recorte? (S/n): ").strip().lower()
        if respuesta == 'n':
            break
    
    print("\n¡Proceso de recorte múltiple completado! Puedes cerrar esta ventana.")

if __name__ == "__main__":
    main()
