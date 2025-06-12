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

# Variables globales para la selección interactiva de ROI
_select_roi_params = {
    "current_roi_display_coords": None,  # (x, y, w, h) en coordenadas de la imagen mostrada
    "drawing": False,
    "start_point_display_coords": (-1, -1),
    "current_mouse_pos_display": (-1,-1), # Posición actual del ratón en la imagen mostrada
    "zoom_factor": 1.0,
    "original_image_ref": None,          # Imagen original (banda_uint8)
    "temp_display_image": None,          # Imagen para mostrar con dibujos temporales
    "window_name": "Selecciona ROI"
}

def mouse_callback_interactive(event, x, y, flags, param):
    """Callback del ratón para la selección interactiva de ROI."""
    p = _select_roi_params

    # Actualizar siempre la posición del ratón
    p["current_mouse_pos_display"] = (x,y)

    if event == cv2.EVENT_LBUTTONDOWN:
        p["drawing"] = True
        p["start_point_display_coords"] = (x, y)
        p["current_roi_display_coords"] = None  # Resetear ROI al empezar nuevo dibujo

    elif event == cv2.EVENT_MOUSEMOVE:
        if p["drawing"]:
            # La lógica de dibujo y texto se maneja en el bucle principal
            pass

    elif event == cv2.EVENT_LBUTTONUP:
        p["drawing"] = False
        end_point_display_coords = (x, y)
        x1, y1 = p["start_point_display_coords"]
        x2, y2 = end_point_display_coords
        
        roi_x_display = min(x1, x2)
        roi_y_display = min(y1, y2)
        roi_w_display = abs(x1 - x2)
        roi_h_display = abs(y1 - y2)

        if roi_w_display > 0 and roi_h_display > 0:
            p["current_roi_display_coords"] = (roi_x_display, roi_y_display, roi_w_display, roi_h_display)
        else:
            p["current_roi_display_coords"] = None
    
    elif event == cv2.EVENT_MOUSEWHEEL:
        prev_zoom_factor = p["zoom_factor"]
        if flags > 0:  # Rueda hacia arriba (Zoom In)
            p["zoom_factor"] *= 1.1
        else:  # Rueda hacia abajo (Zoom Out)
            p["zoom_factor"] /= 1.1
        
        p["zoom_factor"] = max(0.1, min(p["zoom_factor"], 20)) # Limitar zoom

        if p["current_roi_display_coords"]:
            rx, ry, rw, rh = p["current_roi_display_coords"]
            # Reescalar ROI manteniendo su centro relativo en la imagen zoomeada
            factor_cambio = p["zoom_factor"] / prev_zoom_factor
            
            center_x_display = rx + rw / 2
            center_y_display = ry + rh / 2
            
            new_rw_display = rw * factor_cambio
            new_rh_display = rh * factor_cambio
            
            new_rx_display = center_x_display - new_rw_display / 2
            new_ry_display = center_y_display - new_rh_display / 2
            
            p["current_roi_display_coords"] = tuple(map(int, [new_rx_display, new_ry_display, new_rw_display, new_rh_display]))


def select_roi_interactive_custom(image_to_select_on):
    """Permite al usuario seleccionar una ROI de forma interactiva con zoom y visualización de dimensiones."""
    p = _select_roi_params
    p["original_image_ref"] = image_to_select_on.copy()
    p["window_name"] = "ROI (Z/X/Rueda: Zoom, R: Reset, ENTER: OK, ESC: Cancelar)"
    p["zoom_factor"] = 1.0
    p["current_roi_display_coords"] = None
    p["drawing"] = False
    p["start_point_display_coords"] = (-1,-1)
    p["current_mouse_pos_display"] = (-1,-1)


    cv2.namedWindow(p["window_name"])
    cv2.setMouseCallback(p["window_name"], mouse_callback_interactive)

    h_orig, w_orig = p["original_image_ref"].shape[:2]

    while True:
        w_zoomed = int(w_orig * p["zoom_factor"])
        h_zoomed = int(h_orig * p["zoom_factor"])
        
        # Asegurar dimensiones mínimas para resize
        if w_zoomed < 1: w_zoomed = 1
        if h_zoomed < 1: h_zoomed = 1
        
        current_display_base = cv2.resize(p["original_image_ref"], (w_zoomed, h_zoomed), interpolation=cv2.INTER_LINEAR)
        # Si la imagen es monocromática, convertirla a BGR para dibujar en color
        if len(current_display_base.shape) == 2:
            p["temp_display_image"] = cv2.cvtColor(current_display_base, cv2.COLOR_GRAY2BGR)
        else:
            p["temp_display_image"] = current_display_base.copy()

        # Dibujar rectángulo mientras se arrastra
        if p["drawing"] and p["start_point_display_coords"] != (-1,-1) and p["current_mouse_pos_display"] != (-1,-1):
            cv2.rectangle(p["temp_display_image"], p["start_point_display_coords"], p["current_mouse_pos_display"], (0, 255, 0), 1)
            w_disp_rt = abs(p["current_mouse_pos_display"][0] - p["start_point_display_coords"][0])
            h_disp_rt = abs(p["current_mouse_pos_display"][1] - p["start_point_display_coords"][1])
            w_orig_rt = int(w_disp_rt / p["zoom_factor"])
            h_orig_rt = int(h_disp_rt / p["zoom_factor"])
            text_rt = f"Drawing: W:{w_orig_rt} H:{h_orig_rt}"
            cv2.putText(p["temp_display_image"], text_rt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Dibujar ROI seleccionada
        elif p["current_roi_display_coords"]:
            rx_d, ry_d, rw_d, rh_d = p["current_roi_display_coords"]
            cv2.rectangle(p["temp_display_image"], (rx_d, ry_d), (rx_d + rw_d, ry_d + rh_d), (0, 0, 255), 2)
            w_final_orig = int(rw_d / p["zoom_factor"])
            h_final_orig = int(rh_d / p["zoom_factor"])
            text_final = f"Selected: W:{w_final_orig} H:{h_final_orig}"
            cv2.putText(p["temp_display_image"], text_final, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

        info_text = f"Zoom:{p['zoom_factor']:.2f}x (Z/X/Rueda). R:Reset. ENTER:OK. ESC:Cancel."
        cv2.putText(p["temp_display_image"], info_text, (10, p["temp_display_image"].shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(p["window_name"], p["temp_display_image"])
        k = cv2.waitKey(20) & 0xFF

        if k == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        elif k == 13:  # Enter
            if p["current_roi_display_coords"]:
                rx_d, ry_d, rw_d, rh_d = p["current_roi_display_coords"]
                orig_x = int(rx_d / p["zoom_factor"])
                orig_y = int(ry_d / p["zoom_factor"])
                orig_w = int(rw_d / p["zoom_factor"])
                orig_h = int(rh_d / p["zoom_factor"])

                orig_x = max(0, orig_x)
                orig_y = max(0, orig_y)
                if orig_x + orig_w > w_orig: orig_w = w_orig - orig_x
                if orig_y + orig_h > h_orig: orig_h = h_orig - orig_y
                
                if orig_w > 0 and orig_h > 0:
                    cv2.destroyAllWindows()
                    return (orig_x, orig_y, orig_w, orig_h)
                else:
                    print("Advertencia: ROI inválida después de reescalar. Intenta de nuevo.")
                    p["current_roi_display_coords"] = None 
            else:
                print("Ninguna ROI seleccionada.")
        
        elif k == ord('z'): 
            prev_zoom_factor = p["zoom_factor"]
            p["zoom_factor"] *= 1.25
            p["zoom_factor"] = min(p["zoom_factor"], 20) # Limitar zoom
            if p["current_roi_display_coords"]:
                rx,ry,rw,rh = p["current_roi_display_coords"]
                factor_cambio = p["zoom_factor"] / prev_zoom_factor
                center_x_display, center_y_display = rx + rw / 2, ry + rh / 2
                new_rw_display, new_rh_display = rw * factor_cambio, rh * factor_cambio
                new_rx_display, new_ry_display = center_x_display - new_rw_display / 2, center_y_display - new_rh_display / 2
                p["current_roi_display_coords"] = tuple(map(int, [new_rx_display, new_ry_display, new_rw_display, new_rh_display]))


        elif k == ord('x'):
            prev_zoom_factor = p["zoom_factor"]
            p["zoom_factor"] /= 1.25
            p["zoom_factor"] = max(0.1, p["zoom_factor"]) # Limitar zoom
            if p["current_roi_display_coords"]:
                rx,ry,rw,rh = p["current_roi_display_coords"]
                factor_cambio = p["zoom_factor"] / prev_zoom_factor
                center_x_display, center_y_display = rx + rw / 2, ry + rh / 2
                new_rw_display, new_rh_display = rw * factor_cambio, rh * factor_cambio
                new_rx_display, new_ry_display = center_x_display - new_rw_display / 2, center_y_display - new_rh_display / 2
                p["current_roi_display_coords"] = tuple(map(int, [new_rx_display, new_ry_display, new_rw_display, new_rh_display]))

        elif k == ord('r'): 
            p["zoom_factor"] = 1.0
            p["current_roi_display_coords"] = None
            p["drawing"] = False
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

    # 2. Cargar cubo HSI y extraer banda ~728 nm para visualización
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
                if data.max() < 10.0:
                    print(f"Factor de escala encontrado: {scale_factor}. Revirtiendo a valores DN originales...")
                    data *= scale_factor
            except (ValueError, TypeError):
                print("Aviso: no se pudo interpretar el factor de escala.")
        # --- FIN DE LA CORRECCIÓN DE ESCALA ---

    except Exception as e:
        sys.exit(f"Error cargando el cubo hiperespectral: {e}")

    # Intentamos extraer banda ~728 nm
    TARGET_WL = 728.24
    band_idx = 0
    wl_str = ""
    meta_wl = cube_hsi.metadata.get("wavelength", [])
    if meta_wl:
        try:
            wl = np.array(meta_wl, dtype=float)
            band_idx = int(np.argmin(np.abs(wl - TARGET_WL)))
            wl_str = f"_{int(round(wl[band_idx]))}nm"
            print(f"Banda objetivo: índice {band_idx} (≈ {wl[band_idx]:.1f} nm)")
        except Exception:
            print("No se pudo interpretar metadata de longitudes de onda. Se usará banda 0.")
    else:
        print("No hay metadata de longitudes de onda. Se usará la primera banda (0).")

    banda_para_visualizar = data[:, :, band_idx]
    banda_uint8 = cv2.normalize(banda_para_visualizar, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Seleccionar ROI con el ratón de forma interactiva
    # roi = cv2.selectROI("Selecciona ROI y pulsa ENTER", banda_uint8, fromCenter=False, showCrosshair=True)
    # cv2.destroyAllWindows()
    roi = select_roi_interactive_custom(banda_uint8)

    if not roi or roi[2] == 0 or roi[3] == 0: # roi puede ser None si se cancela
        sys.exit("No se definió ninguna ROI válida o se canceló la operación. Saliendo.")
        
    x, y, w, h = roi
    print(f"ROI seleccionada: x={x}, y={y}, ancho={w}, alto={h}")

    # 4. Recortar el cubo
    recorte = data[y : y + h, x : x + w, :].copy()
    rec_rows, rec_cols, rec_bands = recorte.shape
    print(f"Cubo recortado con forma: {recorte.shape} (filas, cols, bandas)")

    # 5. Seleccionar carpeta de salida
    carpeta_salida = seleccionar_carpeta("Selecciona carpeta donde guardar el recorte")
    base = hdr_path.stem
    nombre_recorte = f"{base}_recorte"
    
    # 6. Guardar el cubo recortado en formato ENVI (.bil y .hdr)
    print("-> Guardando cubo recortado en formato ENVI...")
    out_hdr_path = carpeta_salida / f"{nombre_recorte}.hdr"
    out_bil_path = carpeta_salida / f"{nombre_recorte}.bil"
    
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

        envi.save_image(str(out_hdr_path), recorte.astype(save_dtype), force=True, metadata=metanew, interleave='bil')
        
        out_img_path = out_hdr_path.with_suffix('.img')
        if out_img_path.exists() and not out_bil_path.exists():
            out_img_path.rename(out_bil_path)
            hdr_content = out_hdr_path.read_text().replace(out_img_path.name, out_bil_path.name)
            out_hdr_path.write_text(hdr_content)

        if out_hdr_path.exists() and out_bil_path.exists():
            print("Guardado exitoso del cubo hiperespectral:")
            print(f"  - Archivo de cabecera: {out_hdr_path}")
            print(f"  - Archivo de datos:    {out_bil_path}")
        else:
            print(f"Guardado parcial o con errores en: {out_hdr_path.parent}")
            
    except Exception as e:
        sys.exit(f"Error guardando el cubo recortado: {e}")

    # 7. Guardar un JPEG de la ROI
    try:
        roi_uint8 = banda_uint8[y : y + h, x : x + w]
        img_roi = Image.fromarray(roi_uint8)
        ruta_jpg = carpeta_salida / f"{nombre_recorte}.jpg"
        img_roi.save(str(ruta_jpg), format="JPEG", quality=95)
        print(f"JPEG de la ROI guardado en: {ruta_jpg}")
    except Exception as e:
        print(f"¡Aviso! No se pudo guardar la imagen JPEG: {e}")

    # 8. Guardar la matriz de píxeles en un archivo CSV
    print("-> Guardando matriz de píxeles en CSV...")
    try:
        roi_banda_data = recorte[:, :, band_idx]
        nombre_csv = f"{nombre_recorte}_banda{wl_str}.csv"
        ruta_csv = carpeta_salida / nombre_csv
        # --- CAMBIO IMPORTANTE ---
        # Usamos punto y coma (;) como delimitador para compatibilidad con Excel en regiones europeas.
        np.savetxt(str(ruta_csv), roi_banda_data, delimiter=";", fmt="%d")
        print(f"Matriz de píxeles guardada en: {ruta_csv}")
    except Exception as e:
        print(f"¡Aviso! No se pudo guardar el archivo CSV: {e}")

    print("\n¡Proceso completado! Puedes cerrar esta ventana.")

if __name__ == "__main__":
    main()
