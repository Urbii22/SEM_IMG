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

    # 3. Seleccionar ROI con el ratón
    roi = cv2.selectROI("Selecciona ROI y pulsa ENTER", banda_uint8, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        sys.exit("No se definió ninguna ROI válida. Saliendo.")
        
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
