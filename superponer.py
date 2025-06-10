import tkinter as tk
from tkinter import filedialog, Scale
from PIL import Image, ImageTk, ImageEnhance

class ImageSuperposerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Superposición Manual de Imágenes")

        self.base_image_path = None
        self.overlay_image_path = None

        self.base_image_pil_original = None # Original sin escalar
        self.base_image_pil_scaled = None   # Original escalada según self.base_scale
        self.base_image_tk = None
        self.overlay_image_pil_original = None
        self.overlay_image_pil_transformed = None
        self.overlay_image_tk = None

        # Parámetros de transformación
        self.base_scale = 1.0
        self.overlay_x = 50.0
        self.overlay_y = 50.0
        self.overlay_scale = 1.0
        self.overlay_angle = 0.0
        self.overlay_alpha = 0.7

        self._drag_data = {"x_offset": 0, "y_offset": 0, "item": None}

        self.setup_ui()
        self.load_initial_images()

    def setup_ui(self):
        # Canvas para las imágenes
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="lightgrey")
        self.canvas.pack(fill="both", expand=True)

        # Frame para controles
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(fill="x", pady=5)

        tk.Button(controls_frame, text="Cargar Imagen Base", command=self.load_base_image).pack(side="left", padx=5)
        tk.Button(controls_frame, text="Cargar Imagen a Superponer", command=self.load_overlay_image).pack(side="left", padx=5)

        tk.Label(controls_frame, text="Transparencia Overlay:").pack(side="left", padx=(10,0))
        self.alpha_slider = Scale(controls_frame, from_=0, to=100, orient="horizontal", command=self.update_alpha_slider)
        self.alpha_slider.set(int(self.overlay_alpha * 100))
        self.alpha_slider.pack(side="left", padx=5)

        self.canvas.tag_bind("overlay_image_tag", "<ButtonPress-1>", self.on_mouse_press)
        self.canvas.tag_bind("overlay_image_tag", "<B1-Motion>", self.on_mouse_drag)

        # Bindings de teclado
        self.root.bind("<plus>", self.scale_overlay_up) # Renombrado para claridad
        self.root.bind("<equal>", self.scale_overlay_up) # Tecla = a menudo junto a +
        self.root.bind("<minus>", self.scale_overlay_down) # Renombrado
        self.root.bind("<Control-plus>", self.scale_overlay_up_fine) # Renombrado
        self.root.bind("<Control-equal>", self.scale_overlay_up_fine)
        self.root.bind("<Control-minus>", self.scale_overlay_down_fine) # Renombrado

        self.root.bind("<Alt-plus>", self.scale_base_up)
        self.root.bind("<Alt-equal>", self.scale_base_up)
        self.root.bind("<Alt-minus>", self.scale_base_down)
        # Podrías añadir Control-Alt para ajuste fino de la base si lo deseas

        self.root.bind("<Left>", self.rotate_overlay_left) # Renombrado
        self.root.bind("<Right>", self.rotate_overlay_right) # Renombrado
        self.root.bind("<Control-Left>", self.rotate_overlay_left_fine) # Renombrado
        self.root.bind("<Control-Right>", self.rotate_overlay_right_fine) # Renombrado
        
        instructions_text = ("Controles:\n"
                             "Arrastrar Overlay: Click y arrastrar imagen superpuesta.\n"
                             "Zoom Overlay: Teclas +/- (Ctrl para ajuste fino).\n"
                             "Zoom Base: Teclas Alt + Alt -.\n"
                             "Rotar Overlay: Flechas Izquierda/Derecha (Ctrl para ajuste fino).")
        instructions_label = tk.Label(self.root, text=instructions_text, justify="left")
        instructions_label.pack(fill="x", pady=5, padx=5)

    def get_initial_base_scale(self, image_pil):
        self.root.update_idletasks() # Asegurar que el canvas tenga dimensiones
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1 or not image_pil:
            return 1.0 

        original_width, original_height = image_pil.size
        if original_width == 0 or original_height == 0:
            return 1.0

        ratio_w = canvas_width / original_width
        ratio_h = canvas_height / original_height
        scale_factor = min(ratio_w, ratio_h) 
        return scale_factor

    def apply_base_scale_and_update_tk(self):
        if not self.base_image_pil_original:
            self.base_image_pil_scaled = None
            self.base_image_tk = None
            return

        width = int(self.base_image_pil_original.width * self.base_scale)
        height = int(self.base_image_pil_original.height * self.base_scale)

        if width <= 0 or height <= 0: # Evitar tamaño cero o negativo
            self.base_image_pil_scaled = None
            self.base_image_tk = None
            return
        
        try:
            self.base_image_pil_scaled = self.base_image_pil_original.resize((width, height), Image.Resampling.LANCZOS)
        except AttributeError: # Para versiones antiguas de Pillow
            self.base_image_pil_scaled = self.base_image_pil_original.resize((width, height), Image.LANCZOS)
        self.base_image_tk = ImageTk.PhotoImage(self.base_image_pil_scaled)


    def load_base_image(self):
        path = filedialog.askopenfilename(title="Seleccionar Imagen Base", filetypes=[("Archivos de imagen", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if path:
            self.base_image_path = path
            self.base_image_pil_original = Image.open(self.base_image_path)
            self.base_scale = self.get_initial_base_scale(self.base_image_pil_original) # Escala inicial para ajustar
            self.apply_base_scale_and_update_tk()
            self.redraw_canvas()

    def load_overlay_image(self):
        path = filedialog.askopenfilename(title="Seleccionar Imagen a Superponer", filetypes=[("Archivos de imagen", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if path:
            self.overlay_image_path = path
            self.overlay_image_pil_original = Image.open(self.overlay_image_path).convert("RGBA") # RGBA para transparencia
            
            self.root.update_idletasks() # Asegurar que el canvas tenga dimensiones
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            # Lógica mejorada para el escalado inicial y centrado
            if self.overlay_image_pil_original and canvas_w > 1 and canvas_h > 1:
                img_w, img_h = self.overlay_image_pil_original.size
                if img_w > 0 and img_h > 0:
                    # Escalar para que la dimensión más grande sea 1/3 de la dimensión más pequeña del canvas
                    target_dim = min(canvas_w, canvas_h) / 3.0
                    
                    if img_w > img_h: # Imagen más ancha
                        self.overlay_scale = target_dim / img_w
                    else: # Imagen más alta o cuadrada
                        self.overlay_scale = target_dim / img_h
                    
                    # Asegurar que la escala no sea excesivamente pequeña o grande
                    self.overlay_scale = max(0.05, min(self.overlay_scale, 2.0))
                    
                    # Centrar la imagen superpuesta inicialmente
                    # Necesitamos el tamaño después del escalado para centrarla correctamente
                    # Esto se manejará en apply_transformations_and_redraw, aquí solo preparamos la escala
                    # y una posición inicial que se ajustará.
                    # La posición real se calcula basada en el tamaño transformado.
                    # Por ahora, estimamos el centro. El redibujo lo colocará bien.
                    scaled_w = img_w * self.overlay_scale
                    scaled_h = img_h * self.overlay_scale
                    self.overlay_x = (canvas_w - scaled_w) / 2
                    self.overlay_y = (canvas_h - scaled_h) / 2

                else:
                    self.overlay_scale = 0.5 # Default si el tamaño de la imagen es cero
                    self.overlay_x = canvas_w / 4 if canvas_w > 1 else 50.0
                    self.overlay_y = canvas_h / 4 if canvas_h > 1 else 50.0
            else:
                self.overlay_scale = 0.5 
                self.overlay_x = canvas_w / 4 if canvas_w > 1 else 50.0
                self.overlay_y = canvas_h / 4 if canvas_h > 1 else 50.0
            
            self.overlay_angle = 0.0
            self.alpha_slider.set(int(self.overlay_alpha * 100))
            self.apply_transformations_and_redraw()

    def resize_image_to_fit_canvas(self, image_pil):
        self.root.update_idletasks() # Asegurar que el canvas tenga dimensiones
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1: # Canvas no listo
            return image_pil # Devolver original si no se puede calcular tamaño

        original_width, original_height = image_pil.size
        if original_width == 0 or original_height == 0: return image_pil

        ratio = min(canvas_width / original_width, canvas_height / original_height)
        
        # Solo redimensionar si es necesario para encajar (no agrandar)
        if ratio < 1.0:
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            new_width = max(1, new_width) # Evitar tamaño cero
            new_height = max(1, new_height)
            try:
                return image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except AttributeError: # Para versiones antiguas de Pillow
                return image_pil.resize((new_width, new_height), Image.LANCZOS)
        return image_pil


    def apply_transformations_and_redraw(self):
        if not self.overlay_image_pil_original:
            # Si no hay imagen superpuesta, solo necesitamos asegurar que la base esté bien dibujada.
            # self.apply_base_scale_and_update_tk() # Asegurar que la base esté actualizada (si es necesario)
            self.redraw_canvas() 
            return

        # 1. Escalar
        width = int(self.overlay_image_pil_original.width * self.overlay_scale)
        height = int(self.overlay_image_pil_original.height * self.overlay_scale)
        
        if width <= 0 or height <= 0: 
            self.overlay_image_pil_transformed = None
            self.overlay_image_tk = None
            self.redraw_canvas()
            return

        try:
            img_scaled = self.overlay_image_pil_original.resize((width, height), Image.Resampling.LANCZOS)
        except AttributeError:
            img_scaled = self.overlay_image_pil_original.resize((width, height), Image.LANCZOS)


        # 2. Rotar
        try:
            img_rotated = img_scaled.rotate(self.overlay_angle, expand=True, resample=Image.Resampling.BICUBIC)
        except AttributeError:
            img_rotated = img_scaled.rotate(self.overlay_angle, expand=True, resample=Image.BICUBIC)


        # 3. Transparencia
        if img_rotated.mode != 'RGBA':
            img_rotated = img_rotated.convert('RGBA')
            
        alpha_channel = img_rotated.split()[-1]
        alpha_adjusted = ImageEnhance.Brightness(alpha_channel).enhance(self.overlay_alpha)
        img_rotated.putalpha(alpha_adjusted)

        self.overlay_image_pil_transformed = img_rotated
        self.overlay_image_tk = ImageTk.PhotoImage(self.overlay_image_pil_transformed)
        
        # Si es la primera vez que se dibuja después de cargar (o si queremos recentrar)
        # La lógica de centrado inicial en load_overlay_image ya establece overlay_x, overlay_y
        # de forma aproximada. El arrastre permitirá al usuario moverla.
        
        self.redraw_canvas()

    def redraw_canvas(self):
        self.canvas.delete("all") 

        if self.base_image_tk and self.base_image_pil_scaled:
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            img_w = self.base_image_pil_scaled.width
            img_h = self.base_image_pil_scaled.height
            
            base_x = (canvas_w - img_w) / 2
            base_y = (canvas_h - img_h) / 2
            self.canvas.create_image(base_x, base_y, anchor="nw", image=self.base_image_tk, tags="base_image_tag")

        if self.overlay_image_tk and self.overlay_image_pil_transformed:
            self.canvas.create_image(self.overlay_x, self.overlay_y, anchor="nw", image=self.overlay_image_tk, tags="overlay_image_tag")

    def update_alpha_slider(self, val_str):
        self.overlay_alpha = int(val_str) / 100.0
        self.apply_transformations_and_redraw()

    def on_mouse_press(self, event):
        # Comprobar si el click fue sobre la imagen superpuesta
        items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        overlay_item_ids = self.canvas.find_withtag("overlay_image_tag") # Devuelve una tupla
        
        if overlay_item_ids and overlay_item_ids[0] in items:
            self._drag_data["item"] = overlay_item_ids[0]
            # Almacenar el offset del clic respecto a la esquina superior izquierda de la imagen (overlay_x, overlay_y)
            self._drag_data["x_offset"] = event.x - self.overlay_x
            self._drag_data["y_offset"] = event.y - self.overlay_y
        else:
            self._drag_data["item"] = None


    def on_mouse_drag(self, event):
        if self._drag_data["item"]:
            # Nueva posición es la posición del ratón menos el offset inicial
            self.overlay_x = event.x - self._drag_data["x_offset"]
            self.overlay_y = event.y - self._drag_data["y_offset"]
            
            self.redraw_canvas() # Redibujar con la nueva posición

    def scale_overlay_up(self, event=None): # Renombrado
        self.overlay_scale *= 1.1
        self.overlay_scale = min(self.overlay_scale, 10.0) # Limite superior
        self.apply_transformations_and_redraw()

    def scale_overlay_down(self, event=None): # Renombrado
        self.overlay_scale /= 1.1
        self.overlay_scale = max(self.overlay_scale, 0.01) # Limite inferior
        self.apply_transformations_and_redraw()

    def scale_overlay_up_fine(self, event=None): # Renombrado
        self.overlay_scale *= 1.02
        self.overlay_scale = min(self.overlay_scale, 10.0)
        self.apply_transformations_and_redraw()

    def scale_overlay_down_fine(self, event=None): # Renombrado
        self.overlay_scale /= 1.02
        self.overlay_scale = max(self.overlay_scale, 0.01)
        self.apply_transformations_and_redraw()

    def scale_base_up(self, event=None):
        if not self.base_image_pil_original: return
        self.base_scale *= 1.1
        self.base_scale = min(self.base_scale, 10.0) # Limite superior para escala base
        self.apply_base_scale_and_update_tk()
        self.redraw_canvas()

    def scale_base_down(self, event=None):
        if not self.base_image_pil_original: return
        self.base_scale /= 1.1
        self.base_scale = max(self.base_scale, 0.05) # Limite inferior para escala base
        self.apply_base_scale_and_update_tk()
        self.redraw_canvas()

    def rotate_overlay_left(self, event=None): # Renombrado
        self.overlay_angle += 5 
        self.apply_transformations_and_redraw()

    def rotate_overlay_right(self, event=None): # Renombrado
        self.overlay_angle -= 5 
        self.apply_transformations_and_redraw()

    def rotate_overlay_left_fine(self, event=None): # Renombrado
        self.overlay_angle += 1
        self.apply_transformations_and_redraw()

    def rotate_overlay_right_fine(self, event=None): # Renombrado
        self.overlay_angle -= 1
        self.apply_transformations_and_redraw()

    def load_initial_images(self):
        self.root.bind("<Configure>", self.on_window_resize)
        self.root.update_idletasks() 
        self.redraw_canvas() # Dibujo inicial

    def on_window_resize(self, event=None):
        if event and event.widget == self.root:
            if self.base_image_path: # Si hay una imagen base cargada
                # Recargar la original para evitar degradación por múltiples redimensionados
                # self.base_image_pil_original = Image.open(self.base_image_path) 
                # No es necesario recargarla aquí si ya la tenemos en self.base_image_pil_original
                # y no la modificamos directamente.
                # La escala inicial se calcula una vez. El zoom del usuario persiste.
                if self.base_image_pil_original:
                    self.apply_base_scale_and_update_tk() # Re-aplica la escala actual
            
            # La imagen superpuesta se redibuja con sus transformaciones actuales
            # y la imagen base también, ambas centradas/posicionadas como corresponda.
            self.apply_transformations_and_redraw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSuperposerApp(root)
    # Forzar una actualización inicial para que las dimensiones del canvas estén disponibles
    root.update_idletasks() 
    app.redraw_canvas() # Un redibujo inicial después de que todo esté configurado
    root.mainloop()
