import customtkinter as ctk
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from widgets.interactive_image import Interactive_image
import torch
import torch.utils.data
from scipy.ndimage import label
import cv2


#TESTING
import random

sys.path.append(os.path.abspath('.'))

from settings import Frame_settings

class Detection_gui(ctk.CTk):
    def __init__(self, project_path=os.path.abspath("."), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.project_path = project_path
        print(self.project_path)

        # Window theme
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('blue')

        # Inizialicing window
        self.geometry('1300x800+50+50')
        self.title('Labeling app')

        # Window icon
        icon_path = os.path.join(self.project_path, "assets", "icons", "Group_icon.ico")
        self.iconbitmap(icon_path)
        
        # Data managment
        self.list_index = []
        self.list_name = []
        self.list_x = []
        self.list_y = []
        self.list_Pcoordinates = []

        self.metadata_dic = {'Index' : [],
                             "Sample" : [],
                             'Coordinate_x' : [],
                             'Coordinate_y' : [],
                             "Particle_Coordinates" : []}

        # Frames set up
        self.frame_acquisition = ctk.CTkScrollableFrame(self, orientation='horizontal', height=250)
        self.frame_images = ctk.CTkFrame(self)
        self.frame_prediction = ctk.CTkFrame(self)
        self.frame_settings = Frame_settings(self, self.project_path, width=340)
        self.frame_settings.pack_propagate(False)

        self.frame_settings.pack(side = 'right', fill = 'both', padx = 5, pady = 5)
        self.frame_acquisition.pack(side="top", fill = 'x', padx = 10, pady = 10)
        self.frame_prediction.pack(expand = True,side = "left", fill = 'both', padx = 10, pady = 10)
        self.frame_images.pack(expand = True,side = "left", fill = 'both', padx = 10, pady = 10)

        # Button Play
        self.frame_settings.button_play.configure(command = self.toggle_acquisition)

        # Button Select Directory
        self.frame_settings.directory_selector.select_button.configure(command=lambda: (self.frame_settings.directory_selector.change_directory(), self.load_sample()))

        self.frame_settings.change_image.previous_button.configure(command=self.previous_image)
        self.frame_settings.change_image.next_button.configure(command=self.next_image)

        self.is_running = False
        self.zeros = np.zeros((256,256))
        
        # Images set up
        self.image_microscope = Interactive_image(self.frame_images, self.zeros, title = 'Microscope Image', font=('American typewriter', 24), width= 720, height= 720)
        self.image_microscope.pack()

        self.image_prediction = Interactive_image(self.frame_prediction, self.zeros, title = 'Model prediction', font=('American typewriter', 24), width= 720, height= 720)
        self.image_prediction.pack()

        # Simplifying commonly used variables from other scripts
        self.np_number = self.frame_settings.parameters.entrys_list[0]  # Number of nanoparticles to detect
        self.directory = self.frame_settings.directory_selector
        self.np_index = self.frame_settings.nanoparticle_counter.counter_var
        self.number_added = 0
        self.load_sample()


        # Load the model, this should be done only once at the beginning of the program, save this in case we want to use it for changing the model in a future
        self.model_ml = torch.load('ModelNP_newContrast_900.pt')
        

# This is the function that we call when the button Start is pressed, we should start here the routine of detection
    def toggle_acquisition(self):
        if not self.is_running:
            self.start_acquisition()
            self.process_folder_and_update_gui("C:\\Users\\josep\\Desktop\\extra_training\\images", self.model_ml, num_images=int(self.np_number.get()), th=0.8)


            # Testing, we should start here the routine of detection
            #self.number_added = self.np_index.get() + 1
            #self.np_index.set(self.number_added)
            #self.create_particle_image(self.number_added)

        else:
            self.stop_acquisition()
            
    def start_acquisition(self):
        self.is_running = True
        self.frame_settings.button_play.configure(text='Searching...', fg_color='red', command=self.stop_acquisition)
        self.hide_navigation()

    def stop_acquisition(self):
        self.is_running = False
        self.frame_settings.button_play.configure(text='Start acquisition', fg_color='green', command=self.toggle_acquisition)
        self.show_navigation()



# For testing, it creates the image and change it on the elements in the UI
# Change it to obtain the image from microscope instead of creating one
    def create_particle_image(self, number):
        random_array_int = np.random.randint(0, 256, (256, 256)).astype(np.uint8)
        im = Image.fromarray(random_array_int)

        self.save_image(im,self.directory.directory,f"test_images_{number}", save = True)
        self.test_create_np()
        self.image_microscope.change_image(np.array(im))
        self.image_prediction.change_image(np.array(im))

        self.list_index.append(number)
        self.list_x.append(number)
        self.list_y.append(number+1)
        self.save_data()
        

# Save the image in the directory selected on the UI
    def save_image(self, image, directory, im_name, save):
        os.makedirs(directory, exist_ok=True)
        name = f"{im_name}.png"
        image_path = os.path.join(directory, name)

        if save == True:
            self.list_name.append(name)

        image.save(image_path)
    
    def test_create_np(self):

        self.restart_frame_acquisition()
        np_directory = f"{self.directory.directory}\\{self.list_name[self.number_added-1]}_NP_Detected"
        os.makedirs(np_directory, exist_ok=True)
        
        new_list = []
        for i in range(self.number_added):
            new_list.append((i, i+3))
            random_array_int = np.random.randint(0, 256, (256, 256)).astype(np.uint8)

            new_nano = Interactive_image(self.frame_acquisition, random_array_int, title = f'NP_{i+1}', subtitle=f'Coords: ({i}, {i+3})', font=('American typewriter', 24), subfont=('American typewriter', 14), width=150, height=150)
            new_nano.pack(side = 'left')
            im = Image.fromarray(random_array_int)
            self.save_image(im, np_directory,f"Image_np{i}",  save = False)

        self.list_Pcoordinates.append(new_list)


    def load_sample(self):
        self.restart_frame_acquisition()

        if os.path.exists(self.directory.directory + '/metadata.csv'):
            self.load_data()
            self.number_added = int(self.np_index.get())
        else:
            self.restart_sample()

    def restart_sample(self):
        # Restore images to default
        self.image_microscope.change_image(self.zeros)
        self.image_prediction.change_image(self.zeros)

        # Empty data
        self.number_added = 0
        self.np_index.set(0)
        self.list_index = []
        self.list_name = []
        self.list_x = []
        self.list_y = []
        self.list_Pcoordinates = []

    # Save the metadata
    def save_data(self):
        self.metadata_dic = {'Index' : self.list_index,
                        "Sample" : self.list_name,
                        'Coordinate_x' : self.list_x,
                        'Coordinate_y' : self.list_y,
                        "Particles_coordinates": self.list_Pcoordinates}
        
        self.metadata = pd.DataFrame(self.metadata_dic)
        self.metadata.to_csv(self.directory.directory + '\metadata.csv', sep='\t', index = False)
        self.metadata.to_excel(self.directory.directory + '\metadata.xlsx', index = False)

    # Load the metadata
    def load_data(self):
        filepath = self.directory.directory + '/metadata.csv'
        self.metadata = pd.read_csv(filepath, sep='\t')
        self.metadata.fillna('', inplace=True)

        self.list_index = list(self.metadata.Index)
        self.list_name = list(self.metadata.Sample)
        self.list_x = list(self.metadata.Coordinate_x)
        self.list_y = list(self.metadata.Coordinate_y)
        self.list_Pcoordinates = list(self.metadata.Particles_coordinates)

        self.np_index.set(len(self.list_index))

        coordinates_list = eval(self.list_Pcoordinates[self.number_added-1])
        for i in range(len(coordinates_list)):
            im = Image.open(self.directory.directory + "\\" + self.list_name[self.number_added-1] + "_NP_Detected" + "\\" + f"Image_np{i}.png")
            im = np.array(im)
            new_nano = Interactive_image(self.frame_acquisition, im, title = f'NP_{i+1}', subtitle=f'Coords: {coordinates_list[i]}', font=('American typewriter', 24), subfont=('American typewriter', 14), width=150, height=150)
            new_nano.pack(side = 'left')

        im = Image.open(self.directory.directory + "\\" +self.list_name[self.number_added-1])
        im = np.array(im)
        self.image_microscope.change_image(im)
        self.image_prediction.change_image(im)
            
    def previous_image(self):
        if self.number_added > 1:
            self.number_added -= 1
            self.restart_frame_acquisition()
            self.load_data()
        else:
            print("No previous image available")

    def next_image(self):
        if self.number_added < len(self.list_index):
            self.restart_frame_acquisition()
            self.number_added += 1
            self.load_data()
        else:
            print("No next image available")

    def restart_frame_acquisition(self):
        for widget in self.frame_acquisition.winfo_children():
            widget.destroy()

    def hide_navigation(self):
        self.frame_settings.frame_image.pack_forget()

    def show_navigation(self):
        self.frame_settings.frame_image.pack(padx = 5, pady = 10, fill = 'x')


    def crop(self, image, bbox):
        r, c, s = bbox
        crop = image[r:r+s, c:c+s]
        return crop


    def process_folder_and_update_gui(self, folder_path, model, num_images=5, th=0.8):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Opcional: asegura orden
        processed = 0
        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            self.image_microscope.change_image(image)

            # Detecta partículas
            img_with_boxes, final_bboxes = self.detect_particles_in_image(image, model, th=th)
            if img_with_boxes is not None and len(final_bboxes) > 0:
                # Mostrar en la interfaz
                self.image_prediction.change_image(img_with_boxes)
                # Crear widgets pequeños para cada nanopartícula
                self.restart_frame_acquisition()
                for i, bbox in enumerate(final_bboxes):
                    start_row, start_col, side = bbox
                    
                    particle_crop = self.crop(img_with_boxes, bbox)
                    particle_crop = np.array(particle_crop).astype(np.uint8)

                    new_nano = Interactive_image(self.frame_acquisition, particle_crop, title = f'NP_{i+1}', subtitle=f'Coords:', font=('American typewriter', 24), subfont=('American typewriter', 14), width=150, height=150)
                    new_nano.pack(side = 'left')
                    
                # Puedes guardar metadatos aquí usando tus listas
                self.list_index.append(processed)
                self.list_name.append(fname)
                self.list_x.append(start_col)
                self.list_y.append(start_row)
                self.list_Pcoordinates.append(final_bboxes)
                self.save_data()
                processed += 1
                self.np_index.set(processed)
                if processed >= num_images:
                    break


    def isolate_particles(self, pred_mask, threshold=0.5):
        # Convierte la máscara de probabilidades en máscara binaria (0 o 1)
        binary_mask = (pred_mask > threshold).astype(np.uint8)
        # Etiqueta los grupos de píxeles conectados (componentes)
        labeled_mask, num_particles = label(binary_mask)
        return labeled_mask, num_particles

    def get_indices_box(self, indices):
        """Calcula los límites (r_min, r_max, c_min, c_max) a partir de los índices de probabilidad."""
        rows, cols = indices
        if len(rows) == 0 or len(cols) == 0:
            return None
        return (np.min(rows), np.max(rows), np.min(cols), np.max(cols))

    def box_contains_any_center(self,index_box, zone_centers):
        """
        Dada una caja de índices (r_min, r_max, c_min, cmax) y una lista de centros (x, y),
        retorna True si al menos uno de esos centros está contenido en la caja.
        Se asume que x corresponde a la fila y y a la columna.
        """
        rmin, rmax, cmin, cmax = index_box
        for center in zone_centers:
            x, y = center  # Solo se usan las dos primeras coordenadas
            if rmin <= x <= rmax and cmin <= y <= cmax:
                return True
        return False

    def rectangle_distance_indices(self, box1, box2):
        """
        Calcula la distancia entre dos cajas definidas por índices.
        Cada caja es (r_min, r_max, c_min, c_max). La distancia se define a partir del gap horizontal y vertical.
        Si se solapan en alguna dirección, el gap es 0 en esa dirección.
        """
        rmin1, rmax1, cmin1, cmax1 = box1
        rmin2, rmax2, cmin2, cmax2 = box2
        gap_h = max(0, max(cmin1, cmin2) - min(cmax1, cmax2))
        gap_v = max(0, max(rmin1, rmin2) - min(rmax1, rmax2))
        return np.sqrt(gap_h**2 + gap_v**2)

    def merge_index_boxes(self, box1, box2):
        """Fusiona dos cajas (r_min, r_max, c_min, c_max) tomando la unión de sus límites."""
        rmin1, rmax1, cmin1, cmax1 = box1
        rmin2, rmax2, cmin2, cmax2 = box2
        return (min(rmin1, rmin2), max(rmax1, rmax2),
                min(cmin1, cmin2), max(cmax1, cmax2))

    def merge_nearby_index_boxes(self, boxes, merge_threshold):
        """
        Fusiona cajas de índices que estén separadas por una distancia (entre sus límites)
        menor que merge_threshold. Se iterará hasta que ya no se puedan fusionar.
        """
        merged_boxes = boxes.copy()
        merged = True
        while merged:
            merged = False
            new_boxes = []
            used = [False] * len(merged_boxes)
            for i in range(len(merged_boxes)):
                if used[i]:
                    continue
                current = merged_boxes[i]
                for j in range(i+1, len(merged_boxes)):
                    if used[j]:
                        continue
                    if self.rectangle_distance_indices(current, merged_boxes[j]) < merge_threshold:
                        current = self.merge_index_boxes(current, merged_boxes[j])
                        used[j] = True
                        merged = True
                new_boxes.append(current)
                used[i] = True
            merged_boxes = new_boxes
        return merged_boxes

    def index_box_to_square(self, index_box, image_shape):
        """
        Convierte una caja de índices (r_min, r_max, c_min, c_max) en un cuadrado.
        Se calcula el lado como el máximo entre la altura y el ancho y se centra.
        """
        rmin, rmax, cmin, cmax = index_box
        height = rmax - rmin
        width = cmax - cmin
        side = max(height, width)
        center_r = (rmin + rmax) // 2
        center_c = (cmin + cmax) // 2
        half_side = side // 2
        start_row = max(center_r - half_side, 0)
        start_col = max(center_c - half_side, 0)
        if start_row + side > image_shape[0]:
            start_row = image_shape[0] - side
        if start_col + side > image_shape[1]:
            start_col = image_shape[1] - side
        return (start_row, start_col, side)


    def detect_particles_in_image(self, image, model, th=0.8):

        # Modelo debe devolver [pred, zones], igual que tu script
        pred, zones = model.predict(image)
        pred_mask = np.squeeze(pred[0])
        labeled_mask, num_particles = self.isolate_particles(pred_mask, th)
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Extraer cajas
        index_boxes = []
        min_particle_size = 0
        for x in range(1, num_particles + 1):
            particle_indices = np.where(labeled_mask == x)
            ibox = self.get_indices_box(particle_indices)
            if ibox is not None:
                if (ibox[1] - ibox[0] >= min_particle_size) and (ibox[3] - ibox[2] >= min_particle_size):
                    index_boxes.append(ibox)
        # Centros de las zonas
        zone_centers = []
        for key, centers in zones.items():
            for center in centers:
                x, y, _ = center
                if pred_mask[int(x), int(y)] >= th:
                    zone_centers.append((x, y))

        filtered_index_boxes = [ibox for ibox in index_boxes if self.box_contains_any_center(ibox, zone_centers)]
        merged_index_boxes = self.merge_nearby_index_boxes(filtered_index_boxes, 20)
        final_bboxes = [self.index_box_to_square(ibox, image.shape) for ibox in merged_index_boxes]

        # Dibuja los cuadrados sobre la imagen original (para mostrar en la interfaz)
        for bbox in final_bboxes:
            start_row, start_col, side = bbox
            cv2.rectangle(image_color, (start_col, start_row), (start_col + side, start_row + side), (0,0,255), 2)
        # Regresa imagen en formato np.array, y los cuadros para las partículas
        return image_color, final_bboxes

if __name__ == '__main__':
    app = Detection_gui()
    app.mainloop()
        

# Testeo cambiar número de la interfaz
    # self.frame_settings.nanoparticle_counter.update_counter(208)

# Testeo Ggardar imagen en el directorio seleccionado
    # os.makedirs(self.frame_settings.directory_selector.directory, exist_ok=True)
    # im = Image.new(mode="RGB", size=(200, 200))
    # image_path = os.path.join(self.frame_settings.directory_selector.directory, "test_image.png")
    # im.save(image_path)
    # si me creo una variable self.directory = self.frame_settings.directory_selector.directory cambiaría automaticamente