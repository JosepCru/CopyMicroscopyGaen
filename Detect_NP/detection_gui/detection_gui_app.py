import customtkinter as ctk
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from widgets.interactive_image import Interactive_image
import torch
import torch.utils.data
import cv2
from auxiliar_functions import create_boxes, isolate_particles

#from autoscript_tem_microscope_client import TemMicroscopeClient
#from autoscript_tem_microscope_client.enumerations import *
#from autoscript_tem_microscope_client.structures import *


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
        self.model_ml = torch.load('ModelNP_newContrast_900.pt')
        self.load_sample()


# This is the function that we call when the button Start is pressed, we should start here the routine of detection
    def toggle_acquisition(self):
        if not self.is_running:
            self.start_acquisition()

            num_images=int(self.np_number.get())

            for i in range(num_images):
                image = f"C:\\Users\\josep\\Desktop\\extra_training\\images\\image_{i:04d}.png"
                image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

                # Movement and detection

                self.number_added = int(self.np_index.get()) + 1
                self.np_index.set(self.number_added)
                self.detect_and_plot(image, th=0.2)  # Assuming the image is a grayscale image

                #microscope = TemMicroscopeClient()
                #microscope.connect()
                #microscope.optics.optical_mode = OpticalMode.STEM
                #num_images = 1000
                #step_size = 0.0001
                #movement_direction = self.build_spiral_coordinates(total_cells=total_steps)
                #for (x, y) in movement_direction:
                #    self.movement(x, y, step_size, microscope)

            # Testing, we should start here the routine of detection
            #self.number_added = int(self.np_index.get()) + 1
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

    # <-------------------------------------------------------------------------Particle Plotting Functions------------------------------------------------------------------------->
    
    def prediction(self, image, th=0.5):
        pred, zones = self.model_ml.predict(image, th=th)
        pred_mask = np.squeeze(pred[0])
        labeled_mask, num_particles = isolate_particles(pred_mask, th)
        return labeled_mask, zones, num_particles, pred_mask

    def plot_particles(self, image, final_bboxes,image_color=None):
        for bbox in final_bboxes:
            start_row, start_col, side = bbox
            cv2.rectangle(image_color,
                        (start_col, start_row),
                        (start_col + side, start_row + side),
                        (0, 0, 255), 4)
            r, c, s = bbox
            particle_img = image[r:r+s, c:c+s]
            particle_crop = np.array(particle_img).astype(np.uint8)

            new_nano = Interactive_image(self.frame_acquisition, particle_crop, title = f'NP', subtitle=f'Coords:', font=('American typewriter', 24), subfont=('American typewriter', 14), width=150, height=150)
            new_nano.pack(side = 'left')

    def detect_and_plot(self, image, th = 0.5, min_particle_size = 0.1):
        self.image_microscope.change_image(image)
        image_pred, zones, num_particles, pred_mask = self.prediction(image, th=th)
        self.image_prediction.change_image(image_pred)
        centers, final_bboxes = create_boxes(image, min_particle_size=min_particle_size, th=th, 
                                                               num_particles=num_particles, labeled_mask=image_pred, zones=zones, pred_mask=pred_mask)
        
        image_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        self.image_microscope.change_image(image_boxes)

        # Quitar este y cambiar por movimiento y captura de nanoparticulas
        self.plot_particles(image, final_bboxes, image_color=image_boxes)

    # <-------------------------------------------------------------------------Microscope Movement Functions------------------------------------------------------------------------->
    def build_spiral_coordinates(self, total_cells=12):
        coord_initial = []
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        direction_index = 0
        step_count = 0
        step_limit = 1
        direction_changes = 0

        while len(coord_initial) < total_cells:

            coord_initial.append(directions[direction_index])
            step_count += 1

            if step_count == step_limit:
                step_count = 0
                direction_index = (direction_index + 1) % 4
                direction_changes += 1
                
                if direction_changes % 2 == 0:
                    step_limit += 1

        return coord_initial

    #def movement(self, grid_x, grid_y, step_size, microscope):
    #    microscope.specimen.stage.relative_move(StagePosition(x=grid_x * step_size, y=grid_y * step_size))
    #    image = microscope.acquisition.acquire_stem_image(DetectorType.HAADF, ImageSize.PRESET_512, 1e-6) # Acquire the image
    #    self.detect_and_plot(image=image, th=0.2)  # Detect and plot the particles in the image

    
# This is the main function that runs the GUI application   
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