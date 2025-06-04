import customtkinter as ctk
import tkinter as tk
import os
from PIL import Image, ImageTk

from widgets.titled_frame import Frame_titled
from widgets.parameters_frame import Parameters_frame
from widgets.counter_frame import Counter_frame 
from widgets.directory_frame import Directory_frame
from widgets.changeimage_frame import ChangeImage_Frame

class Frame_settings(ctk.CTkFrame):
    def __init__(self, root, project_path, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        ctk.CTkLabel(self, text = 'Settings',anchor = 'center', font=('American typewriter', 32), fg_color='#1f6aa5').pack(fill = 'x', padx = 5, pady = 5)
 
        # Frame Acquisition
        self.frame_acquisition = Frame_titled(self, 'Acquistion settings')
        self.parameters = Parameters_frame(self.frame_acquisition, ['# NPs images'])
        self.nanoparticle_counter = Counter_frame(self.frame_acquisition, text = "Nanoparticles Detected:", font=('American typewriter', 20))
        self.button_play = ctk.CTkButton(self.frame_acquisition, text = 'Start acquisition', cursor =  'hand2', fg_color='green', font=('American typewriter', 20))
        self.frame_image= Frame_titled(self, 'Navigation')
        self.directory_selector = Directory_frame(self.frame_image, max_length=12, default_directory = os.path.abspath(os.path.join("..", "Nanoparticle_Name_Images")))
        self.change_image = ChangeImage_Frame(self.frame_image, default_directory = os.path.abspath(os.path.join("..", "Nanoparticle_Name_Images")))

        # Layout
        self.frame_acquisition.pack(padx = 5, pady = 10, fill = 'x')
        self.parameters.pack(padx = 5, pady = 10, fill = 'x')
        self.nanoparticle_counter.pack(padx=5, pady=10, fill='x')
        self.button_play.pack(padx = 5, pady = 10, fill = 'x')
        self.frame_image.pack(padx = 5, pady = 10, fill = 'x')
        self.change_image.pack(padx = 5, pady = 10, fill = 'x')
        self.directory_selector.pack(padx=5, pady=10, fill='x')  # 🔹 **Añadir al layout**