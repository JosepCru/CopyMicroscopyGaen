import tkinter as tk
import customtkinter as ctk

class ChangeImage_Frame(ctk.CTkFrame):
    def __init__(self, parent, default_directory="directorio", *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.title_label = ctk.CTkLabel(self, text="Change image", font = ('American typewriter', 18), anchor='center')
        self.title_label.pack(padx = 5, pady = 10)

        self.previous_button = ctk.CTkButton(self, text="Change image", font=('American typewriter', 18))
        self.next_button = ctk.CTkButton(self, text="Change image", font=('American typewriter', 18))

        self.previous_button.pack(side="left", padx=10, pady=10)
        self.next_button.pack(side="left", padx=10, pady=10)

