import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog

class Directory_frame(ctk.CTkFrame):
    def __init__(self, parent, default_directory="directorio", max_length=30, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.max_length = max_length

        self.directory = default_directory
        self.directory_var = tk.StringVar(self, self.shorten_path(default_directory))

        self.select_button = ctk.CTkButton(self, text="Change directory", command=self.change_directory, font=('American typewriter', 18))
        self.directory_label = ctk.CTkLabel(self, textvariable=self.directory_var, font=('American typewriter', 18), anchor="w")

        self.select_button.pack(side="left", padx=10, pady=10)
        self.directory_label.pack(side="left", padx=10, pady=10, fill="x", expand=True)
   
    def change_directory(self):
        new_directory = filedialog.askdirectory(title="Select a directory to save the images")

        if new_directory:
            self.directory = new_directory

            # ChatGPT, make the directory name shorter
            self.directory_var.set(self.shorten_path(new_directory))

    # ChatGPT, make the directory name shorter
    def shorten_path(self, path):
        if len(path) > self.max_length:
            return f"...{path[-self.max_length:]}"  # Mostrar solo los últimos caracteres con "..."
        return path

