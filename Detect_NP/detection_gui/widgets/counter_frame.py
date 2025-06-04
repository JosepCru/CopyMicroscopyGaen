import tkinter as tk
import customtkinter as ctk

class Counter_frame(ctk.CTkFrame):
    def __init__(self, root, text, font = ('American typewriter', 24), *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        self.counter_var = tk.IntVar(self, 0)

        self.counter_label = ctk.CTkLabel(self, text=text, font=font)
        self.counter_count = ctk.CTkLabel(self, textvariable=self.counter_var, font=('American typewriter', 18))

        self.counter_label.pack(pady=5)
        self.counter_count.pack(pady=5)

    def update_counter(self, count):
        self.counter_var.set(count)