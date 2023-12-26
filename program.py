import tkinter as tk
from tkinter import filedialog
import customtkinter
from PIL import ImageTk, Image
from signe_langauge.algos.SVM import SVM
from signe_langauge.algos.knnAlgo import KNN
from signe_langauge.algos.perceptron import perceptron
from signe_langauge.algos.preproces import preproces
from signe_langauge.algos.TREE import TREE
from signe_langauge.algos.regression_logistic import RL
from signe_langauge.algos.Random_forest import RF
Algorithmes = [KNN, perceptron, SVM, TREE, RF, RL]
buttons = ['K Nearest Neighbors', 'Perceptron', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
           'Logistic regression']
button_colors = ['#FFC107', '#E91E63', '#9C27B0', '#3F51B5', '#009688', '#FF5722']


class CTkButton(tk.Button):
    def _init_(self, master=None, **kwargs):
        super()._init_(master, **kwargs)
        self.configure(bg='blue', fg='white', activebackground='#333333', activeforeground='white')


class CustomApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("ASL")
        self.geometry("800x800")
        self.configure(bg='#708090')

        self.button_colors = button_colors

        self.top_frame = tk.Frame(self, bg='#F5F5F5')  # Gris clair
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.top_frame, bg='#FFFFFF')  # Blanc
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_label = None

        self.right_frame = tk.Frame(self.top_frame, bg='#FFFFFF')  # Blanc
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.bottom_frame = tk.Frame(self, bg='#F5F5F5')  # Gris clair
        self.bottom_frame.pack(fill=tk.BOTH, expand=True)

        self.user_input_label = customtkinter.CTkLabel(text="",master=self.bottom_frame, width=120, height=15,bg_color="#F5F5F5")
        self.user_input_label.pack(pady=5)

        self.image_frame = tk.Frame(self.left_frame, bg='#E5E5E5')  # Gris légèrement plus foncé
        self.image_frame.pack(side=tk.TOP, pady=10)

        self.uplead = customtkinter.CTkButton(master=self.left_frame, text="Give Image", command=self.button_event,
                                              width=200, height=32, border_width=0, corner_radius=8)
        self.uplead.pack(side=tk.TOP, pady=10)

        self.text_frame = tk.Frame(self.left_frame, bg='#FFFFFF')  # Blanc
        self.text_frame.pack(side=tk.TOP, pady=10)

        self.label = customtkinter.CTkLabel(master=self.text_frame, text="This image is", width=10, height=3)
        self.label.grid(row=1, column=0, padx=10, pady=10)

        self.entry = customtkinter.CTkEntry(master=self.text_frame, width=100, height=3, corner_radius=10)
        self.entry.grid(row=1, column=1, padx=10, pady=10)

        self.small_circle_button = customtkinter.CTkButton(master=self.left_frame, text="Get Text",
                                                           command=self.get_text,
                                                           width=200, height=32, border_width=0, corner_radius=8)
        self.small_circle_button.pack(side=tk.TOP, pady=10)

        self.image_path = ""

        self.display_image()

        self.button_frame = tk.Frame(self.right_frame, bg='#F5F5F5')  # Blanc
        self.button_frame.pack(pady=10)

        for i in range(len(buttons)):
            button = customtkinter.CTkButton(master=self.right_frame, text=buttons[i],
                                             command=lambda idx=i + 1: self.button_callback(idx),
                                             width=120, height=32, border_width=0, corner_radius=8,
                                             fg_color=button_colors[i % len(button_colors)])
            button.pack(fill=tk.X, padx=20, pady=10)

        self.label_frame = tk.Frame(self.bottom_frame, bg='#F5F5F5')  # Blanc
        self.label_frame.pack(pady=10)

        self.result_labels = []
        for i in range(len(buttons)):
            result_label = customtkinter.CTkLabel(master=self.label_frame, text=f"{buttons[i]}",
                                                   corner_radius=20,text_color=button_colors[i]
                                                 , width=120, height=15,bg_color='#F5F5F5')
             # Gris clair pour le fond, Noir pour le texte
            result_label.pack(pady=10)
            self.result_labels.append(result_label)

    def get_text(self):
        text = self.entry.get()

        # Display the user's input in the button_frame
        self.user_input_label.configure(text=f"This sign  is: {text}")

    def display_image(self):
        if self.image_label:
            self.image_label.destroy()  # Détruit le widget Label existant

        if self.image_path:
            image = Image.open(self.image_path)
            self.img = preproces(self.image_path)
        else:
            image = Image.open("asl_data.jpg")  # Chemin de l'image par défaut

        image = image.resize((200, 200), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(self.image_frame, image=photo, bg='black')
        image_label.image = photo
        image_label.pack()

        self.image_label = image_label  # Met à jour la référence de self.image_label

        self.algorithm_frames = []  # Store the frames for each algorithm

        for i in range(len(buttons)):
            algorithm_frame = tk.Frame(self.bottom_frame, bg=self.button_colors[i % len(self.button_colors)])
            self.algorithm_frames.append(algorithm_frame)

        self.hide_algorithm_frames()

    def hide_algorithm_frames(self):
        for algorithm_frame in self.algorithm_frames:
            algorithm_frame.pack_forget()

    def show_algorithm_frame(self, idx):
        self.hide_algorithm_frames()
        algorithm_frame = self.algorithm_frames[idx - 1]
        algorithm_frame.pack(fill=tk.BOTH, expand=True)

    def button_callback(self, idx):
        self.result_labels[idx - 1].configure(text=f"Configuring {buttons[idx - 1]}... Please wait.")
        result=Algorithmes[idx - 1](self.img)

        self.result_labels[idx - 1].configure(text=f"{buttons[idx-1]} thinks this is  {result} ")
        self.show_algorithm_frame(idx)

    def button_event(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                              filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
        if filename:
            self.image_path = filename
            self.display_image()
        else:
            print("No file selected")


if __name__ == "__main__":
    app = CustomApp()
    app.mainloop()