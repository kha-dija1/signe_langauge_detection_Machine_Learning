import tkinter as tk
from tkinter import filedialog
import customtkinter
from PIL import ImageTk, Image
class CTkButton(tk.Button):
    def _init_(self, master=None, **kwargs):
        print("true CTkButton")
        super()._init_(master, **kwargs)
        self.configure(bg='blue', fg='white', activebackground='#333333', activeforeground='white')

class CustomApp(tk.Tk):
    def __init__(self):
        super().__init__()
        print("true CustomApp")

        self.title("Custom App")
        self.geometry("300x500")
        self.configure(bg='#708090')

        self.top_frame = tk.Frame(self)
        self.top_frame.pack(fill=tk.BOTH, expand=True)
        self.left_frame = tk.Frame(self.top_frame, bg='#708090')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self.top_frame, bg='#708090')
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.bottom_frame = tk.Frame(self, bg='#556B2F',)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True)

        self.image_frame = tk.Frame(self.left_frame, bg='black')
        self.image_frame.pack(side=tk.TOP, pady=10)
        self.uplead=customtkinter.CTkButton(master=self.left_frame,
                                 text="give image",
                                 command=self.button_event,
                                 width=200,
                                 height=32,
                                 border_width=0,
                                 corner_radius=8)
        self.uplead.pack(side=tk.TOP, pady=10)
        self.text_frame = tk.Frame(self.left_frame,bg="#708090")
        self.text_frame.pack(side=tk.TOP,pady=10)
        self.label = customtkinter.CTkLabel(master=self.text_frame,
                              text="this image is",
                              width=10,
                              height=3,
                              )
        self.label.grid(row=1, column=0, padx=10, pady=10)

        self.entry = customtkinter.CTkEntry(master=self.text_frame,
                              width=100,
                              height=3,
                              corner_radius=10)
        self.entry.grid(row=1, column=1, padx=10, pady=10)

        self.small_circle_button = customtkinter.CTkButton(master=self.left_frame,
                                                     text="Get Text",
                                                     command=self.get_text,
                                                           width=200,
                                                           height=32,
                                                           border_width=0,
                                                           corner_radius=8)
        self.small_circle_button.pack(side=tk.TOP, pady=10)
        self.image_path = "../download (1).png"  # Replace with your image path
        self.display_image()



        self.button_frame = tk.Frame(self.right_frame, bg='#000080')
        self.button_frame.pack(pady=10)

        for i in range(6):

            button=customtkinter.CTkButton(master=self.right_frame,
                                    text=f"Button {i+1}",
                                    command=lambda idx=i+1: self.button_callback(idx),
                                    width=120,
                                    height=32,
                                    border_width=0,
                                    corner_radius=8,fg_color="#556B2F")
            button.pack(fill=tk.X, padx=20, pady=10)
        self.label_frame = tk.Frame(self.bottom_frame, bg='#000080')
        self.label_frame.pack(pady=10)


        for i in range(6):
            result = tk.StringVar()
            result_label = tk.Label(self.label_frame, textvariable=result, bg='black', fg='white')
            result_label.pack(pady=10)

    def get_text(self):
        text = self.entry.get()
    def display_image(self):
        image = Image.open(self.image_path)
        image = image.resize((200, 200), Image.ANTIALIAS)  # Adjust image size as needed
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(self.image_frame, image=photo, bg='black')
        image_label.image = photo
        image_label.pack()
    def button_callback(self, idx):
        self.result.set(f"Button {idx} clicked!")

    def button_event(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                              filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
        if filename:
            # Do something with the selected file, such as processing or displaying it
            print("Selected file:", filename)

        else:
            print("No file selected")

if __name__ == "__main__":
    print("true")
    app = CustomApp()
    app.mainloop()
