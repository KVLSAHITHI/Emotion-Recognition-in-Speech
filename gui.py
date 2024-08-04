#u can edit the gui code according to your requirements this is just a basic implementation

import tkinter as tk
from tkinter import filedialog, messagebox

def browse_file(predict_emotion_func):
    filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if filename:
        emotion = predict_emotion_func(filename)
        messagebox.showinfo("Predicted Emotion", f"The emotion in the speech is: {emotion}")

def create_gui(predict_emotion_func):
    root = tk.Tk()
    root.title("Speech Emotion Recognition")
    root.geometry("500x300")

    browse_button = tk.Button(root, text="Upload Audio File", command=lambda: browse_file(predict_emotion_func))
    browse_button.pack(pady=20)

    root.mainloop()
