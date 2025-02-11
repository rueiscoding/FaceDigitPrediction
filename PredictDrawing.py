import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from PredictDigit import PredictDigit

class PredictDrawing:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Number!")
        self.root.geometry("300x400")
        # self.root.configure(bg="pink")

        self.canvas = tk.Canvas(root, bg="white", width=280, height=280)
        self.canvas.pack(pady=10)

        self.button_save = tk.Button(root, text="Submit", command=self.save_and_process, relief="flat")
        self.button_save.pack()

        self.button_clear = tk.Button(root, text="Clear", command=self.clear_canvas, relief="flat")
        self.button_clear.pack()

        #label for result
        self.result_label = tk.Label(root, text="Prediction: ", font=("Arial", 12))
        self.result_label.pack(pady=10)

        self.image = Image.new("L", (280, 280), 255) 
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=12)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

    def image_to_feature(self, image):
        count = 0
        result_array = [0] * 784
        for i in range(28):
            for j in range(28):
                if image[i][j] != 255: # 255 is white space
                    result_array[count] = 1
                count+=1

        return result_array

    def save_and_process(self):
        self.image = self.image.resize((28, 28))
        # img_array = np.array(self.image) / 255.0 
        img_array = np.array(self.image)

        #load theta1 and theta2 
        theta1 = np.loadtxt("digitstheta1.txt")
        theta2 = np.loadtxt("digitstheta2.txt")

        feature_array = self.image_to_feature(img_array)

        a_3 = PredictDigit.forward_pass(theta1, theta2, feature_array)
        prediction = np.argmax(a_3)
        # print("Neural Network Prediction: " + str(prediction))
        self.result_label.config(text=f"Prediction: {prediction}")


def main():
    root = tk.Tk()
    app = PredictDrawing(root)
    root.mainloop()

if __name__ == "__main__":
    main()