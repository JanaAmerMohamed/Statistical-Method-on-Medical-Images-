import cv2
import numpy as np
from tkinter import Tk, Button, Label, filedialog, Text, PhotoImage, Canvas
from PIL import Image, ImageTk

# --- Image Processing Functions ---

def calculate_average(image):
    avg = np.mean(image, axis=(0, 1))
    return avg

def calculate_variance(image):
    var = np.var(image, axis=(0, 1))
    return var

def calculate_normalized_variance(variance):
    max_var = np.max(variance)
    normalized_var = variance / max_var if max_var != 0 else variance
    return normalized_var

def calculate_nth_moment(image, n):
    mean = calculate_average(image)
    moment = np.mean((image - mean) ** n, axis=(0, 1))
    return moment

def calculate_uniformity(image):
    uniformity = []
    for channel in range(3):  # RGB
        hist, _ = np.histogram(image[:, :, channel], bins=256, range=(0, 256), density=True)
        uniform = np.sum(hist ** 2)
        uniformity.append(uniform)
    return uniformity

def calculate_entropy(image):
    entropy = []
    for channel in range(3):
        hist, _ = np.histogram(image[:, :, channel], bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        ent = -np.sum(hist * np.log2(hist))
        entropy.append(ent)
    return entropy

# --- GUI Application ---

class ImageStatsApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Image Statistical Analyzer")

        self.label = Label(root, text="Choose an Image to Analyze", font=('Arial', 14))
        self.label.pack(pady=10)

        self.canvas = Canvas(root, width=300, height=300)
        self.canvas.pack()

        self.button = Button(root, text="Browse Image", command=self.load_image)
        self.button.pack(pady=10)

        self.results_text = Text(root, height=15, width=80)
        self.results_text.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display image
            pil_image = Image.fromarray(image_rgb)
            pil_image_resized = pil_image.resize((300, 300))
            self.tk_image = ImageTk.PhotoImage(pil_image_resized)
            self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

            # Compute stats
            avg = calculate_average(image_rgb)
            var = calculate_variance(image_rgb)
            norm_var = calculate_normalized_variance(var)
            nth_moment = calculate_nth_moment(image_rgb, 3)  # 3rd moment
            uniformity = calculate_uniformity(image_rgb)
            entropy = calculate_entropy(image_rgb)

            result_str = f"""
Image: {file_path}

Average (R, G, B): {avg.round(2)}
Variance (R, G, B): {var.round(2)}
Normalized Variance (R, G, B): {norm_var.round(4)}
3rd Moments (R, G, B): {nth_moment.round(2)}
Uniformity (R, G, B): {np.round(uniformity, 4)}
Entropy (R, G, B): {np.round(entropy, 4)}
"""
            self.results_text.delete(1.0, "end")
            self.results_text.insert("end", result_str)

# --- Main Execution ---

if _name_ == "_main_":
    root = Tk()
    app = ImageStatsApp(root)
    root.mainloop()