"""
Author: Emmanuel Bamidele
License: Apache 2.0
"""

import tkinter as tk
from tkinter import filedialog, simpledialog, Canvas, Frame
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from PIL import ImageFilter
from matplotlib import cm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class MicrographProcessing:
    def __init__(self, master):
        self.master = master
        self.master.title("Micrograph Processing,  © Emmanuel Bamidele")

        # Create canvas with initial dimensions, these could be anything
        self.canvas = Canvas(self.master, bg="white", width=500, height=350)
        self.canvas.pack()

        # Create a frame for the buttons
        button_frame = Frame(self.master)
        button_frame.pack(side="bottom", fill="x")

        # Create the buttons and pack them using the pack() geometry manager

        self.load_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side="left", padx=5, pady=5)

        self.grain_size_button = tk.Button(button_frame, text="Calculate Grain Size", command=self.calculate_grain_size)
        self.grain_size_button.pack(side="left", padx=5, pady=5)

        self.grain_boundary_button = tk.Button(button_frame, text="Find Grain Boundaries",
                                               command=self.find_grain_boundary)
        self.grain_boundary_button.pack(side="left", padx=5, pady=5)

        self.smooth_button = tk.Button(button_frame, text="Improve Quality", command=self.smooth_image)
        self.smooth_button.pack(side="left", padx=5, pady=5)

        self.ebsd_button = tk.Button(button_frame, text="Colored Grains", command=self.create_ebsd_image)
        self.ebsd_button.pack(side="left", padx=5, pady=5)

        self.save_image_button = tk.Button(button_frame, text="Save Image", command=self.save_image)
        self.save_image_button.pack(side="left", padx=5, pady=5)

        self.export_data_button = tk.Button(button_frame, text="Export Data to TXT", command=self.export_data)
        self.export_data_button.pack(side="left", padx=5, pady=5)

        # Show the initial instructions pop-up
        self.show_initial_instructions()

        # Initialize Variables
        self.img = None
        self.cv_img = None
        self.original_cv_img = None
        self.tk_img = None
        self.scale_factor = None
        self.canvas_image_id = None
        self.preview_img = None
        self.line = None
        self.grain_size = None
        self.processed_img = None
        self.top_grain_size = None  # Individual grain size for top line
        self.middle_grain_size = None  # Individual grain size for middle line
        self.bottom_grain_size = None  # Individual grain size for bottom line
        self.avg_grain_size = None  # Average grain size
        self.line_locations = None  # Added this line


        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.image_canvas = Canvas(self.master, bg="white", width=500, height=350)

        self.preview_img = None

        self.line = None
        self.grain_size = None

    def show_initial_instructions(self):
        instructions = (
            "Welcome to Micrograph Processing Software!\n\n"
            "Follow these steps to process your micrograph:\n\n"
            "1. Click the 'Load Image' button to open an image.\n"
            "2. Draw a line on the image to set image scale.\n"
            "3. Click the 'Calculate Grain Size' button to compute grain size.\n"
            "4. Click the 'Find Grain Boundaries' button to highlight boundaries.\n"
            "5. Click the 'Colored Grains' button to create an EBSD-like image.\n\n"
            "Remember to save your work using the 'Save Image' button."
        )
        messagebox.showinfo("Instructions", instructions)

    def clear_canvas(self):
        # Delete any existing canvas items, including the instructions
        self.canvas.delete("all")
        self.instructions_text_id = None  # Reset the instructions ID

    def load_image(self):
        self.clear_canvas()  # Clear the canvas first

        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            self.cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self.original_cv_img = self.cv_img.copy()  # Store the original image

            # Convert PIL Image to OpenCV format for further processing
            self.cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Aspect ratio of the original image
            aspect_ratio = img.width / img.height

            # Choose the desired width or height of the image on the canvas
            # For example, I chose a width of 400 pixels
            new_width = 400

            # Calculate corresponding height while maintaining aspect ratio
            new_height = int(new_width / aspect_ratio)

            # Add padding (e.g., 50 pixels on each side)
            canvas_width = new_width + 100
            canvas_height = new_height + 100

            # Resize canvas
            self.canvas.config(width=canvas_width, height=canvas_height)

            # Resize image
            img = img.resize((new_width, new_height), 1)

            self.img = ImageTk.PhotoImage(img)

            # Add image to canvas, considering the padding
            self.canvas_image_id = self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.img)

            # Update the canvas immediately
            self.canvas.update_idletasks()

    def show_image(self):
        if self.img is not None:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.tk_img = ImageTk.PhotoImage(image=img_pil)

            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Calculate position to center the image
            x_pos = canvas_width // 2
            y_pos = canvas_height // 2

            if self.canvas_image_id:
                self.canvas.delete(self.canvas_image_id)
            self.canvas_image_id = self.canvas.create_image(x_pos, y_pos, image=self.tk_img)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.line = None

    def on_mouse_drag(self, event):
        if not self.line:
            self.line = self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill="red")
        else:
            self.canvas.coords(self.line, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        end_x, end_y = event.x, event.y
        line_length_pixel = ((end_x - self.start_x) ** 2 + (end_y - self.start_y) ** 2) ** 0.5
        line_length_real = simpledialog.askfloat("Input", "Enter the real length of the line:")
        unit = simpledialog.askstring("Input", "Enter the unit of the length:")

        if line_length_real is not None and unit:  # Ensure the user entered a value and unit
            self.scale_factor = line_length_real / line_length_pixel
            messagebox.showinfo("Info", f"Scale factor set: {self.scale_factor} {unit}/pixel")

    def calculate_grain_size(self):
        if self.img is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        if self.scale_factor is None:
            messagebox.showwarning("Warning", "Please set the scale factor first by drawing a line.")
            return

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate line locations (starting from index 1 to skip the first line)
        line_spacing = canvas_height // 16  # Adjust this value for desired spacing
        line_padding = canvas_width // 8  # Adjust this value to control line padding

        line_locations = [3 * line_spacing, 5 * line_spacing, 7 * line_spacing, 9 * line_spacing, 11 * line_spacing]

        line_ids = []  # To store the IDs of the drawn lines

        # Draw the lines at desired positions, spanning the width of the canvas
        for loc in line_locations:
            line_id = self.canvas.create_line(line_padding, loc, canvas_width - line_padding, loc, fill="red")
            line_ids.append(line_id)

        grain_counts = []  # To store the grain counts for each line

        # Get grain counts for each line
        for i in range(5):
            try:
                grains = int(self.get_user_input(f"Enter the number of grains on line {i + 1}:"))
                grain_counts.append(grains)
            except ValueError:
                messagebox.showwarning("Warning", "Invalid input. Please enter a valid number.")
                return

        # Calculate individual grain sizes for each line
        grain_sizes = [((canvas_width - 2 * line_padding) / grain) * self.scale_factor for grain in grain_counts]

        # Calculate average grain size
        avg_grain_size = sum(grain_sizes) / len(grain_sizes)

        # Delete the drawn lines
        for line_id in line_ids:
            self.canvas.delete(line_id)

        # Display individual grain sizes and average grain size
        result_message = "\n".join([f"Line {i + 1} Grain Size: {grain_sizes[i]:.2f}" for i in range(len(grain_sizes))])
        result_message += f"\n\nAverage Grain Size: {avg_grain_size:.2f}"
        messagebox.showinfo("Grain Sizes", result_message)

    def show_image_with_lines(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(image=img_pil)

        # Update the canvas image without changing the canvas dimensions
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_img)

    def get_user_input(self, prompt):
        return simpledialog.askstring("Input", prompt)

    def find_grain_boundary(self):
        if self.cv_img is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Convert the image to grayscale
        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)

        # apply a Gaussian blur to the image (this can help with edge detection)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use the Canny edge detector to find edges in the image
        edges = cv2.Canny(gray_blurred, 50, 150)

        # Optional: Dilate the edges to make them more visible
        dilated_edges = cv2.dilate(edges, None)

        # Color the boundaries
        # Creating a mask from the edges
        mask = dilated_edges > 0

        # Create a blank output image with the same dimensions as cv_img
        output = np.zeros_like(self.cv_img)

        # Assign new color to the boundary regions
        output[mask] = [0, 0, 255]  # Assigning red color to the boundaries

        # Combine with original image
        combined = cv2.addWeighted(self.cv_img, 0.8, output, 0.2, 0)

        # Convert combined image to RGB
        img_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img_pil = Image.fromarray(img_rgb)

        # Calculate aspect ratio
        aspect_ratio = img_pil.width / img_pil.height

        # Set new dimensions
        max_width = 400  # or whatever maximum width you want for display
        max_height = int(max_width / aspect_ratio)

        # Resize image
        img_resized = img_pil.resize((max_width, max_height), 1)

        # Create PhotoImage
        self.tk_img = ImageTk.PhotoImage(image=img_resized)

        # Calculate position to center image on canvas
        center_x = self.canvas.winfo_width() // 2
        center_y = self.canvas.winfo_height() // 2

        # Update the canvas image
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_img)

        # Update canvas dimensions
        self.canvas.config(width=max_width, height=max_height)

        # Move the image to the center of the canvas
        self.canvas.coords(self.canvas_image_id, center_x, center_y)

        messagebox.showinfo("Info", "Grain boundaries found and colored.")
        self.cv_img = combined

    def create_ebsd_image(self):
        if self.cv_img is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Convert the image to grayscale
        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)

        # Apply K-means clustering
        flattened_image = gray.flatten().reshape(-1, 1)
        kmeans = KMeans(n_clusters=10)  # You can adjust the number of clusters
        kmeans.fit(flattened_image)
        labels = kmeans.labels_.reshape(gray.shape)

        # Initialize an empty color image
        output = np.zeros((self.cv_img.shape[0], self.cv_img.shape[1], 3), dtype=np.uint8)

        # Assign colors based on labels
        for label in np.unique(labels):
            # Get a color from the colormap
            color = cm.jet(float(label) / len(np.unique(labels)))
            color_rgb = tuple((np.array(color[:3]) * 255).astype(np.uint8))

            # Assign this color to all pixels belonging to this label
            output[labels == label] = color_rgb

        # Convert to RGB for PIL
        img_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img_pil = Image.fromarray(img_rgb)

        # Update the processed image
        self.processed_img = output

        # Create PhotoImage and update canvas
        self.tk_img = ImageTk.PhotoImage(image=img_pil)
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_img)

        messagebox.showinfo("Info", "EBSD-like image created.")

    def save_image(self):
        if self.cv_img is not None:  # Checking if an image has been loaded
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("All Files", "*.*")])
            if file_path:
                if self.ebsd_button['state'] == tk.NORMAL:  # EBSD button is enabled
                    cv2.imwrite(file_path, self.processed_img)  # Save the processed image
                else:
                    cv2.imwrite(file_path, self.cv_img)  # Save the original image
                messagebox.showinfo("Info", f"Image saved at {file_path}")
        else:
            messagebox.showwarning("Warning", "Please load an image first.")

    def smooth_image(self):
        if self.cv_img is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Resize the image to a larger size
        resized_img = cv2.resize(self.cv_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

        # Apply bilateral filter for better quality blurring
        smoothed_img = cv2.bilateralFilter(resized_img, 25, 100, 100)

        # Convert the smoothed image to RGB for display
        result_rgb = cv2.cvtColor(smoothed_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(result_rgb)

        # Calculate aspect ratio
        aspect_ratio = img_pil.width / img_pil.height

        # Set new dimensions for display
        max_width = 400  # or whatever maximum width you want for display
        max_height = int(max_width / aspect_ratio)

        # Resize image for canvas
        img_resized = img_pil.resize((max_width, max_height), 1)

        # Create PhotoImage and update canvas
        self.tk_img = ImageTk.PhotoImage(image=img_resized)
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_img)

        # Calculate position to center image on canvas
        center_x = self.canvas.winfo_width() // 2
        center_y = self.canvas.winfo_height() // 2

        # Update canvas dimensions
        self.canvas.config(width=max_width, height=max_height)

        # Move the image to the center of the canvas
        self.canvas.coords(self.canvas_image_id, center_x, center_y)

        messagebox.showinfo("Info", "Image quality improved.")
        self.processed_img = smoothed_img

    def export_data(self):
        if self.cv_img is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Gather information
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w") as f:
                f.write("Micrograph Processing Data Report\n")
                f.write("-" * 40 + "\n")

                # Image information
                f.write("Image File: {}\n".format(file_path))
                f.write("Image Dimensions: {} x {}\n".format(self.img.width(), self.img.height()))
                f.write("Scale Factor: {}\n".format(self.scale_factor))

                # Processed image information
                if self.processed_img is not None:
                    processed_img_name = "processed_image.png"  # Update with the actual processed image name
                    f.write("Processed Image: {}\n".format(processed_img_name))
                else:
                    f.write("Processed Image: Not available\n")

                # Grain size information
                if self.avg_grain_size:
                    f.write("Average Grain Size: {:.2f} (real-world units)\n".format(self.avg_grain_size))
                else:
                    f.write("Average Grain Size: Not calculated\n")

                # Individual grain sizes for each line
                f.write("-" * 40 + "\n")
                f.write("Individual Grain Sizes:\n")
                for i, grain_size in enumerate(self.grain_sizes, start=1):
                    f.write(f"Line {i} Grain Size: {grain_size:.2f} (real-world units)\n")

            messagebox.showinfo("Info", f"Data exported to {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MicrographProcessing(root)
    root.mainloop()
