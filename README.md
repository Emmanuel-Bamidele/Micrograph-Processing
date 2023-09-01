## Micrograph Processing Software
This is a Python application for processing micrograph images using the Tkinter GUI toolkit. The software provides several features for analyzing and enhancing micrograph images. It allows users to load images, set scale factors, calculate grain sizes, find grain boundaries, create colored grain images (similar to EBSD), and save processed images. The software is designed to assist users in performing various tasks related to micrograph analysis.

## Prerequisites

Python 3.x installed on your system.

Required libraries: tkinter, cv2 (OpenCV), numpy, PIL (Pillow), matplotlib, sklearn.

## Installation

Make sure you have Python and the required libraries installed.

Copy and paste the provided code into a Python file (e.g., micrograph_processing.py).

## Usage

Run the script using the command: python micrograph_processing.py.

The application window will open, providing a user interface to perform various tasks on micrograph images.

## Features

Load Image: Click the "Load Image" button to open an image using a file dialog.

Set Scale Factor: Draw a line on the image to set the scale factor for real-world measurements.

Calculate Grain Size: After setting the scale factor, calculate individual and average grain sizes by drawing lines and entering grain counts.

Find Grain Boundaries: Highlight grain boundaries in the image.

Colored Grains (EBSD): Create a colored grain image using K-means clustering.

Improve Quality: Apply smoothing to improve image quality.

Save Image: Save the processed image or the original image if no processing is applied.

Export Data to TXT: Export micrograph processing data to a text file.

## Notes

The software provides instructions on how to perform each step.

Each feature operates on the currently loaded image.

Grain size calculations are based on user-defined grain counts and scale factors.

The software allows visual enhancements and analysis of micrograph images.

## Author

Author: Emmanuel Bamidele

License: Apache 2.0

Feedback and Contributions

Feel free to provide feedback, suggestions, or contribute to this project by opening issues or pull requests on the GitHub repository. Your contributions are welcome and appreciated.
