# Computer Vision App

This repository contains a single Python application (`app.py`) that demonstrates various computer vision concepts, including image manipulation, filtering, geometric transformations, camera calibration, and Augmented Reality.

### 1. Requirements and Setup

Before running the application, you need to have Python and a few libraries installed.

* **Python:** Ensure Python 3.6 or newer is installed.

* **Libraries:** Install the required libraries using pip:


pip install opencv-python numpy


* **Files:** Place the following files in the same directory as `app.py`:

* `A4_ArUco_Marker.png`

* `A4_Chessboard_9x6.png`

* `trex_model.obj`

### 2. How to Run the App

Simply run the main application file from your terminal:


python app.py


A window named "Computer Vision App" will appear, showing your webcam feed. You can switch between different modes by pressing the keys listed below.

### 3. Application Functions & Controls

Use the following keyboard shortcuts to control the application's functions.

* `ESC`: Exit the application.

* `s`: Save a snapshot of the current view.

**Core Image Functions**

* `c`: **Color Conversion**. Press `c` repeatedly to cycle between RGB (normal), Grayscale, and HSV color spaces. A live histogram is also displayed in this mode.

* `b`: **Brightness & Contrast**. Use trackbars to adjust the brightness and contrast of the image.

* `g`: **Gaussian Filter**. Blurs the image using a Gaussian kernel. Use the "Kernel Size" trackbar to adjust the blur level.

* `f`: **Bilateral Filter**. A non-linear filter that smooths the image while preserving edges. Use the trackbars to control its parameters.

* `e`: **Canny Edge Detection**. Detects edges in the image. Use the "Threshold1" and "Threshold2" trackbars to control the sensitivity.

* `h`: **Hough Transform**. Detects straight lines in the image using the Hough Transform algorithm. Adjust parameters with trackbars.

**Geometric Transformations**

* `t`: **Image Translation, Rotation, & Scale**. Adjust the "tx", "ty", "angle", and "scale" trackbars to apply a 2D affine transformation to the image.

**Camera Calibration**

* `l`: **Calibrate the Camera**. Point your camera at a printed 9x6 chessboard pattern (like `A4_Chessboard_9x6.png`). When the board is detected, press `s` to save a frame. Once enough frames are collected, the script will compute and save the camera matrix and distortion coefficients to `calibration.npz`.

**Augmented Reality (AR)**

* `a`: **Augmented Reality Mode**. This mode requires a pre-calibrated camera and the `A4_ArUco_Marker.png` file. The application will detect the marker and project a 3D TREX model (from `trex_model.obj`) onto it. The model's size has been increased for better visibility.

  * **Note:** The `.obj` file is parsed using a custom-written function.

**Panorama**

* `p`: **Panorama Mode**. This is a placeholder function for the panorama feature. To use this, take multiple images of a scene by pressing `s` as you pan your camera. You would then need to write a separate script to load these images and perform image stitching.

### 4. Custom Functions

This application uses custom functions for the following tasks, as requested in the assignment:

* **OBJ File Parsing**: A class `ObjectLoader` is implemented to manually read the `trex_model.obj` file and extract the vertex and face data.

* **Camera Calibration**: The calibration process is implemented from scratch, using `findChessboardCorners` and `calibrateCamera` to compute the camera matrix and distortion coefficients.

* **Hough Transform**: The line detection is performed using the `HoughLines` function.

* **Image Transformations**: The geometric transformations are applied using a custom-built affine transformation matrix.
