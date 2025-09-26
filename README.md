# Computer-Vision-AR-app
Webcam Computer Vision Explorer
This is a Python application that lets you explore a wide range of computer vision features interactively using your webcam. It's built with OpenCV and designed to be a hands-on learning tool.

Features
The application comes with multiple modes you can cycle through:

Normal Mode – A plain, unaltered webcam feed.

Camera Calibration – Use a chessboard pattern to calibrate your webcam and save the parameters.

Augmented Reality – Project a 3D T-Rex model onto a printed ArUco marker. 🦖

Color Space Conversion – Switch between RGB, Grayscale, and HSV color models.

Histogram Visualization – See the live distribution of pixel intensities for the R, G, and B channels.

Brightness & Contrast – Adjust the brightness and contrast of the live feed.

Gaussian & Bilateral Blur – Apply and adjust different types of smoothing filters.

Canny Edge Detection – View a real-time Canny edge detection filter.

Hough Line Detection – Detect and highlight straight lines in the video feed.

Geometric Transformations – Interactively rotate, translate, and scale the video stream.

Panorama Mode – A framework to capture two images to be stitched together.

Requirements
Python 3.8+

A working webcam.

A terminal or command prompt (Windows, macOS, or Linux).

Installation
Download the Project Files
Make sure you have all the project files (app.py, trex_model.obj, etc.) in a single folder on your computer.

Open a Terminal in the Project Folder

Windows: Navigate into the folder, hold Shift + Right Click on an empty space, and select Open PowerShell window here.

macOS/Linux: Open your terminal and use the cd command to navigate to your project folder.

cd path/to/your/project-folder

Create a Virtual Environment (Recommended)
This keeps the project's libraries separate from your system's Python.

python -m venv venv

Activate the environment:

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

You should see (venv) appear at the start of your terminal prompt.

Install Required Libraries
Install all the necessary packages using the provided requirements.txt file.

pip install -r requirements.txt

Run the Application
You're all set! Run the main script to start the application.

python app.py

How to Use
Printable Assets
You need to print two files for the Calibration and AR modes to work:

Calibration Chessboard → Print the A4_Chessboard_9x6.png file.

ArUco Marker → Print the A4_ArUco_Marker.png file.

General Controls
Key

Function

m

Cycle to the next mode

q

Quit the application

Mode-Specific Controls
CALIBRATE Mode

SPACE - Capture the current view of the chessboard.

c - Perform calibration (requires at least 10 captures).

r - Reset and delete all captured images.

COLOR_SPACE Mode

c - Cycle between RGB, GRAY, and HSV.

BRIGHTNESS_CONTRAST Mode

w / s - Increase / Decrease Brightness.

e / d - Increase / Decrease Contrast.

GAUSSIAN Mode

g / h - Increase / Decrease Gaussian kernel size.

BILATERAL Mode

b / n - Increase / Decrease Bilateral filter diameter.

TRANSFORM Mode

w/a/s/d - Translate the image Up/Left/Down/Right.

, / . - Rotate the image Left / Right.

z / x - Scale the image Up / Down.

PANORAMA Mode

c - Capture an image (capture two to trigger stitching).

r - Reset the captured images.
