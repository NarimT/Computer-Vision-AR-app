import cv2
import numpy as np
import os
import time

def load_obj(filename, scale_factor=1.0):
    """
    Loads a .obj file, centers it, normalizes its size, and applies a scale factor.
    Also rotates the T-Rex model to stand upright.
    """
    vertices = []
    faces = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.split()
                    face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    faces.append(face)
    except FileNotFoundError:
        print(f"Error: The model file '{filename}' was not found.")
        return None, None

    vertices = np.array(vertices, dtype=np.float32)
    
    if vertices.size == 0: return vertices, faces

    # Center, normalize, and scale
    mean_vertex = np.mean(vertices, axis=0)
    vertices -= mean_vertex
    max_coord = np.max(np.abs(vertices))
    if max_coord > 0:
        vertices /= max_coord
    vertices *= scale_factor

    # Rotate T-Rex to stand upright
    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    vertices = vertices @ rotation_matrix.T
    return vertices, faces

def draw_text(frame, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    """Draws white text with a black outline for better visibility."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# --- Main Application Class ---

class ComputerVisionApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.window_name = 'Computer Vision Assignment'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # --- Mode Management ---
        self.modes = [
            'NORMAL', 'CALIBRATE', 'AR', 'COLOR_SPACE', 'HISTOGRAM', 
            'BRIGHTNESS_CONTRAST', 'GAUSSIAN', 'BILATERAL', 'CANNY', 'HOUGH', 
            'TRANSFORM', 'PANORAMA'
        ]
        self.mode_index = 0
        self.mode = self.modes[self.mode_index]

        # --- State Variables for Different Modes ---
        self.brightness = 0
        self.contrast = 1.0
        self.gaussian_ksize = 5
        self.bilateral_d = 9
        self.transform_angle = 0
        self.transform_scale = 1.0
        self.transform_tx = 0
        self.transform_ty = 0
        
        # --- Color Space State ---
        self.color_space_options = ['RGB', 'GRAY', 'HSV']
        self.color_space_index = 0

        # --- Panorama State ---
        self.pano_image1 = None
        self.pano_image2 = None
        self.panorama_result = None

        # --- Calibration State ---
        self.calib_objpoints = []
        self.calib_imgpoints = []
        self.calib_feedback = ""
        self.calib_feedback_time = 0
        self.calib_chessboard_size = (9, 6)
        self.calib_square_size_mm = 25
        self.calib_target_images = 20

        # --- AR & Calibration Data ---
        self.mtx = None
        self.dist = None
        self.load_calibration_data()
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.trex_vertices, self.trex_faces = load_obj('trex_model.obj', scale_factor=1.0)

    def load_calibration_data(self):
        """Loads camera calibration data from file."""
        if os.path.exists('calibration.npz'):
            data = np.load('calibration.npz')
            self.mtx, self.dist = data['mtx'], data['dist']
            print("Camera calibration data loaded successfully.")
        else:
            print("WARNING: 'calibration.npz' not found. AR and Undistort modes will not be accurate.")
            # Use placeholder values
            w, h = self.cap.get(3), self.cap.get(4)
            self.mtx = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
            self.dist = np.zeros((1, 5), dtype=np.float32)

    def run(self):
        """Main application loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret: break

            processed_frame = self.process_frame(frame.copy())
            self.display_ui(processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if self.handle_key_press(key):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """Applies processing based on the current mode."""
        mode_function_map = {
            'NORMAL': lambda f: f,
            'CALIBRATE': self.process_calibration,
            'AR': self.render_ar,
            'COLOR_SPACE': self.apply_color_space,
            'HISTOGRAM': self.draw_histogram,
            'BRIGHTNESS_CONTRAST': self.adjust_brightness_contrast,
            'GAUSSIAN': self.apply_gaussian,
            'BILATERAL': self.apply_bilateral,
            'CANNY': lambda f: cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 100, 200),
            'HOUGH': self.detect_lines_hough,
            'TRANSFORM': self.apply_interactive_transform,
            'PANORAMA': self.create_panorama
        }
        return mode_function_map.get(self.mode, lambda f: f)(frame)

    # --- INDIVIDUAL MODE IMPLEMENTATIONS ---

    def render_ar(self, frame):
        if self.trex_vertices is None: return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.mtx, self.dist)
            for i in range(len(ids)):
                img_pts, _ = cv2.projectPoints(self.trex_vertices, rvecs[i], tvecs[i], self.mtx, self.dist)
                img_pts = np.int32(img_pts).reshape(-1, 2)
                for face in self.trex_faces:
                    points = np.array([img_pts[vertex_idx] for vertex_idx in face])
                    cv2.polylines(frame, [points], True, (0, 255, 100), 1, cv2.LINE_AA)
        return frame

    def apply_color_space(self, frame):
        current_space = self.color_space_options[self.color_space_index]
        if current_space == 'GRAY':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        elif current_space == 'HSV':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else: # RGB
            return frame

    def adjust_brightness_contrast(self, frame):
        return cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)

    def apply_gaussian(self, frame):
        ksize = self.gaussian_ksize if self.gaussian_ksize % 2 != 0 else self.gaussian_ksize + 1
        return cv2.GaussianBlur(frame, (ksize, ksize), 0)

    def apply_bilateral(self, frame):
        return cv2.bilateralFilter(frame, d=self.bilateral_d, sigmaColor=75, sigmaSpace=75)

    def detect_lines_hough(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            for line in lines: cv2.line(frame, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2)
        return frame

    def draw_histogram(self, frame):
        h, w, _ = frame.shape
        hist_h, hist_w = 256, 256
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
        for i, col in enumerate(colors):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            for j, val in enumerate(hist):
                cv2.line(hist_img, (j, hist_h), (j, hist_h - int(val)), col)
        frame[h-hist_h-10:h-10, 10:10+hist_w] = hist_img
        return frame

    def apply_interactive_transform(self, frame):
        rows, cols, _ = frame.shape
        center = (cols / 2, rows / 2)
        M = cv2.getRotationMatrix2D(center, self.transform_angle, self.transform_scale)
        M[0, 2] += self.transform_tx
        M[1, 2] += self.transform_ty
        return cv2.warpAffine(frame, M, (cols, rows))

    def create_panorama(self, frame):
        if self.panorama_result is not None:
            h, w, _ = frame.shape
            rh, rw, _ = self.panorama_result.shape
            scale = min(h/rh, w/rw, 1.0)
            return cv2.resize(self.panorama_result, (0, 0), fx=scale, fy=scale)
        if self.pano_image1 is not None and self.pano_image2 is None:
            h, w, _ = self.pano_image1.shape
            frame[0:h, 0:w] = cv2.addWeighted(frame[0:h, 0:w], 0.5, self.pano_image1, 0.5, 0)
        return frame

    def stitch_images(self):
        """STUDENT IMPLEMENTATION: This is where you write your panorama code."""
        print("Stitching images... (STUDENT: You need to implement this!)")
        # Placeholder: side-by-side concatenation
        self.panorama_result = np.concatenate((self.pano_image1, self.pano_image2), axis=1)

    def process_calibration(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.calib_chessboard_size, None)
        if ret:
            cv2.drawChessboardCorners(frame, self.calib_chessboard_size, corners, ret)
        return frame

    # --- UI AND KEY HANDLING ---

    def display_ui(self, frame):
        """Displays all UI elements on the frame."""
        draw_text(frame, f"Mode: {self.mode}", (20, 40), 1.0)
        draw_text(frame, "M: Cycle, Q: Quit", (20, 70))

        # Context-sensitive controls
        controls_text = ""
        if self.mode == 'BRIGHTNESS_CONTRAST':
            controls_text = f"Bright(W/S):{self.brightness} | Contrast(E/D):{self.contrast:.1f}"
        elif self.mode == 'GAUSSIAN':
            controls_text = f"Kernel Size (G/H): {self.gaussian_ksize if self.gaussian_ksize % 2 != 0 else self.gaussian_ksize + 1}"
        elif self.mode == 'BILATERAL':
            controls_text = f"Diameter (B/N): {self.bilateral_d}"
        elif self.mode == 'TRANSFORM':
            controls_text = "Scale(Z/X) | Rotate(</>) | Move(WASD)"
        elif self.mode == 'PANORAMA':
            controls_text = "C: Capture | R: Reset"
        elif self.mode == 'COLOR_SPACE':
            current_space = self.color_space_options[self.color_space_index]
            controls_text = f"Current Space: [{current_space}] | C: Cycle"
        elif self.mode == 'CALIBRATE':
            captured_text = f"Captured: {len(self.calib_imgpoints)}/{self.calib_target_images}"
            draw_text(frame, captured_text, (20, 130))
            controls_text = "SPACE: Capture | C: Calibrate | R: Reset"
            if time.time() < self.calib_feedback_time + 1.5:
                draw_text(frame, self.calib_feedback, (20, 160), color=(0, 255, 0))
        
        if controls_text:
            draw_text(frame, controls_text, (20, 100))

        cv2.imshow(self.window_name, frame)

    def handle_key_press(self, key):
        """Handles all keyboard inputs."""
        if key == ord('q'): return True
        elif key == ord('m'):
            self.mode_index = (self.mode_index + 1) % len(self.modes)
            self.mode = self.modes[self.mode_index]
            
        # Mode-specific key handlers
        handler = getattr(self, f"handle_{self.mode.lower()}_keys", None)
        if handler: handler(key)

        return False

    def handle_brightness_contrast_keys(self, key):
        if key == ord('w'): self.brightness += 5
        elif key == ord('s'): self.brightness -= 5
        elif key == ord('e'): self.contrast += 0.1
        elif key == ord('d'): self.contrast -= 0.1

    def handle_gaussian_keys(self, key):
        if key == ord('g'): self.gaussian_ksize += 2
        elif key == ord('h'): self.gaussian_ksize = max(1, self.gaussian_ksize - 2)

    def handle_bilateral_keys(self, key):
        if key == ord('b'): self.bilateral_d += 2
        elif key == ord('n'): self.bilateral_d = max(1, self.bilateral_d - 2)
        
    def handle_color_space_keys(self, key):
        if key == ord('c'):
            self.color_space_index = (self.color_space_index + 1) % len(self.color_space_options)

    def handle_transform_keys(self, key):
        if key == ord('z'): self.transform_scale += 0.05
        elif key == ord('x'): self.transform_scale -= 0.05
        elif key == ord(','): self.transform_angle -= 5 # <
        elif key == ord('.'): self.transform_angle += 5 # >
        elif key == ord('w'): self.transform_ty -= 10 # Up
        elif key == ord('s'): self.transform_ty += 10 # Down
        elif key == ord('a'): self.transform_tx -= 10 # Left
        elif key == ord('d'): self.transform_tx += 10 # Right

    def handle_panorama_keys(self, key):
        if key == ord('c'):
            _, frame = self.cap.read()
            if self.pano_image1 is None:
                self.pano_image1 = frame
                print("Panorama: Image 1 captured.")
            elif self.pano_image2 is None:
                self.pano_image2 = frame
                print("Panorama: Image 2 captured. Stitching...")
                self.stitch_images()
        elif key == ord('r'):
            self.pano_image1, self.pano_image2, self.panorama_result = None, None, None
            print("Panorama reset.")

    def handle_calibrate_keys(self, key):
        if key == ord(' '): # Spacebar
            self._capture_calibration_frame()
        elif key == ord('c'): # Calibrate
            self._run_calibration()
        elif key == ord('r'): # Reset
            self.calib_imgpoints, self.calib_objpoints = [], []
            self._set_calib_feedback("Calibration reset.")

    def _capture_calibration_frame(self):
        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.calib_chessboard_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((self.calib_chessboard_size[0] * self.calib_chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.calib_chessboard_size[0], 0:self.calib_chessboard_size[1]].T.reshape(-1, 2)
            objp *= self.calib_square_size_mm
            
            self.calib_objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.calib_imgpoints.append(corners2)
            self._set_calib_feedback(f"Captured! ({len(self.calib_imgpoints)})")
        else:
            self._set_calib_feedback("Corners not found!")

    def _run_calibration(self):
        if len(self.calib_imgpoints) < 10:
            self._set_calib_feedback("Need at least 10 images.")
            return
        print("Running calibration...")
        _, frame = self.cap.read()
        gray_shape = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).shape[::-1]
        ret, mtx, dist, _, _ = cv2.calibrateCamera(self.calib_objpoints, self.calib_imgpoints, gray_shape, None, None)
        if ret:
            np.savez('calibration.npz', mtx=mtx, dist=dist)
            self.mtx, self.dist = mtx, dist # Update live
            self._set_calib_feedback("Success! calibration.npz saved.")
            print("Calibration successful. Data saved and loaded.")
        else:
            self._set_calib_feedback("Calibration failed.")

    def _set_calib_feedback(self, msg):
        self.calib_feedback = msg
        self.calib_feedback_time = time.time()


if __name__ == '__main__':
    app = ComputerVisionApp()
    app.run()

