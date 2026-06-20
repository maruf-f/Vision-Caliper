import cv2
import numpy as np
import json
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

# Configuration and Calibration file names
APP_CONFIG_FILE = 'app_config.json'
CALIBRATION_FILE = 'calibration.json'

# FALLBACK CLASSES (COCO Dataset - 80 classes)
# Used if 'class_names.txt' is not found, ensuring the UI has a list to work with.
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class VisionCaliper(QThread):
    """
    A worker thread to handle all vision processing and measurements,
    preventing the GUI from freezing.
    """
    # Signals for communication with the GUI
    frame_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)
    measurement_signal = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.reference_pixel_width = None
        self.is_running = True
        self.cap = None
        self.is_calibrating = False
        self.measurement_history = []
        self.HISTORY_SIZE = 10

        # Configuration attributes initialized with defaults, then updated by config file
        self.reference_object_label = 'cup' 
        self.known_width_inches = 3.62     

        self.load_config()

        self.CLASSES = []
        try:
            with open('class_names.txt', 'r') as f:
                self.CLASSES = f.read().splitlines()
        except FileNotFoundError:
            self.status_signal.emit("Error: 'class_names.txt' not found. Using COCO fallback classes.")
            self.CLASSES = COCO_CLASSES 
        
        # Ensure the selected label is valid in the loaded classes
        if self.reference_object_label not in self.CLASSES and self.CLASSES:
            self.reference_object_label = self.CLASSES[0]
            self.save_config() # Save the corrected label

        self.prototxt_path = 'yolov3.cfg'
        self.model_path = 'yolov3.weights'
        try:
            self.net = cv2.dnn.readNet(self.prototxt_path, self.model_path)
        except cv2.error as e:
            self.status_signal.emit(f"Error loading model files: {e}")
            self.is_running = False

        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def load_config(self):
        """Loads calibration configuration (ref object and width) from file."""
        try:
            with open(APP_CONFIG_FILE, 'r') as f:
                data = json.load(f)
                self.reference_object_label = data.get('reference_object_label', self.reference_object_label)
                self.known_width_inches = data.get('known_width_inches', self.known_width_inches)
        except (IOError, json.JSONDecodeError):
            pass 

    def save_config(self):
        """Saves calibration configuration (ref object and width) to file."""
        data = {
            'reference_object_label': self.reference_object_label,
            'known_width_inches': self.known_width_inches
        }
        try:
            with open(APP_CONFIG_FILE, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Error saving application config file: {e}")

    # New methods for GUI to update configuration
    def set_reference_object_label(self, label):
        """Sets the new reference object label and saves config."""
        self.reference_object_label = label
        self.save_config()
        # Reset calibration to force a recalibration
        self.reset_calibration()
        self.status_signal.emit(f"Ref Object set to: {label}. Please Recalibrate.")

    def set_known_width(self, width):
        """Sets the new known width and saves config."""
        try:
            width = float(width)
            if width > 0:
                self.known_width_inches = width
                self.save_config()
                # Reset calibration to force a recalibration
                self.reset_calibration()
                self.status_signal.emit(f"Known Width set to: {width:.2f} inches. Please Recalibrate.")
            else:
                self.status_signal.emit("Width must be positive.")
        except ValueError:
            self.status_signal.emit("Invalid width value.")
    
    def get_object_classes(self):
        """Helper to expose the list of detected object classes to the GUI."""
        return self.CLASSES

    # --- Core Vision Logic (Updated to use instance attributes) ---
    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_signal.emit("Error: Cannot open webcam.")
            self.is_running = False
            return
        
        self.reference_pixel_width = self.load_calibration()
        if self.reference_pixel_width is None:
            self.is_calibrating = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_signal.emit("Failed to grab a frame.")
                break
            
            h, w = frame.shape[:2]
            
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # Increase confidence threshold for more reliable detections
                    if confidence > 0.5:
                        if np.isinf(detection[2]) or np.isnan(detection[2]) or np.isinf(detection[3]) or np.isnan(detection[3]):
                            continue
                            
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        width = int(detection[2] * w)
                        height = int(detection[3] * h)
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
                        boxes.append([x, y, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            rects_to_measure = self.find_and_measure_rectangle(frame, boxes, class_ids, confidences, indexes, self.reference_object_label)

            if self.is_calibrating:
                # Use instance attributes
                self.status_signal.emit(f"Place a {self.reference_object_label} in frame (Known Width: {self.known_width_inches:.2f} in) & click 'Calibrate'")
            else:
                self.status_signal.emit("Calibrated! Place object to measure or click 'Recalibrate'")
                
                if not rects_to_measure:
                    self.measurement_signal.emit({"label": "No objects detected."})
                    self.measurement_history = []
                else:
                    for rect in rects_to_measure:
                        box_points = cv2.boxPoints(rect['rotated_rect'])
                        box_points = np.intp(box_points)
                        cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)
                        
                        width_pixels = min(rect['rotated_dimensions'])
                        height_pixels = max(rect['rotated_dimensions'])
                        
                        width_inches, width_cm = self.pixel_to_units(width_pixels, self.reference_pixel_width)
                        height_inches, height_cm = self.pixel_to_units(height_pixels, self.reference_pixel_width)
                        
                        if width_inches and height_inches:
                            area_sq_inches = width_inches * height_inches
                            area_sq_cm = width_cm * height_cm
                            
                            # Add current measurement to history
                            self.measurement_history.append({
                                'label': rect['label'],
                                'width_inches': width_inches,
                                'width_cm': width_cm,
                                'height_inches': height_inches,
                                'height_cm': height_cm,
                                'area_sq_inches': area_sq_inches,
                                'area_sq_cm': area_sq_cm
                            })
                            
                            # Trim history to the desired size
                            if len(self.measurement_history) > self.HISTORY_SIZE:
                                self.measurement_history.pop(0)
                            
                            # Average the measurements
                            if self.measurement_history:
                                avg_width_inches = np.mean([m['width_inches'] for m in self.measurement_history])
                                avg_width_cm = np.mean([m['width_cm'] for m in self.measurement_history])
                                avg_height_inches = np.mean([m['height_inches'] for m in self.measurement_history])
                                avg_height_cm = np.mean([m['height_cm'] for m in self.measurement_history])
                                avg_area_sq_inches = np.mean([m['area_sq_inches'] for m in self.measurement_history])
                                avg_area_sq_cm = np.mean([m['area_sq_cm'] for m in self.measurement_history])

                                # Emit the averaged results
                                self.measurement_signal.emit({
                                    'label': rect['label'],
                                    'width_inches': avg_width_inches,
                                    'width_cm': avg_width_cm,
                                    'height_inches': avg_height_inches,
                                    'height_cm': avg_height_cm,
                                    'area_sq_inches': avg_area_sq_inches,
                                    'area_sq_cm': avg_area_sq_cm
                                })

            self.frame_signal.emit(frame)
            QThread.msleep(10)

        self.cap.release()
        self.status_signal.emit("Camera feed stopped.")

    def calibrate(self):
        self.status_signal.emit("Attempting to calibrate...")
        if self.known_width_inches <= 0:
            self.status_signal.emit("Error: Known Width is not set or is zero/negative. Set it before calibrating.")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_signal.emit("Could not grab frame for calibration.")
            return

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Increase confidence threshold for more reliable detections
                if confidence > 0.5:
                    if np.isinf(detection[2]) or np.isnan(detection[2]) or np.isinf(detection[3]) or np.isnan(detection[3]):
                        continue
                        
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # Use instance attribute
        rects = self.find_and_measure_rectangle(frame, boxes, class_ids, confidences, indexes, self.reference_object_label)

        reference_object_found = False
        for rect in rects:
            if rect['label'] == self.reference_object_label:
                # Use average of dimensions for more robust calibration
                rotated_width, rotated_height = rect['rotated_dimensions']
                self.reference_pixel_width = (rotated_width + rotated_height) / 2
                
                self.save_calibration(self.reference_pixel_width)
                self.is_calibrating = False
                self.status_signal.emit("Calibration successful!")
                reference_object_found = True
                break
        
        if not reference_object_found:
            self.status_signal.emit(f"Could not find '{self.reference_object_label}'. Please ensure it's in the frame and retry.")

    def reset_calibration(self):
        self.reference_pixel_width = None
        self.is_calibrating = True
        self.measurement_history = []
        self.status_signal.emit("Calibration reset. Please calibrate again.")
        
    def stop(self):
        self.is_running = False
        
    def pixel_to_units(self, pixel_distance, reference_pixel_width):
        if reference_pixel_width is None or reference_pixel_width == 0:
            return None, None
        
        # Use instance attribute
        inches = self.known_width_inches * (pixel_distance / reference_pixel_width)
        cm = inches * 2.54
        return inches, cm

    def save_calibration(self, reference_pixel_width_value):
        data = {'reference_pixel_width': reference_pixel_width_value}
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(data, f)
            print(f"Calibration saved to {CALIBRATION_FILE}")
        except IOError as e:
            print(f"Error saving calibration file: {e}")

    def load_calibration(self):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                data = json.load(f)
                return data.get('reference_pixel_width')
        except (IOError, json.JSONDecodeError):
            return None

    def find_and_measure_rectangle(self, frame, boxes, class_ids, confidences, indexes, reference_label):
        rect_data = []
        is_calibrating = (self.reference_pixel_width is None)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w_obj, h_obj = boxes[i]
                label = str(self.CLASSES[class_ids[i]])
                
                if label == reference_label or not is_calibrating:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x, y), (x + w_obj, y + h_obj), 255, -1)
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        rotated_rect = cv2.minAreaRect(largest_contour)
                        
                        rotated_width, rotated_height = rotated_rect[1]
                        
                        rect_data.append({
                            'label': label,
                            'rotated_rect': rotated_rect,
                            'rotated_dimensions': (rotated_width, rotated_height)
                        })
        
        return rect_data