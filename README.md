# Real-Time Vision Caliper 🔍📐

Vision Caliper is a real-time computer vision application built with Python, OpenCV, and PyQt5. It utilizes deep learning-based object detection (YOLO) to dynamically measure the real-world physical dimensions (width, height, and area) of objects via a live video stream using a known reference calibration object.

![Application Showcase](https://via.placeholder.com/800x450.png?text=Vision+Caliper+Interface+Preview) ## ✨ Features

- **Real-Time Measurement:** Seamlessly calculates and displays widths, heights, and surface areas in both inches and centimeters concurrently.
- **Dynamic Calibration:** Calibrate settings on the fly using a reference object of known dimensions to establish pixel-to-metric ratios.
- **Flexible Configuration:** Easily select your tracking reference class from detected YOLO object classes and update physical dimensions directly through the GUI.
- **Multithreaded Architecture:** Keeps the PyQt5 interface highly responsive by handling heavy video rendering and CV processing on a dedicated worker thread.
- **Modern UI/UX:** Styled with a clean, grid-based card layout featuring persistent Theme Modes (Light/Dark preference memory saved across sessions).

---

## 🛠️ System Architecture

The application is split into modular layers to maintain high frames-per-second (FPS) processing.
1. gui.py(main app)
2. camera_measure.py(back-end calculation)

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You will also need your respective `camera_measure.py` script and its underlying YOLO weights configuration files in the root directory.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/vision-caliper.git](https://github.com/yourusername/vision-caliper.git)
   cd vision-caliper

**Install the dependencies:
pip install numpy opencv-python PyQt5
(Note: Ensure you install any additional deep-learning packages, such as ultralytics or torch, if required by your underlying camera_measure.py setup).

**Verify Configuration Files:
Ensure camera_measure.py is present in the workspace, along with your model weights. The app automatically creates config structures like app_theme.json on execution.

**Running the Application:
Execute the application launcher directly from your terminal:
python gui.py

📖 How To Use
1. Position Your Setup: Place a known reference object (e.g., a standard coffee cup or alignment card) into the camera's view frame alongside the target item you wish to measure.

2. Select Reference Target: Click on Select Ref. Object to define what item the algorithm should look for as its calibration standard.

3. Input Dimensions: Click Set Known Width to enter the precise physical width (in inches) of your chosen reference device.

4. Calibrate: Press the Calibrate button to calculate the operational baseline. The application will track targets and display calculations in real-time under the Measurements card.

5. Reset: Hit Recalibrate at any time to clear metrics and adjust your framing environment.

🔒 Security & Best Practices
If you plan to modify or contribute to this codebase, remember:

##Never commit local environment settings or local paths. Keep app_theme.json and any generated target tracker configs out of tracking.

##Do not hardcode custom API tokens or path strings if expanding to cloud-hosted ML inferences.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

all the essential parts are uploaded except for "yolov3.weihgts" because it is a large file. i wil add a drive link below where you can find this.
if u want to use a custom-trained model u can do that as well, just change the name of the model in "gui.py" and "camera_measure.py".
IF YOU WANT TO IMPROVE THE PROJECT PLEASE BE MY GUEST, AND PLEASE CONTACT ME AND LET ME SEE WHAT YOU IMPROVED!!! at "marufnrb@gmail.com"
The main app is the gui.py.
https://drive.google.com/file/d/1LmGeYxI4xhO0ywwwhIm8mQ9kXZVzc4lY/view?usp=sharing
