import sys
import cv2
import numpy as np
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLabel, QHBoxLayout, QMainWindow, QGroupBox,
                             QFrame, QDialog, QGridLayout, QDialogButtonBox,
                             QComboBox, QLineEdit, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from camera_measure import VisionCaliper, APP_CONFIG_FILE # Import APP_CONFIG_FILE

# New constant for theme file
APP_THEME_FILE = 'app_theme.json' 

# --- NEW DIALOGS ---

class ObjectSelectionDialog(QDialog):
    """Dialog to select the calibration object from a list of YOLO classes."""
    object_selected = pyqtSignal(str)

    def __init__(self, parent=None, class_list=None, current_label='cup'):
        super().__init__(parent)
        self.setWindowTitle("Select Calibration Object")
        self.setFixedSize(350, 200)
        self.setStyleSheet(parent.styleSheet())
        
        self.class_list = class_list if class_list else []
        self.current_label = current_label
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose the object type you will use for calibration:"))

        self.combo_box = QComboBox()
        self.combo_box.addItems(self.class_list)
        
        # Set the current label as default selection
        if self.current_label in self.class_list:
            self.combo_box.setCurrentText(self.current_label)
            
        layout.addWidget(self.combo_box)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def accept_selection(self):
        selected_object = self.combo_box.currentText()
        if selected_object:
            self.object_selected.emit(selected_object)
            self.accept()

class WidthInputWidget(QDialog):
    """Dialog to input the known physical width of the calibration object."""
    width_entered = pyqtSignal(str) # Emit as string to handle thread conversion
    
    def __init__(self, parent=None, current_width=3.62):
        super().__init__(parent)
        self.setWindowTitle("Set Known Width")
        self.setFixedSize(350, 150)
        self.setStyleSheet(parent.styleSheet())
        self.current_width = current_width
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout(self)
        layout.addWidget(QLabel("Known Width (Inches):"), 0, 0)

        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("e.g., 3.62")
        self.width_input.setText(f"{self.current_width:.2f}")
        layout.addWidget(self.width_input, 0, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_input)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 1, 0, 1, 2)
        
    def accept_input(self):
        text = self.width_input.text()
        try:
            width = float(text)
            if width <= 0:
                QMessageBox.warning(self, "Invalid Input", "The known width must be a positive number.")
                return
            self.width_entered.emit(text)
            self.accept()
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Please enter a valid number for the known width.")

# --- EXISTING DIALOG (THEME) ---

class SettingsDialog(QDialog):
    """
    A dialog for application settings, including theme selection.
    """
    theme_changed = pyqtSignal(str)

    def __init__(self, parent=None, current_theme='light'):
        super().__init__(parent)
        self.setWindowTitle("Theme Settings")
        self.setFixedSize(300, 150)
        self.current_theme = current_theme
        self.setStyleSheet(parent.styleSheet())
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.toggle_button = QPushButton("Toggle Dark/Light Mode")
        self.toggle_button.clicked.connect(self.toggle_theme)
        layout.addWidget(self.toggle_button)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.update_button_text()

    def update_button_text(self):
        if self.current_theme == 'dark':
            self.toggle_button.setText("Switch to Light Mode")
        else:
            self.toggle_button.setText("Switch to Dark Mode")

    def toggle_theme(self):
        if self.current_theme == 'light':
            new_theme = 'dark'
        else:
            new_theme = 'light'
            
        self.theme_changed.emit(new_theme)
        self.current_theme = new_theme
        self.update_button_text()

# --- MAIN APPLICATION ---

class VisionCaliperApp(QMainWindow):
    # Define styles as class attributes to reflect Google's design
    light_theme_style = """
        QMainWindow {
            background-color: #F1F3F4;
            color: #202124;
            font-family: 'Roboto', sans-serif, 'Segoe UI', Arial;
        }
        QLabel {
            font-size: 16px;
            color: #5F6368;
            padding: 5px;
        }
        QPushButton {
            background-color: #4285F4;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: bold;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #357ae8;
        }
        QGroupBox {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #DADCE0;
            border-radius: 12px;
            background-color: #FFFFFF;
            color: #202124;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QFrame#mainFrame {
            background-color: #F8F9FA;
            border-radius: 12px;
            margin: 20px;
        }
    """

    dark_theme_style = """
        QMainWindow {
            background-color: #202124;
            color: #E8EAED;
            font-family: 'Roboto', sans-serif, 'Segoe UI', Arial;
        }
        QLabel {
            font-size: 16px;
            color: #BDC1C6;
            padding: 5px;
        }
        QPushButton {
            background-color: #8AB4F8;
            color: #202124;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: bold;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #A6C5F6;
        }
        QGroupBox {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #5F6368;
            border-radius: 12px;
            background-color: #3C4043;
            color: #E8EAED;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QFrame#mainFrame {
            background-color: #292A2C;
            border-radius: 12px;
            margin: 20px;
        }
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Vision Caliper")
        self.setGeometry(100, 100, 1000, 700)
        
        # Load theme preference
        self.current_theme = self.load_theme_preference() 
        self.apply_styles(self.current_theme)

        # Initialize thread and connect signals
        self.caliper_thread = VisionCaliper()
        self.caliper_thread.frame_signal.connect(self.update_image)
        self.caliper_thread.status_signal.connect(self.update_status)
        self.caliper_thread.measurement_signal.connect(self.update_measurements)
        
        # Get initial calibration settings from the thread
        self.ref_object_label = self.caliper_thread.reference_object_label
        self.known_width = self.caliper_thread.known_width_inches
        
        self.init_ui()
        self.caliper_thread.start()

    # --- THEME PERSISTENCE METHODS ---
    def load_theme_preference(self):
        try:
            with open(APP_THEME_FILE, 'r') as f:
                data = json.load(f)
                theme = data.get('theme', 'light')
                if theme in ['light', 'dark']:
                    return theme
                return 'light'
        except (IOError, json.JSONDecodeError):
            return 'light'

    def save_theme_preference(self, theme):
        data = {'theme': theme}
        try:
            with open(APP_THEME_FILE, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Error saving theme preference: {e}")
            
    def apply_styles(self, theme_name):
        if theme_name == 'light':
            self.setStyleSheet(self.light_theme_style)
        else:
            self.setStyleSheet(self.dark_theme_style)

    # --- UI INITIALIZATION ---
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.main_frame = QFrame(self)
        self.main_frame.setObjectName("mainFrame")
        main_layout.addWidget(self.main_frame)
        
        inner_layout = QVBoxLayout(self.main_frame)
        inner_layout.setContentsMargins(20, 20, 20, 20)

        # Video feed label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 10px;")
        
        video_layout = QHBoxLayout()
        video_layout.addStretch()
        video_layout.addWidget(self.video_label)
        video_layout.addStretch()
        inner_layout.addLayout(video_layout)

        # Status and controls section in a horizontal layout
        controls_layout = QHBoxLayout()

        # Status GroupBox styled as a card
        status_box = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-size: 18px; color: #5F6368;")
        status_layout.addWidget(self.status_label)
        status_box.setLayout(status_layout)
        controls_layout.addWidget(status_box)

        # Buttons GroupBox styled as a card (Updated to include new buttons)
        buttons_box = QGroupBox("Controls")
        buttons_layout = QGridLayout()
        
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.caliper_thread.calibrate)
        self.recalibrate_button = QPushButton("Recalibrate")
        self.recalibrate_button.clicked.connect(self.caliper_thread.reset_calibration)
        
        # NEW BUTTONS
        self.select_object_button = QPushButton("Select Ref. Object")
        self.select_object_button.clicked.connect(self.show_object_selection_dialog)
        self.set_width_button = QPushButton("Set Known Width")
        self.set_width_button.clicked.connect(self.show_width_input_dialog)
        
        self.settings_button = QPushButton("Theme Settings")
        self.settings_button.clicked.connect(self.show_settings_dialog)

        # Layout the buttons (2x3 grid)
        buttons_layout.addWidget(self.calibrate_button, 0, 0)
        buttons_layout.addWidget(self.recalibrate_button, 0, 1)
        buttons_layout.addWidget(self.settings_button, 0, 2)
        buttons_layout.addWidget(self.select_object_button, 1, 0)
        buttons_layout.addWidget(self.set_width_button, 1, 1)

        buttons_box.setLayout(buttons_layout)
        controls_layout.addWidget(buttons_box)

        inner_layout.addLayout(controls_layout)

        # Measurements GroupBox
        measurements_box = QGroupBox("Measurements")
        measurements_layout = QGridLayout()
        
        self.label_width_in = QLabel("Width (in):")
        self.value_width_in = QLabel("---")
        self.label_width_cm = QLabel("Width (cm):")
        self.value_width_cm = QLabel("---")
        
        self.label_height_in = QLabel("Height (in):")
        self.value_height_in = QLabel("---")
        self.label_height_cm = QLabel("Height (cm):")
        self.value_height_cm = QLabel("---")

        self.label_area_in = QLabel("Area (sq.in):")
        self.value_area_in = QLabel("---")
        self.label_area_cm = QLabel("Area (sq.cm):")
        self.value_area_cm = QLabel("---")

        measurements_layout.addWidget(self.label_width_in, 0, 0)
        measurements_layout.addWidget(self.value_width_in, 0, 1)
        measurements_layout.addWidget(self.label_width_cm, 0, 2)
        measurements_layout.addWidget(self.value_width_cm, 0, 3)

        measurements_layout.addWidget(self.label_height_in, 1, 0)
        measurements_layout.addWidget(self.value_height_in, 1, 1)
        measurements_layout.addWidget(self.label_height_cm, 1, 2)
        measurements_layout.addWidget(self.value_height_cm, 1, 3)

        measurements_layout.addWidget(self.label_area_in, 2, 0)
        measurements_layout.addWidget(self.value_area_in, 2, 1)
        measurements_layout.addWidget(self.label_area_cm, 2, 2)
        measurements_layout.addWidget(self.value_area_cm, 2, 3)

        measurements_box.setLayout(measurements_layout)
        inner_layout.addWidget(measurements_box)

    # --- DIALOG HANDLERS ---
    def show_object_selection_dialog(self):
        # Pass the current list of detected classes and the currently selected object
        class_list = self.caliper_thread.get_object_classes()
        current_label = self.caliper_thread.reference_object_label
        
        dialog = ObjectSelectionDialog(self, class_list, current_label)
        dialog.object_selected.connect(self.set_reference_object)
        dialog.exec_()
        
    def set_reference_object(self, label):
        self.ref_object_label = label
        # Signal the worker thread to update its configuration and reset calibration
        self.caliper_thread.set_reference_object_label(label)
        
    def show_width_input_dialog(self):
        current_width = self.caliper_thread.known_width_inches
        
        dialog = WidthInputWidget(self, current_width)
        dialog.width_entered.connect(self.set_known_width)
        dialog.exec_()
        
    def set_known_width(self, width_str):
        try:
            width = float(width_str)
            self.known_width = width
            # Signal the worker thread to update its configuration and reset calibration
            self.caliper_thread.set_known_width(width)
        except ValueError:
            # Should be handled by the dialog, but good practice to keep
            pass 

    def show_settings_dialog(self):
        dialog = SettingsDialog(self, self.current_theme)
        dialog.theme_changed.connect(self.set_theme)
        dialog.exec_()

    def set_theme(self, theme_name):
        if theme_name == 'light':
            self.setStyleSheet(self.light_theme_style)
            self.status_label.setStyleSheet("font-size: 18px; color: #5F6368;")
        else:
            self.setStyleSheet(self.dark_theme_style)
            self.status_label.setStyleSheet("font-size: 18px; color: #E0E0E0;")
        
        self.current_theme = theme_name
        self.save_theme_preference(theme_name)

    # --- DATA UPDATES ---
    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_status(self, message):
        self.status_label.setText(message)

    def update_measurements(self, measurements):
        if measurements.get('label') == "No objects detected.":
            self.value_width_in.setText("---")
            self.value_width_cm.setText("---")
            self.value_height_in.setText("---")
            self.value_height_cm.setText("---")
            self.value_area_in.setText("---")
            self.value_area_cm.setText("---")
            self.status_label.setText("No objects detected.")
            return

        measurement_keys = [
            ('width_inches', self.value_width_in),
            ('width_cm', self.value_width_cm),
            ('height_inches', self.value_height_in),
            ('height_cm', self.value_height_cm),
            ('area_sq_inches', self.value_area_in),
            ('area_sq_cm', self.value_area_cm)
        ]

        for key, label_widget in measurement_keys:
            try:
                value = measurements[key]
                if value is None:
                    label_widget.setText("N/A")
                else:
                    label_widget.setText(f"{float(value):.2f}")
            except (KeyError, ValueError, TypeError):
                label_widget.setText("N/A")

        self.status_label.setText(f"Measuring: {measurements.get('label', 'Unknown Object')}")
        
    def closeEvent(self, event):
        self.caliper_thread.stop()
        self.caliper_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionCaliperApp()
    window.show()
    sys.exit(app.exec_())