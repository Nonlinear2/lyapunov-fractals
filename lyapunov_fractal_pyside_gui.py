import sys
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, QDoubleSpinBox,
                               QPushButton, QColorDialog, QMessageBox, QScrollArea,
                               QGroupBox, QSpinBox, QFileDialog)

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QRegularExpressionValidator
from numba.cuda import get_current_device
from lyapunov_core import ComputeFractals
from utils import valid_hex_string
from dataclasses import dataclass

def make_spinbox(value, singleStep, minimum, maximum):
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(singleStep)
    box.setValue(value)
    return box

def make_dspinbox(value, singleStep, minimum, maximum, decimals):
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(singleStep)
    box.setValue(value)
    return box

@dataclass
class Config:
    MIN_FRACTAL_REGION_SIZE = (600, 600)
    APP_MIN_SIZE = (1100, 800)
    ZOOM_FACTOR = 0.98
    ZOOM_TIMER_INTERVAL = 100
    REGENERATION_DELAY = 300
    DEFAULT_COLORS = ["#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]

# worker thread to avoid blocking the GUI
class FractalWorker(QThread):
    finished = Signal(np.ndarray)
    
    def __init__(self, fractal_params, z_value):
        super().__init__()
        self.fractal_params = fractal_params
        self.z_value = z_value

    def run(self):
        fractal = ComputeFractals(**self.fractal_params)
        img_array = fractal.compute_fractal(self.z_value)
        self.finished.emit(img_array)

# handle mouse clicks for zooming
class FractalLabel(QLabel):
    zoom = Signal(float, float, float)
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(*Config.MIN_FRACTAL_REGION_SIZE)
        self.setStyleSheet("background-color: black;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Start real time generation first, then hold left mouse to zoom in, right mouse to zoom out")
        
        self.is_zooming = False
        self.zoom_type = None
        self.last_mouse_pos = None

        self.zoom_timer = QTimer()
        self.zoom_timer.timeout.connect(self.continuous_zoom)
        self.zoom_timer.setInterval(Config.ZOOM_TIMER_INTERVAL)

    def mousePressEvent(self, event):
        if self.pixmap():
            # Calculate click position as ratio of fractal canvas
            x_ratio = event.position().x() / self.width()
            y_ratio = event.position().y() / self.height()
            self.last_mouse_pos = (x_ratio, y_ratio)
            
            if event.button() in [Qt.LeftButton, Qt.RightButton]:
                self.is_zooming = True
                if (event.button() == Qt.LeftButton):
                    # zoom in
                    self.zoom_proportion = Config.ZOOM_FACTOR
                else:
                    # zoom out
                    self.zoom_proportion = 1/Config.ZOOM_FACTOR

                self.zoom.emit(x_ratio, y_ratio, self.zoom_proportion)
                self.zoom_timer.start()

    def mouseReleaseEvent(self, event):
        if self.is_zooming:
            self.is_zooming = False
            self.zoom_timer.stop()

    def mouseMoveEvent(self, event):
        if self.is_zooming and self.pixmap():
            # Update mouse position for continuous zooming
            x_ratio = event.position().x() / self.width()
            y_ratio = event.position().y() / self.height()
            self.last_mouse_pos = (x_ratio, y_ratio)

    def continuous_zoom(self):
        if self.is_zooming and self.last_mouse_pos:
            self.zoom.emit(*self.last_mouse_pos, self.zoom_proportion)


class FractalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lyapunov Fractal Generator")
        self.setMinimumSize(*Config.APP_MIN_SIZE)

        # Initialize variables
        self.current_high_res_image = None
        self.worker = None
        self.current_fractal = None  # Store current fractal instance for real-time mode
        self.real_time_mode = False  # Track current mode

        self.z = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        self.low_res_btn_text = "Start Real-Time Generation"
        self.low_res_btn_style = "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }"     
        self.high_res_btn_text = "Generate High-Res Image"
        self.high_res_btn_style = "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }"
        self.save_btn_text = "Save High-Res Image"
        self.save_btn_style = "QPushButton { background-color: #FF9800; color: white; padding: 8px; }"

        gpu = get_current_device()
        self.max_image_size = (gpu.MAX_GRID_DIM_X * gpu.MAX_THREADS_PER_BLOCK)**0.5

        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        self.create_top_fields(left_layout)
        
        self.create_resolution_settings(left_layout)
        
        self.create_color_pickers(left_layout)
        
        self.create_buttons(left_layout)
        
        # push everything to top
        left_layout.addStretch()

        # Right panel for fractal display
        self.fractal_label = FractalLabel()
        self.fractal_label.zoom.connect(self.on_zoom)
        
        # Create scroll area for the fractal display
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.fractal_label)
        scroll_area.setWidgetResizable(True)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(scroll_area, 1)

    def create_top_fields(self, layout):
        # Create grid layout for input fields
        grid_layout = QGridLayout()
        
        coords_fields = {
            'z': ('Z', 2.86),
            'x_min': ('X Min', 1.009),
            'x_max': ('X Max', 1.244),
            'y_min': ('Y Min', 3.662),
            'y_max': ('Y Max', 3.898),
        }

        self.pattern = QLineEdit("yyxxyyyyyzz")
        self.pattern.textChanged.connect(self.on_parameter_changed)
        self.pattern.setValidator(QRegularExpressionValidator("^[xyzXYZ]{0,50}$"))
        grid_layout.addWidget(QLabel("Pattern"), 0, 0)
        grid_layout.addWidget(self.pattern, 0, 1)

        for i, (field_name, (label_text, default_value)) in enumerate(coords_fields.items(), 1):
            spin_box = make_dspinbox(value=default_value, singleStep=0.1,  minimum=0, maximum=4, decimals=5)
            spin_box.valueChanged.connect(self.on_parameter_changed)

            setattr(self, field_name, spin_box)

            grid_layout.addWidget(QLabel(label_text), i, 0)
            grid_layout.addWidget(spin_box, i, 1)

        self.color_res = make_spinbox(value=1900, singleStep=50, minimum=50, maximum=10_000)
        self.color_res.valueChanged.connect(self.on_parameter_changed)

        grid_layout.addWidget(QLabel("Color Resolution"), 6, 0)
        grid_layout.addWidget(self.color_res, 6, 1)

        layout.addLayout(grid_layout)

    def create_resolution_settings(self, layout):
        res_group = QGroupBox("Resolution Settings")
        res_layout = QGridLayout(res_group)
        
        realtime_label = QLabel("Real-Time Mode:")
        realtime_label.setStyleSheet("font-weight: bold;")
        res_layout.addWidget(realtime_label, 0, 0, 1, 2)
        
        res_layout.addWidget(QLabel("Size:"), 1, 0)
        self.low_res_size = make_spinbox(value=300, singleStep=50, minimum=100, maximum=1500)
        self.low_res_size.setKeyboardTracking(False)
        self.low_res_size.valueChanged.connect(self.on_parameter_changed)
        res_layout.addWidget(self.low_res_size, 1, 1)

        res_layout.addWidget(QLabel("Iterations:"), 2, 0)
        self.low_res_iter = make_spinbox(value=200, singleStep=50, minimum=50, maximum=5000)
        self.low_res_iter.setKeyboardTracking(False)
        self.low_res_iter.valueChanged.connect(self.on_parameter_changed)
        res_layout.addWidget(self.low_res_iter, 2, 1)
        
        high_res_label = QLabel("High Resolution Mode:")
        high_res_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        res_layout.addWidget(high_res_label, 3, 0, 1, 2)
        
        res_layout.addWidget(QLabel("Size:"), 4, 0)
        self.high_res_size = make_spinbox(value=min(1200, self.max_image_size), singleStep=50,
                                          minimum=500, maximum=self.max_image_size)
        self.high_res_size.setKeyboardTracking(False)
        res_layout.addWidget(self.high_res_size, 4, 1)
        
        res_layout.addWidget(QLabel("Iterations:"), 5, 0)
        self.high_res_iter = make_spinbox(value=2000, singleStep=50, minimum=500, maximum=50_000)
        self.high_res_iter.setKeyboardTracking(False)
        res_layout.addWidget(self.high_res_iter, 5, 1)
        
        layout.addWidget(res_group)
    
    def create_color_pickers(self, layout):
        # Colors section
        color_group = QGroupBox("Colors")
        color_layout = QVBoxLayout(color_group)
        
        # Create 6 color pickers
        self.color_inputs = []
        self.color_buttons = []
        
        for i in range(6):
            color_row_layout = QHBoxLayout()
            
            color_input = QLineEdit(Config.DEFAULT_COLORS[i])
            color_input.setMaximumWidth(80)
            color_input.setValidator(QRegularExpressionValidator("^#[0-9A-Fa-f]{6}$"))
            color_input.textChanged.connect(self.on_parameter_changed)
            
            color_button = QPushButton(f"Color {i+1}")
            color_button.clicked.connect(lambda checked, idx=i: self.pick_color(idx))
            
            color_row_layout.addWidget(color_input)
            color_row_layout.addWidget(color_button)
            
            self.color_inputs.append(color_input)
            self.color_buttons.append(color_button)
            
            color_layout.addLayout(color_row_layout)
        
        layout.addWidget(color_group)
    
    def create_buttons(self, layout):
        # Real-time generation button
        self.low_res_btn = QPushButton(self.low_res_btn_text)
        self.low_res_btn.clicked.connect(self.toggle_real_time_mode)
        self.low_res_btn.setStyleSheet(self.low_res_btn_style)
        layout.addWidget(self.low_res_btn)
        
        # Generate high-res button
        self.high_res_btn = QPushButton(self.high_res_btn_text)
        self.high_res_btn.clicked.connect(lambda _: self.start_image_gen(is_low_res=False))
        self.high_res_btn.setStyleSheet(self.high_res_btn_style)
        layout.addWidget(self.high_res_btn)
        
        # Save button (for high-res images only)
        self.save_btn = QPushButton(self.save_btn_text)
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setStyleSheet(self.save_btn_style)
        layout.addWidget(self.save_btn)

    # Correct bounds to ensure min < max
    def sanitize_inputs(self, changed_field):
        try:
            x_min = self.x_min.value()
            x_max = self.x_max.value()
            y_min = self.y_min.value()
            y_max = self.y_max.value()

            if changed_field == self.x_min and x_min >= x_max:
                self.x_min.setValue(x_max)

            elif changed_field == self.x_max and x_max <= x_min:
                self.x_max.setValue(x_min)

            if changed_field == self.y_min and y_min >= y_max:
                self.y_min.setValue(y_max)

            elif changed_field == self.y_max and y_max <= y_min:
                self.y_max.setValue(y_min)

        except ValueError:
            # Ignore invalid number formats (let user continue typing)
            pass

    # Handle parameter changes by regenerating the fractal if in real-time mode
    def on_parameter_changed(self):
        # determined the changed field by the sender
        self.sanitize_inputs(self.sender())

        if self.real_time_mode and self.current_fractal:
            # Debounce the regeneration using a timer
            if not hasattr(self, 'regeneration_timer'):
                self.regeneration_timer = QTimer()
                self.regeneration_timer.setSingleShot(True)
                self.regeneration_timer.timeout.connect(self.regenerate_realtime_fractal)
            
            # Reset the timer - this debounces rapid changes
            self.regeneration_timer.stop()
            self.regeneration_timer.start(Config.REGENERATION_DELAY)
    
    def regenerate_realtime_fractal(self):
        if not self.real_time_mode:
            return

        # Get new parameters and recreate fractal instance
        params, z_value = self.get_fractal_params(is_low_res=True)
        self.current_fractal = ComputeFractals(**params)
        
        # Generate and display new image
        img_array = self.current_fractal.compute_fractal(z_value)
        img = Image.fromarray(np.swapaxes(img_array.astype(np.uint8), 0, 1))
        self.display_image(img)

    def pick_color(self, index):
        color = QColorDialog.getColor()
        
        if color.isValid():
            hex_color = color.name()
            self.color_inputs[index].setText(hex_color)
            # The textChanged signal will automatically trigger regeneration
    
    # extract parameters from GUI fields
    def get_fractal_params(self, is_low_res=True):
        colors = []
        for color_input in self.color_inputs:
            color = color_input.text().strip().lower()
            if valid_hex_string(color):
                colors.append(color)

        if not colors:
            colors = ["#000000"]  # Default color if none provided
        
        # Choose resolution settings based on mode
        if is_low_res:
            size = self.low_res_size.value()
            iterations = self.low_res_iter.value()
        else:
            size = self.high_res_size.value()
            iterations = self.high_res_iter.value()
        
        params = {
            'pattern': self.pattern.text().strip(),
            'x_min': self.x_min.value(),
            'x_max': self.x_max.value(),
            'y_min': self.y_min.value(),
            'y_max': self.y_max.value(),
            'size': size,
            'color_resolution': self.color_res.value(),
            'num_iter': iterations,
            'colors': colors
        }

        return params, self.z.value()

    def toggle_real_time_mode(self):
        if self.real_time_mode:
            self.real_time_mode = False
            self.current_fractal = None

            # Stop any pending regeneration timer
            if hasattr(self, 'regeneration_timer'):
                self.regeneration_timer.stop()
            
            # Clear the display
            self.fractal_label.clear()
            self.fractal_label.setText(
                "Start Real-Time Generation first, then left-click to zoom in, right-click to zoom out"
            )
            
            # Update button text
            self.low_res_btn.setText(self.low_res_btn_text)
            self.low_res_btn.setStyleSheet(self.low_res_btn_style)
    
        else:
            self.real_time_mode = True

            # Update button to show exit option
            self.low_res_btn.setText("Exit Real-Time Mode")
            self.low_res_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
            self.low_res_btn.setEnabled(True)

            self.start_image_gen(is_low_res=True)

    def start_image_gen(self, is_low_res):
        params, z_value = self.get_fractal_params(is_low_res=is_low_res)
        
        if is_low_res:
            # Update button state
            self.low_res_btn.setText("Generating...")
            self.low_res_btn.setEnabled(False)

        elif self.real_time_mode:
            self.real_time_mode = False
            self.current_fractal = None
            self.low_res_btn.setText(self.low_res_btn_text)
            self.low_res_btn.setStyleSheet(self.low_res_btn_style)

        # Start worker thread
        self.worker = FractalWorker(params, z_value)
        self.worker.finished.connect(lambda img: self.on_image_generated(img, is_low_res))
        self.worker.start()

    def on_image_generated(self, img, is_low_res):

        if is_low_res:
            # Store fractal instance for real-time zooming
            params, z_value = self.get_fractal_params(is_low_res=True)
            self.current_fractal = ComputeFractals(**params)

        else:
            self.current_high_res_image = img

            # Re-enable generate button
            self.high_res_btn.setText(self.high_res_btn_text)
            self.high_res_btn.setEnabled(True)
        self.display_image(img)

    def display_image(self, img):
        
        img = np.ascontiguousarray(np.swapaxes(img.astype(np.uint8), 0, 1))
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(qimg)
        self.fractal_label.setPixmap(pixmap)
        self.fractal_label.resize(pixmap.size())

    def on_zoom(self, x_ratio, y_ratio, zoom_proportion):
        if not self.current_fractal or not self.real_time_mode:
            return

        mouse_pos = self.center_zoom(x_ratio, y_ratio, 0.1)
        mouse_coords = self.get_mouse_coords_in_region(mouse_pos[0], mouse_pos[1])

        new_bounds = self.zoom_to(mouse_coords, zoom_proportion)

        # Update input fields
        self.x_min.setValue(new_bounds[0])
        self.x_max.setValue(new_bounds[1])
        self.y_min.setValue(new_bounds[2])
        self.y_max.setValue(new_bounds[3])

        # Update fractal region and generate new image quickly
        self.current_fractal.set_region(*new_bounds)
        img = self.current_fractal.compute_fractal(self.z.value())

        # Display immediately
        self.display_image(img)

    # convert mouse position ratios to fractal coordinates
    def get_mouse_coords_in_region(self, pos_x_ratio, pos_y_ratio):
        x_min = self.x_min.value()
        x_max = self.x_max.value()
        y_min = self.y_min.value()
        y_max = self.y_max.value()

        pos_x = x_min + (x_max - x_min) * pos_x_ratio
        # we flip the y axis because it is pointing downwards
        pos_y = y_min + (y_max - y_min) * (1 - pos_y_ratio)
        
        return pos_x, pos_y
    
    def center_zoom(self, pos_x_ratio, pos_y_ratio, coef):
        size = self.low_res_size.value()
        new_pos_x = coef * (pos_x_ratio * size - size/2) + size/2
        new_pos_y = coef * (pos_y_ratio * size - size/2) + size/2
        return (new_pos_x / size, new_pos_y / size)
    
    # calculate new region bounds after zoom
    def zoom_to(self, pos, zoom_proportion):
        x_min = self.x_min.value()
        x_max = self.x_max.value()
        y_min = self.y_min.value()
        y_max = self.y_max.value()

        new_x_min = pos[0] - (zoom_proportion * (x_max - x_min)) / 2
        new_y_min = pos[1] - (zoom_proportion * (y_max - y_min)) / 2
        new_x_max = new_x_min + zoom_proportion * (x_max - x_min)
        new_y_max = new_y_min + zoom_proportion * (y_max - y_min)
        
        # boundary checks
        if (new_x_max - new_x_min) > 4:
            new_x_min = 0.01
            new_x_max = 4
        
        if (new_y_max - new_y_min) > 4:
            new_y_min = 0.01
            new_y_max = 4
        
        if new_x_min < 0:
            new_x_max -= new_x_min - 0.01
            new_x_min = 0.01
        
        if new_y_min < 0:
            new_y_max -= new_y_min - 0.01
            new_y_min = 0.01
        
        if new_x_max > 4:
            new_x_min -= new_x_max - 4
            new_x_max = 4
        
        if new_y_max > 4:
            new_y_min -= new_y_max - 4
            new_y_max = 4
        
        return new_x_min, new_x_max, new_y_min, new_y_max

    # save the current high resolution image
    def save_image(self):
        if self.current_high_res_image:
            try:                
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save High-Res Fractal Image",
                    "fractal_highres.png",
                    "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
                )

                if file_path:
                    img = Image.fromarray(np.swapaxes(self.current_high_res_image.astype(np.uint8), 0, 1))
                    img.save(file_path)
                    QMessageBox.information(self, "Success", f"High-resolution image saved to {file_path}")

            except Exception as e:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Error")
                msg_box.setText(f"Error saving image: {e}")
                msg_box.exec()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)

    app.setStyle('Fusion')

    window = FractalApp()
    window.show()

    app.exec()


if __name__ == "__main__":
    main()