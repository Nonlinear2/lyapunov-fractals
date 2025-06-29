import sys
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, QDoubleSpinBox,
                               QPushButton, QColorDialog, QMessageBox, QScrollArea,
                               QGroupBox, QSpinBox, QFileDialog)

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QRegularExpressionValidator, QColor, QIcon
from numba.cuda import get_current_device
from lyapunov_core import ComputeFractals
from utils import valid_hex_string
from dataclasses import dataclass
from utils import ColorPalettes

# enum for GUI state
IDLE = 1
REAL_TIME = 2
GENERATING = 3
HIGH_RES = 4


def make_integer_spinbox(value, singleStep, minimum, maximum):
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(singleStep)
    box.setValue(value)
    return box

def make_float_spinbox(value, singleStep, minimum, maximum, decimals):
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(singleStep)
    box.setValue(value)
    return box

def make_button(text, style, callback, focus_policy=Qt.NoFocus):
    button = QPushButton(text)
    button.clicked.connect(callback)
    button.setStyleSheet(style)
    button.setFocusPolicy(focus_policy)
    return button

@dataclass
class Config:
    MIN_FRACTAL_REGION_SIZE = (600, 600)
    APP_MIN_SIZE = (1025, 800)
    LEFT_PANEL_MAX_WIDTH = 350
    ZOOM_FACTOR = 0.98
    TIMER_INTERVAL = 100
    REGENERATION_DELAY = 300
    DEFAULT_COLORS = ColorPalettes.red_orange_yellow
    DEFAULT_REAL_TIME_SIZE = 500
    DEFAULT_REAL_TIME_ITER = 500
    DEFAULT_HIGH_RES_SIZE = 2000
    DEFAULT_HIGH_RES_ITER = 4000

# worker thread to avoid blocking the GUI
class FractalWorker(QThread):
    finished = Signal(np.ndarray)
    stage_updated = Signal(str, int)

    def __init__(self):
        super().__init__()
        self.fractal_computer = ComputeFractals()
        self.total_stage_number = self.fractal_computer.total_stage_number
        self.fractal_computer.set_progress_callback(self.stage_updated.emit)

    def set_parameters(self, fractal_params):
        self.fractal_computer.set_parameters(**fractal_params)

    def run(self):
        self.fractal_computer.compute_fractal()
        img_array = self.fractal_computer.apply_gradient()
        self.finished.emit(img_array)

    @property
    def fractal(self):
        return self.fractal_computer.output

# handle mouse clicks for zooming
class FractalRegion(QLabel):
    zoom = Signal(float, float, float)

    def __init__(self):
        super().__init__()
        self.BACKGROUND_TEXT = \
        "Start real-time generation first.\n" \
        "Hold the left mouse button to zoom in, and the right mouse button to zoom out.\n" \
        "Press Space to increase the Z coordinate, Backspace to decrease it.\n" \
        "Press C to cycle the pattern.\n"

        self.setMinimumSize(*Config.MIN_FRACTAL_REGION_SIZE)
        self.setStyleSheet("background-color: black; font: 15pt;")
        self.setAlignment(Qt.AlignCenter)
        self.setText(self.BACKGROUND_TEXT)

        self.is_zooming = False
        self.last_mouse_pos = None

        self.zoom_timer = QTimer()
        self.zoom_timer.timeout.connect(self.continuous_zoom)
        self.zoom_timer.setInterval(Config.TIMER_INTERVAL)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        size = min(event.size().width(), event.size().height())
        self.setFixedSize(size, size)

    def mousePressEvent(self, event):
        if event.button() in [Qt.LeftButton, Qt.RightButton]:
            # Calculate click position as ratio of fractal canvas
            x_ratio = event.position().x() / self.width()
            y_ratio = event.position().y() / self.height()
            self.last_mouse_pos = (x_ratio, y_ratio)
            
            if event.button() == Qt.LeftButton:
                self.zoom_proportion = Config.ZOOM_FACTOR
            else:
                self.zoom_proportion = 1/Config.ZOOM_FACTOR

            self.is_zooming = True
            self.zoom_timer.start()
            self.zoom.emit(x_ratio, y_ratio, self.zoom_proportion)

    def mouseReleaseEvent(self, event):
        if self.is_zooming:
            self.is_zooming = False
            self.zoom_timer.stop()

    def mouseMoveEvent(self, event):
        if self.is_zooming:
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

        icon = QIcon()
        icon.addFile(".\outputs\lyapunov_xyyyxyxyy.png")
        self.setWindowIcon(icon)

        # Initialize variables
        self.current_high_res_image = None
        self.worker = None
        self.mode = IDLE

        self.z = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        self.LOW_RES_BTN_TEXT = "Start Real-Time Generation"
        self.LOW_RES_BTN_STYLE = "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }"     
        self.HIGH_RES_BTN_TEXT = "Generate High-Res Image"
        self.GENERATING_BTN_TEXT = "Generating High-Res Image..."
        self.HIGH_RES_BTN_STYLE = "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }"
        self.SAVE_BTN_TEXT = "Save High-Res Image"
        self.SAVE_BTN_STYLE = "QPushButton { background-color: #FF9800; color: white; padding: 8px; }"

        gpu = get_current_device()
        self.max_image_size = int((gpu.MAX_GRID_DIM_X * gpu.MAX_THREADS_PER_BLOCK)**0.5)

        self.worker = FractalWorker()
        self.worker.finished.connect(self.on_image_generated)
        self.worker.stage_updated.connect(self.on_progress_updated)

        # timer to debounce fractal generation
        self.regeneration_timer = QTimer()
        self.regeneration_timer.setSingleShot(True)
        self.regeneration_timer.timeout.connect(self.start_image_gen)

        self.setFocusPolicy(Qt.StrongFocus)

        self.keys_pressed = set()
    
        # Timer for continuous z parameter changes
        self.z_timer = QTimer()
        self.z_timer.timeout.connect(self.continuous_z_change)
        self.z_timer.setInterval(Config.TIMER_INTERVAL)

        self.loading_animation_frames = ['|', '/', '-', '\\']
        self.loading_animation_index = 0
        self.loading_animation_timer = QTimer(self)
        self.loading_animation_timer.timeout.connect(self.update_loading_text)

        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QWidget()
        left_panel.setMaximumWidth(Config.LEFT_PANEL_MAX_WIDTH)
        left_layout = QVBoxLayout(left_panel)
        
        self.create_top_fields(left_layout)
        
        self.create_resolution_settings(left_layout)
        
        self.create_color_pickers(left_layout)
        
        self.create_buttons(left_layout)
        
        # push everything to top
        left_layout.addStretch()

        # Right panel for fractal display
        self.fractal_region = FractalRegion()
        self.fractal_region.zoom.connect(self.on_zoom)
        
        # Create scroll area for the fractal display
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.fractal_region)
        scroll_area.setWidgetResizable(True)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(scroll_area, 1)

    def create_top_fields(self, layout):
        # Create grid layout for input fields
        grid_layout = QGridLayout()
        
        coords_fields = {
            "z": ("Z", 2.86),
            "x_min": ("X Min", 1.009),
            "x_max": ("X Max", 1.244),
            "y_min": ("Y Min", 3.662),
            "y_max": ("Y Max", 3.898),
        }

        self.pattern = QLineEdit("yyxxyyyyyzz")
        self.pattern.textChanged.connect(self.on_parameter_changed)
        self.pattern.setValidator(QRegularExpressionValidator("^[xyzXYZ]{0,50}$"))
        grid_layout.addWidget(QLabel("Pattern"), 0, 0)
        grid_layout.addWidget(self.pattern, 0, 1)

        for i, (field_name, (label_text, default_value)) in enumerate(coords_fields.items(), 1):
            spin_box = make_float_spinbox(value=default_value, singleStep=0.1,  minimum=0.01, maximum=4, decimals=5)
            spin_box.valueChanged.connect(self.on_parameter_changed)

            setattr(self, field_name, spin_box)

            grid_layout.addWidget(QLabel(label_text), i, 0)
            grid_layout.addWidget(spin_box, i, 1)

        self.color_res = make_integer_spinbox(value=1900, singleStep=50, minimum=50, maximum=10_000)
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
        self.low_res_size = make_integer_spinbox(value=Config.DEFAULT_REAL_TIME_SIZE,
                                         singleStep=50, minimum=100, maximum=1500)
        self.low_res_size.setKeyboardTracking(False)
        self.low_res_size.valueChanged.connect(self.on_parameter_changed)
        res_layout.addWidget(self.low_res_size, 1, 1)

        res_layout.addWidget(QLabel("Iterations:"), 2, 0)
        self.low_res_iter = make_integer_spinbox(value=Config.DEFAULT_REAL_TIME_ITER,
                                         singleStep=50, minimum=50, maximum=5000)
        self.low_res_iter.setKeyboardTracking(False)
        self.low_res_iter.valueChanged.connect(self.on_parameter_changed)
        res_layout.addWidget(self.low_res_iter, 2, 1)
        
        high_res_label = QLabel("High Resolution Mode:")
        high_res_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        res_layout.addWidget(high_res_label, 3, 0, 1, 2)
        
        res_layout.addWidget(QLabel("Size:"), 4, 0)
        self.high_res_size = make_integer_spinbox(value=min(Config.DEFAULT_HIGH_RES_SIZE, self.max_image_size),
                                                  singleStep=50, minimum=500, maximum=self.max_image_size)
        self.high_res_size.setKeyboardTracking(False)
        res_layout.addWidget(self.high_res_size, 4, 1)
        
        res_layout.addWidget(QLabel("Iterations:"), 5, 0)
        self.high_res_iter = make_integer_spinbox(value=Config.DEFAULT_HIGH_RES_ITER, singleStep=50,
                                                  minimum=500, maximum=50_000)
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
            color_button.setFocusPolicy(Qt.NoFocus)

            color_row_layout.addWidget(color_input)
            color_row_layout.addWidget(color_button)
            
            self.color_inputs.append(color_input)
            self.color_buttons.append(color_button)
            
            color_layout.addLayout(color_row_layout)
        
        layout.addWidget(color_group)

    def create_buttons(self, layout):
        self.low_res_btn = make_button(
            self.LOW_RES_BTN_TEXT, 
            self.LOW_RES_BTN_STYLE,
            lambda: self.set_mode({IDLE: REAL_TIME,
                                   REAL_TIME: IDLE,
                                   HIGH_RES: REAL_TIME,
                                   GENERATING: REAL_TIME}[self.mode])
        )
        
        def on_high_res_clicked():
            self.set_mode(GENERATING)
            self.start_image_gen()

        self.high_res_btn = make_button(
            self.HIGH_RES_BTN_TEXT,
            self.HIGH_RES_BTN_STYLE, 
            on_high_res_clicked
        )
        
        self.save_btn = make_button(
            self.SAVE_BTN_TEXT,
            self.SAVE_BTN_STYLE,
            self.save_image
        )
        self.save_btn.setEnabled(False)
        
        layout.addWidget(self.low_res_btn)
        layout.addWidget(self.high_res_btn)
        layout.addWidget(self.save_btn)

    def pick_color(self, index):
        if self.mode not in [IDLE, REAL_TIME, HIGH_RES]:
            return

        current_color = QColor(self.color_inputs[index].text())
        
        dialog = QColorDialog(current_color, self)
        dialog.setOption(QColorDialog.DontUseNativeDialog, True)
        
        dialog.currentColorChanged.connect(
            lambda color: self.set_image_color(index, color)
        )

        # apply or revert based on user choice
        if dialog.exec() == QColorDialog.Accepted:
            self.color_inputs[index].setText(dialog.selectedColor().name())
        else:
            self.set_image_color(index, current_color)

    def set_image_color(self, index, color):
        if color.isValid() and self.mode in [REAL_TIME, HIGH_RES]:
            colors = list(map(lambda x: x.text(), self.color_inputs))
            colors[index] = color.name()
            img = self.worker.fractal_computer.apply_gradient(colors)
            if self.mode == HIGH_RES:
                self.current_high_res_image = img
            self.display_image(img, size=self.fractal_region.size())

    # Correct bounds to ensure min < max
    def sanitize_inputs(self, changed_field):
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

    # Handle parameter changes by regenerating the fractal if in real-time mode
    def on_parameter_changed(self, *, delay=Config.REGENERATION_DELAY):
        self.sanitize_inputs(self.sender())

        if self.mode == REAL_TIME:
            self.regeneration_timer.stop()
            self.regeneration_timer.start(delay)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Space, Qt.Key_Backspace) and self.mode == REAL_TIME:
            self.keys_pressed.add(key)
            self.continuous_z_change()
            if not self.z_timer.isActive():
                self.z_timer.start()
            event.accept()
        elif key == Qt.Key_C and self.mode == REAL_TIME:
            # avoid regeneration delay
            self.pattern.blockSignals(True)
            self.pattern.setText(self.pattern.text()[-1] + self.pattern.text()[:-1])
            self.pattern.blockSignals(False)

            self.regeneration_timer.stop()  # cancel any pending regeneration
            self.start_image_gen()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Space, Qt.Key_Backspace):
            self.keys_pressed.discard(key)
            if not {Qt.Key_Space, Qt.Key_Backspace} & self.keys_pressed:
                self.z_timer.stop()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def continuous_z_change(self):
        delta = 0.01
        if Qt.Key_Backspace in self.keys_pressed:
            delta *= -1

        self.z.blockSignals(True)
        self.z.setValue((self.z.value() + delta) % 4)
        self.z.blockSignals(False)

        self.regeneration_timer.stop()  # cancel any pending regeneration
        self.start_image_gen()

    # extract parameters from GUI fields
    def get_fractal_params(self):
        assert self.mode in [GENERATING, REAL_TIME]
        colors = []
        for color_input in self.color_inputs:
            color = color_input.text().lower()
            if valid_hex_string(color):
                colors.append(color)

        if not colors:
            colors = ["#000000"]  # default color if none provided
        
        # choose resolution settings based on mode
        if self.mode == REAL_TIME:
            size = self.low_res_size.value()
            iterations = self.low_res_iter.value()
        elif self.mode == GENERATING:
            size = self.high_res_size.value()
            iterations = self.high_res_iter.value()

        params = {
            "pattern": self.pattern.text(),
            "x_min": self.x_min.value(),
            "x_max": self.x_max.value(),
            "y_min": self.y_min.value(),
            "y_max": self.y_max.value(),
            "z": self.z.value(),
            "size": size,
            "color_resolution": self.color_res.value(),
            "num_iter": iterations,
            "colors": colors
        }

        return params

    def set_mode(self, mode):
        self.mode = mode
        if mode == REAL_TIME:
            self.save_btn.setEnabled(False)

            # update button to show exit option
            self.low_res_btn.setText("Exit Real-Time Mode")
            self.low_res_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
            self.low_res_btn.setEnabled(True)

            self.start_image_gen()

        # stop any pending regeneration timer
        self.regeneration_timer.stop()
        # clear the display
        self.fractal_region.clear()

        if mode == IDLE:
            self.fractal_region.setText(self.fractal_region.BACKGROUND_TEXT)

            # Update button text
            self.low_res_btn.setText(self.LOW_RES_BTN_TEXT)
            self.low_res_btn.setStyleSheet(self.LOW_RES_BTN_STYLE)
        
        elif mode == GENERATING:
            self.fractal_region.setText(self.fractal_region.BACKGROUND_TEXT)

            # Update button text
            self.low_res_btn.setText(self.LOW_RES_BTN_TEXT)
            self.low_res_btn.setStyleSheet(self.LOW_RES_BTN_STYLE) 

            self.high_res_btn.setText(self.GENERATING_BTN_TEXT)

        elif mode == HIGH_RES:
            self.loading_animation_timer.stop()

            # re enable generate button
            self.high_res_btn.setText(self.HIGH_RES_BTN_TEXT)
            self.high_res_btn.setEnabled(True)

            # enable save button
            self.save_btn.setEnabled(True)
    

    def start_image_gen(self):
        assert self.mode in [REAL_TIME, GENERATING]
        if self.worker.isRunning():
            return

        params = self.get_fractal_params()
        self.worker.set_parameters(params)
        self.worker.start()

    def on_image_generated(self, img):
        assert self.mode not in [IDLE, HIGH_RES]

        if self.mode == REAL_TIME:
            self.display_image(img, size=self.fractal_region.size())
        elif self.mode == GENERATING:
            self.set_mode(HIGH_RES)

            self.current_high_res_image = img
            # Show high-res image in new window using PIL
            image = Image.fromarray(np.swapaxes(img, 0, 1))
            image.show()

            self.display_image(img, size=self.fractal_region.size())

    def on_progress_updated(self, info_string, stage):
        if self.mode == GENERATING:
            self.info_string = info_string
            self.generation_stage = stage
            self.loading_animation_timer.start(100)

    def update_loading_text(self):
        frame = self.loading_animation_frames[self.loading_animation_index]
        self.loading_animation_index = (self.loading_animation_index + 1) % len(self.loading_animation_frames)
        self.fractal_region.setText(
            f"{self.generation_stage}/{self.worker.total_stage_number}: {self.info_string} {frame}"
        )

    def display_image(self, img, size=None):
        img = np.ascontiguousarray(np.swapaxes(img, 0, 1))
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimg)
        if size:
            pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.fractal_region.setPixmap(pixmap)

    def on_zoom(self, x_ratio, y_ratio, zoom_proportion):
        if self.mode != REAL_TIME:
            return

        new_bounds = self.zoom_to(x_ratio, y_ratio, zoom_proportion)

        # update input fields
        self.x_min.setValue(new_bounds[0])
        self.x_max.setValue(new_bounds[1])
        self.y_min.setValue(new_bounds[2])
        self.y_max.setValue(new_bounds[3])

        self.start_image_gen()
    
    # calculate new region bounds after zoom
    def zoom_to(self, x_ratio, y_ratio, zoom_proportion):
        x_min = self.x_min.value()
        x_max = self.x_max.value()
        y_min = self.y_min.value()
        y_max = self.y_max.value()

        # center zoom
        size = self.low_res_size.value()
        pos_x = 0.1 * (x_ratio * size - size/2) + size/2
        pos_y = 0.1 * (y_ratio * size - size/2) + size/2

        # convert mouse position ratios to fractal coordinates
        pos_x = x_min + (x_max - x_min) * pos_x/size
        # we flip the y axis because it is pointing downwards
        pos_y = y_min + (y_max - y_min) * (1 - pos_y/size)

        new_x_min = pos_x - (zoom_proportion * (x_max - x_min)) / 2
        new_y_min = pos_y - (zoom_proportion * (y_max - y_min)) / 2
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
        if self.mode == HIGH_RES:
            try:
                colors = [color_input.text().replace('#', '').lower() for color_input in self.color_inputs]

                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save High-Res Fractal Image",
                    self.pattern.text() + "_" + 
                    str(round(self.x_min.value(), 3)) + "_" + 
                    str(round(self.x_max.value(), 3)) + "_" + 
                    str(round(self.y_min.value(), 3)) + "_" + 
                    str(round(self.y_max.value(), 3)) + "_z_" +
                    str(round(self.z.value(), 3)) + "_res_" + 
                    str(self.color_res.value()) + "_" +
                    "-".join(colors) + ".png",
                    "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
                )

                if file_path:
                    img = Image.fromarray(np.swapaxes(self.current_high_res_image, 0, 1))
                    img.save(file_path)
                    QMessageBox.information(self, "Success", f"High-resolution image saved to {file_path}")

            except Exception as e:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Error")
                msg_box.setText(f"Error saving image: {e}")
                msg_box.exec()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = FractalApp()
    window.show()

    app.exec()


if __name__ == "__main__":
    main()