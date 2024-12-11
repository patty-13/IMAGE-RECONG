from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QScrollArea

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor

class TimeRangeSelector(QWidget):
    def __init__(self):
        self.start_time = 480
        self.end_time = 1020


        # start slider
        self.start_time_box_main = QVBoxLayout()
        self.start_time_box_sub = QHBoxLayout()
        self.start_time_input = QLineEdit(self)
        self.start_time_input.setText(self.format_time(self.start_time))
        self.start_time_input.textChanged.connect(self.on_input_change_start)
        self.start_time_box_sub.addWidget(QLabel("Start Time (HH:MM)"))
        self.start_time_box_sub.addWidget(self.start_time_input)

        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.setRange(0, 1440)
        self.start_slider.setValue(self.start_time)
        self.start_slider.valueChanged.connect(self.update_start_time)
        self.start_time_main_box.addWidget(self.start_time_box_sub)
        self.start_time_box_main.addWidget(self.start_slider)

        # end slider

        self.end_time_box_main = QVBoxLayout()
        self.end_time_box_sub = QHBoxLayout()

        self.end_time_input = QLineEdit(self)
        self.end_time_input.setText(self.format_time(self.end_time))
        self.end_time_input.textChanged.connect(self.on_input_change_end)
        self.end_time_box_sub.addWidget(QLabel("End Time (HH:MM)"))
        self.end_time_box_sub.addWidget(self.end_time_input)

        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.setRange(0, 1440)  # 0 to 1440 minutes
        self.end_slider.setValue(self.end_time)
        self.end_slider.valueChanged.connect(self.update_end_time)
        self.end_time_box_main.addWidget(self.end_time_box_sub)
        self.end_time_box_main.addWidget(self.end_slider)

    def update_start_time(self):
        """Update start time input field when slider value changes."""
        self.start_time = self.start_slider.value()
        self.start_time_input.setText(self.format_time(self.start_time))

    def update_end_time(self):
        """Update end time input field when slider value changes."""
        self.end_time = self.end_slider.value()
        self.end_time_input.setText(self.format_time(self.end_time))

    def on_input_change_start(self):
        """Update start time slider when the input field changes."""
        try:
            hours, minutes = map(int, self.start_time_input.text().split(":"))
            self.start_time = hours * 60 + minutes
            self.start_slider.setValue(self.start_time)
        except ValueError:
            pass  # Invalid input, don't update the slider

    def on_input_change_end(self):
        """Update end time slider when the input field changes."""
        try:
            hours, minutes = map(int, self.end_time_input.text().split(":"))
            self.end_time = hours * 60 + minutes
            self.end_slider.setValue(self.end_time)
        except ValueError:
            pass  # Invalid input, don't update the slider

    def format_time(self, minutes):
        """Convert minutes to HH:MM format."""
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}"
