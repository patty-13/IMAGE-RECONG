import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QScrollArea

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import pyqtSignal

from test_data import *
from SeqFinder import *

class LeftPanel(QWidget):
    row_data_signal = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        # Variables
        self.start_time = 0  # Initial value in minutes (8:00 AM)
        self.end_time = 0   # Initial value in minutes (5:00 PM)

        self.ST = 0
        self.ET = 0

        self.backend_data = getTestData()
        self.seqData = None
        self.matchNumber = 5
        self.portNumber = 52428

        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setSpacing(20)  # Reduce spacing between widgets
        self.layout.setContentsMargins(10, 10, 10, 10)  # Reduce margins
        self.setLayout(self.layout)

        # Header
        self.header1 = QLabel("SELECTED IMAGE")
        self.header1.setStyleSheet("font-weight: bold; font-size: 10px;")
        self.header1.setAlignment(Qt.AlignLeft)  # Center-align header
        self.layout.addWidget(self.header1, alignment=Qt.AlignTop)

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)  # Set fixed size for the image
        #self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignTop)

        # Load and display the image
        self.load_image()  # Replace with actual path

        # Controls header
        self.header2 = QLabel("CONTROLS")
        self.header2.setStyleSheet("font-weight: bold; font-size: 10px;")
        self.header2.setAlignment(Qt.AlignLeft)  # Center-align hader
        self.layout.addWidget(self.header2, alignment=Qt.AlignTop)

        # Start slider
        self.start_time_box_main = QVBoxLayout()
        self.start_time_box_main.setSpacing(5)  # Tighten spacing within this layout
        self.start_time_box_sub = QHBoxLayout()
        self.start_time_box_sub.setSpacing(100)  # Tighten spacing in sub-layout
        self.start_time_input = QLineEdit(self)
        self.start_time_input.setText(self.format_time(self.start_time))
        self.start_time_input.setMinimumSize(10, 1)
        self.start_time_input.textChanged.connect(self.on_input_change_start)
        self.start_time_box_sub.addWidget(QLabel("Start Time (HH:MM)"))
        self.start_time_box_sub.addWidget(self.start_time_input)

        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.setRange(0, 1440)
        self.start_slider.setValue(self.start_time)
        self.start_slider.valueChanged.connect(self.update_start_time)
        self.start_time_box_main.addLayout(self.start_time_box_sub)
        self.start_time_box_main.addWidget(self.start_slider)

        self.layout.addLayout(self.start_time_box_main)

        # End slider
        self.end_time_box_main = QVBoxLayout()
        self.end_time_box_main.setSpacing(5)
        self.end_time_box_sub = QHBoxLayout()
        self.end_time_box_sub.setSpacing(110)

        self.end_time_input = QLineEdit(self)
        self.end_time_input.setText(self.format_time(self.end_time))
        self.end_time_input.setMinimumSize(10, 1)
        self.end_time_input.textChanged.connect(self.on_input_change_end)
        self.end_time_box_sub.addWidget(QLabel("End Time (HH:MM)"))
        self.end_time_box_sub.addWidget(self.end_time_input)

        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.setRange(0, 1440)  # 0 to 1440 minutes
        self.end_slider.setValue(self.end_time)
        self.end_slider.valueChanged.connect(self.update_end_time)
        self.end_time_box_main.addLayout(self.end_time_box_sub)
        self.end_time_box_main.addWidget(self.end_slider)

        self.layout.addLayout(self.end_time_box_main)

        # Top matches input
        # Top matches slider
        self.top_match_box_main = QVBoxLayout()
        self.top_match_box_main.setSpacing(5)

        self.top_match_box_sub = QHBoxLayout()
        self.top_match_box_sub.setSpacing(130)

        self.Nmatch_input = QLineEdit(self)
        self.Nmatch_input.setText(str(self.matchNumber))
        self.Nmatch_input.textChanged.connect(self.on_input_change_top_matches)

        self.top_match_box_sub.addWidget(QLabel("Top Matches"))
        self.top_match_box_sub.addWidget(self.Nmatch_input)

        self.top_match_slider = QSlider(Qt.Horizontal)
        self.top_match_slider.setRange(1, 100)  # Set range for top matches (1 to 100)
        self.top_match_slider.setValue(self.matchNumber)
        self.top_match_slider.valueChanged.connect(self.update_top_matches)

        self.top_match_box_main.addLayout(self.top_match_box_sub)
        self.top_match_box_main.addWidget(self.top_match_slider)

        self.layout.addLayout(self.top_match_box_main)

        # Add a stretchable area before the button to push it upwards
        self.layout.addStretch(1)

        # Submit button
        self.submit_button = QPushButton('Find Matches')
        self.submit_button.setMinimumSize(300, 50)  # Larger button size
        self.submit_button.clicked.connect(self.find_sequences)
        self.layout.addWidget(self.submit_button, alignment=Qt.AlignCenter)

        # Add a stretchable area below to balance remaining space
        # self.layout.addStretch(1)



    # def load_image(self, image_path):
    #     """Load and display the image."""
    #     pixmap = QPixmap(image_path)
    #     # Resize the image to fit within the label's size
    #     pixmap = pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #     self.image_label.setPixmap(pixmap)
    #     self.image_label.setAlignment(Qt.AlignCenter)
    def load_image(self):
        """Load and display the image."""
        test_folder_path = "D:\pycharm_projects\CLEARAIMODELS\IMAGERECOGNIZER\TEST_PLOT"
        image_path = os.path.join(test_folder_path, f"TEST_SEQ_PLOT.png")

        # Ensure backend_data is properly handled
        try:
            if isinstance(self.backend_data, str):
                y_data = eval(self.backend_data)
            else:
                y_data = self.backend_data  # Assume it's already a list or compatible object

            x_data = range(len(y_data))  # Generate x_data based on y_data length
            self.create_test_plot(x_data, y_data, f"TEST_SEQ_PLOT")

            pixmap = QPixmap(image_path)
            if not pixmap.isNull():  # Check if image is loaded successfully
                # Resize the image to fit within the label's size
                pixmap = pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
            else:
                print(f"Failed to load image at {image_path}")
        except Exception as e:
            print(f"Error processing backend data: {e}")

    def create_test_plot(self, x_data, y_data, title):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.plot(x_data, y_data, label="ochl_avg")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.axis('off')

        # Save the plot
        plot_path = f"D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/TEST_PLOT/{title}.png"
        plt.savefig(plot_path)
        plt.close(fig)

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
            self.sendToBackend()
        except ValueError:
            pass  # Invalid input, don't update the slider

    def on_input_change_end(self):
        """Update end time slider when the input field changes."""
        try:
            hours, minutes = map(int, self.end_time_input.text().split(":"))
            self.end_time = hours * 60 + minutes
            self.end_slider.setValue(self.end_time)
            self.sendToBackend()
        except ValueError:
            pass  # Invalid input, don't update the slider

    def format_time(self, minutes):
        """Convert minutes to HH:MM format."""
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}"

    def update_top_matches(self):
        """Update top matches input field when slider value changes."""
        self.matchNumber = self.top_match_slider.value()
        self.Nmatch_input.setText(str(self.matchNumber))

    def on_input_change_top_matches(self):
        """Update top matches slider when the input field changes."""
        try:
            value = int(self.Nmatch_input.text())
            if 1 <= value <= 100:  # Ensure value is within slider range
                self.matchNumber = value
                self.top_match_slider.setValue(value)
        except ValueError:
            pass  # Invalid input, don't update the slider


    ############# SEQUENCE FINDER ###################################
    def find_sequences(self):
        try:
            print(self.backend_data)
            seqMatcher = sequenceFinder(self.backend_data)
            self.seqData = seqMatcher.seqMatches(self.matchNumber, False)
            self.store_plots(self.seqData)
        except Exception as e:
            print(f"Error in find_sequences: {e}")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.rename(columns={'<DATE>': 'date',
                           '<TIME>': 'time',
                           '<OPEN>': 'open',
                           '<HIGH>': 'high',
                           '<LOW>': 'low',
                           '<CLOSE>': 'close',
                           '<TICKVOL>': 'tickvol',
                           '<VOL>': 'volume',
                           '<SPREAD>': 'spread'},
                  inplace=True)
        return df

    def store_plots(self, data):
        row_data= []
        plot_dir = 'D:\pycharm_projects\CLEARAIMODELS\IMAGERECOGNIZER\PLOT_IMAGES'
        complete_chart = pd.read_csv('D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/@ENQ_M5.csv', sep='\t').pipe(self.preprocess)
        print(complete_chart)
        for i, entry in enumerate(data[:5]):  # Top 5 results

            # defining paths
            match_plot_path = os.path.join(plot_dir, f"Matching_Plot_{i+1}.png")
            complete_plot_path = os.path.join(plot_dir, f"Complete_Plot_{i+1}.png")

            # creating matching plot
            self.create_plot(range(len(eval(entry['ochl_avg']))), eval(entry['ochl_avg']), f"Matching_Plot_{i + 1}")

            # saving complete chart
            complete_chart = complete_chart[complete_chart['date'] == entry['date']].reset_index(drop=True)
            self.create_complete_plot(complete_chart, f"Complete_Plot_{i + 1}")

            # Information column
            info = (
                f"Date: {entry['date']}\n"
                f"Start Time: {entry['start_time']}\n"
                f"End Time: {entry['end_time']}\n"
                f"DTW Distance: {entry['dtw_distance']:.6f}"
            )
            self.ST = entry['start_time']
            self.ET = entry['end_time']

            # Add row
            row_data.append(
                [
                    f"{complete_plot_path}",
                    f"{match_plot_path}",
                    info
                ]
            )
        self.row_data_signal.emit(row_data)

    def create_plot(self, x_data, y_data, title):

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.plot(x_data, y_data, label="ochl_avg")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.axis('off')

        # Save the plot
        plot_path = f"D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES/{title}.png"
        plt.savefig(plot_path)
        plt.close(fig)

    def create_complete_plot(self, complete_chart, title):
        fig = go.Figure(data=[go.Candlestick(x=complete_chart['time'],
                                             open=complete_chart['open'],
                                             high=complete_chart['high'],
                                             low=complete_chart['low'],
                                             close=complete_chart['close'])])
        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    x0= self.ST,  # Set start time
                    x1= self.ET,  # Set end time
                    y0=0,  # Y-axis start (you can adjust this depending on your data)
                    y1=1,  # Y-axis end (usually the range for the candlestick data)
                    xref="x",  # Use x-axis scale
                    yref="paper",  # Y-axis is defined from 0 to 1 (height of the chart)
                    fillcolor="rgba(0, 100, 255, 0.3)",  # Set color of the bounding box
                    line=dict(color="rgba(0, 100, 255, 0.5)", width=2)  # Set border of the bounding box
                )
            ],
            xaxis_rangeslider_visible=False,  # Disable the range slider
            title=title,
            width=800,  # Set the width of the chart
            height=600  # Set the height of the chart
        )

        fig.write_image(f'D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES/{title}.png')



######################## FLASK CONNECTION ###################################

    # Connection to backend and getting data from backend
    def sendToBackend(self):
        data = {
            'start_time': self.start_time,
            'end_time': self.end_time
        }
        print(f"DATA :{data}")
        try:
            response = requests.post(f'http://127.0.0.1:{self.portNumber}/uiSendData', json=data)
            if response.status_code == 200:
                print("Data sent successfully")
                self.getDataFromBackend()
            else:
                print(f"Error sending time : {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error from sendToBackend :{e}")
    def getDataFromBackend(self):
        try:
            response = requests.get(f'http://127.0.0.1:{self.portNumber}/sendDatatoUI')
            if response.status_code == 200:
                self.backend_data = response.json()
                print("Data received from backend", self.backend_data)
            else:
                print(f"Error fetching data: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error from getDataFromBackend: {e}")


class RightPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Scroll area for the table
        self.scroll_area = QScrollArea(self)
        self.layout.addWidget(self.scroll_area)

        # Create a table widget
        self.table_widget = QTableWidget()
        self.scroll_area.setWidget(self.table_widget)
        self.scroll_area.setWidgetResizable(True)

        # Set the table columns and rows
        self.table_widget.setColumnCount(3)  # 3 columns: Image 1, Image 2, Data Info
        self.table_widget.setHorizontalHeaderLabels(['CHART', 'MATCHING PLOT', 'DATA INFO'])

        # Example data to populate the table
        # row_data = [
        #     ("D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES/Complete_Plot_1.png",
        #      "D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES/Matching_Plot_1.png", "Data Info 1"),
        #     ("D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES/Complete_Plot_2.png",
        #      "D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES/Matching_Plot_2.png", "Data Info 2"),
        #     ("D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES/Complete_Plot_3.png",
        #      "D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES/Matching_Plot_3.png", "Data Info 3")
        # ]
        #
        # # Populate the table with the data
        # self.update_table(row_data)

    def update_table(self, row_data):
        """Populate the table with image data and information."""
        self.table_widget.setRowCount(len(row_data))  # Set the number of rows based on data length
        self.table_widget.setColumnWidth(0, 1000)  # Set width of the first column (Image 1)
        self.table_widget.setColumnWidth(1, 350)  # Set width of the second column (Image 2)
        self.table_widget.setColumnWidth(2, 200)  # Set width of the third column (Data Info)

        for row_index, row in enumerate(row_data):
            # Image 1 (first column)
            self.table_widget.setRowHeight(row_index, 500)
            image_1 = QLabel()
            pixmap_1 = QPixmap(row[0])  # Load the first image
            image_1.setPixmap(pixmap_1.scaled(500, 3000, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            image_1.setAlignment(Qt.AlignCenter)
            self.table_widget.setCellWidget(row_index, 0, image_1)

            # Image 2 (second column)
            image_2 = QLabel()
            pixmap_2 = QPixmap(row[1])  # Load the second image
            image_2.setPixmap(pixmap_2.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            image_2.setAlignment(Qt.AlignCenter)
            self.table_widget.setCellWidget(row_index, 1, image_2)

            # Data info (third column)
            data_info_item = QTableWidgetItem(row[2])  # Add data text
            self.table_widget.setItem(row_index, 2, data_info_item)

class MainApp(QApplication):
    def __init__(self):
        super().__init__([])
        self.window = QWidget()
        self.layout = QHBoxLayout(self.window)

        self.left_panel = LeftPanel()
        self.left_panel.setFixedWidth(350)  # Set fixed width for the left panel
        self.layout.addWidget(self.left_panel)

        self.right_panel = RightPanel()
        self.layout.addWidget(self.right_panel)  # Right panel will take remaining space
        self.left_panel.row_data_signal.connect(self.right_panel.update_table)

        self.window.setLayout(self.layout)
        self.window.show()


if __name__ == "__main__":
    app = MainApp()
    app.exec_()
