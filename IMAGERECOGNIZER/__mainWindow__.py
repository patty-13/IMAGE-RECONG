import threading

from SeqFinder import *
from timeSlider import *
import yfinance as yf
from test_data import *

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.pagelayout import PageLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.recycleview import RecycleView
from kivy.core.window import Window

from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.slider import MDSlider
from kivymd.uix.textfield import MDTextField
from kivy.metrics import dp
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.screen import MDScreen


class LeftPanel(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spacing = 10
        self.padding = 10
        self.orientation = 'vertical'
        self.matchNumber = 5
        self.start_time = 0
        self.end_time = 0
        self.backend_data = getTestData()
        self.faiss_index_path = None  # path to stored files
        self.matchedStoredSeq = None  # path to stored files
        self.seqData = None

        # IMAGE DISPLAY
        header_1 = MDLabel(text="SELECTED IMAGE", halign='left', size_hint=(1, 0.1))
        self.add_widget(header_1)

        # CHART CONTRONLS
        header_2 = MDLabel(text="CONTROLS", halign='left', size_hint=(1, 0.1))
        self.add_widget(header_2)

        # SLIDER CONTROLS
        self.time_selector = TimeRangeSelector(size_hint=(1, 0.4))
        self.time_selector.start_slider.bind(value=self.save_time_parameters)
        self.time_selector.end_slider.bind(value=self.save_time_parameters)
        self.time_selector.start_slider_input.bind(text=self.save_time_parameters)
        self.time_selector.end_slider_input.bind(text=self.save_time_parameters)
        self.add_widget(self.time_selector)

        # TOP N MATCHES
        # TOP N MATCHES
        Nmatch_slider_layout_main = BoxLayout(orientation='vertical', spacing=10, padding=[0, 10, 0, 10])

        # Sub-layout for label and text input on one line
        Nmatch_slider_layout_sub = BoxLayout(orientation='horizontal', spacing=1, size_hint=(1, None), height=40)
        Nmatch_slider_layout_sub.add_widget(MDLabel(text="TOP MATCHES", size_hint=(0.5, 1)))
        self.Nmatch_input = MDTextField(
            hint_text="number",
            text="5",
            mode="rectangle",
            size_hint=(0.5, 1)
        )
        self.Nmatch_input.bind(text=self.on_match_change)
        Nmatch_slider_layout_sub.add_widget(self.Nmatch_input)

        # Add the sub-layout to the main layout
        Nmatch_slider_layout_main.add_widget(Nmatch_slider_layout_sub)

        # Slider on the next line
        self.Nmatch_slider = MDSlider(min=0, max=100, step=1, size_hint=(1, None), height=40)
        self.Nmatch_slider.bind(value=self.update_nmatch)
        Nmatch_slider_layout_main.add_widget(self.Nmatch_slider)

        # Add the main layout to the panel
        self.add_widget(Nmatch_slider_layout_main)

        # FIND MATCHES
        self.submit_button = Button(text='Find Matches', size_hint=(1, 0.1))

        self.submit_button.bind(on_press=lambda instance: self.findSequences())
        self.add_widget(self.submit_button)

    def save_time_parameters(self, *args):
        self.start_time = self.time_selector.start_slider_input.text
        self.end_time = self.time_selector.end_slider_input.text
        self.send_to_backend()
        print(f"start Time:{self.start_time} , end Time : {self.end_time}")

    def save_match_parameter(self, *args):
        self.matchNumber = int(self.Nmatch_input.text)

    def on_match_change(self, instance, text):
        try:
            value = int(text)
            if 0 <= value <= 100:
                self.Nmatch_slider.value = value
                self.save_match_parameter()
            else:
                raise ValueError("out of range")
        except ValueError:
            self.Nmatch_input.text = str(self.matchNumber)

    def update_nmatch(self, instance, value):
        self.Nmatch_input.text = str(int(value))
        self.save_match_parameter()

    # def convertBackendData(self):
    #     self.backend_data['ochl_avg'] = self.backend_data[]

    def findSequences(self):
        seqMatcher = sequenceFinder(self.backend_data)

        self.seq_data = seqMatcher.seqMatches(self.matchNumber, True)
        print(self.seq_data)
        self.store_plots(self.seq_data)


    # def store_plots(self, data):
    #     row_data= []
    #     plot_dir = 'D:\pycharm_projects\CLEARAIMODELS\IMAGERECOGNIZER\PLOT_IMAGES'
    #     complete_chart = pd.read_csv('D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/@ENQ_M5.csv', sep='\t').pipe(self.preprocess)
    #     print(complete_chart)
    #     for i, entry in enumerate(data[:5]):  # Top 5 results
    #
    #         # defining paths
    #         match_plot_path = os.path.join(plot_dir, f"Matching_Plot_{i+1}.png")
    #         complete_plot_path = os.path.join(plot_dir, f"Complete_Plot_{i+1}.png")
    #
    #         # creating matching plot
    #         self.create_plot(range(len(eval(entry['ochl_avg']))), eval(entry['ochl_avg']), f"Matching_Plot_{i + 1}")
    #
    #         # saving complete chart
    #         complete_chart = complete_chart[complete_chart['date'] == entry['date']].reset_index(drop=True)
    #         self.create_complete_plot(complete_chart, f"Complete_Plot_{i + 1}")
    #
    #         # Information column
    #         info = (
    #             f"Date: {entry['date']}\n"
    #             f"Start Time: {entry['start_time']}\n"
    #             f"End Time: {entry['end_time']}\n"
    #             f"DTW Distance: {entry['dtw_distance']:.6f}"
    #         )
    #
    #         # Add row
    #         row_data.append(
    #             [
    #                 f"[img]{complete_plot_path}[/img]",
    #                 f"[img]{match_plot_path}[/img]",
    #                 info
    #             ]
    #         )
    #     app = MDApp.get_running_app()
    #     app.right_panel.update_table(row_data)
    def store_plots(self, data):
        row_data = []
        plot_dir = 'D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES'
        complete_chart = pd.read_csv('D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/@ENQ_M5.csv', sep='\t').pipe(
            self.preprocess)

        for i, entry in enumerate(data[:5]):  # Top 5 results
            match_plot_path = os.path.join(plot_dir, f"Matching_Plot_{i + 1}.png")
            complete_plot_path = os.path.join(plot_dir, f"Complete_Plot_{i + 1}.png")

            # Create the plots
            self.create_plot(range(len(eval(entry['ochl_avg']))), eval(entry['ochl_avg']), f"Matching_Plot_{i + 1}")
            complete_chart = complete_chart[complete_chart['date'] == entry['date']].reset_index(drop=True)
            self.create_complete_plot(complete_chart, f"Complete_Plot_{i + 1}")

            # Information column
            info = (
                f"Date: {entry['date']}\n"
                f"Start Time: {entry['start_time']}\n"
                f"End Time: {entry['end_time']}\n"
                f"DTW Distance: {entry['dtw_distance']:.6f}"
            )

            # Create image widgets
            match_img = Image(source=match_plot_path, size_hint=(None, None), size=(dp(200), dp(200)))
            complete_img = Image(source=complete_plot_path, size_hint=(None, None), size=(dp(200), dp(1000)))

            # BoxLayout for combining images and information
            row_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(200))
            row_layout.add_widget(complete_img)
            row_layout.add_widget(match_img)
            row_layout.add_widget(Label(text=info, size_hint=(None, None), width=dp(200)))

            row_data.append(row_layout)  # Add the row layout

        app = MDApp.get_running_app()
        app.right_panel.update_table(row_data)  # Update the table with new rows

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


    def create_plot(self, x_data, y_data, title):

        fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
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

        fig.write_image(f'D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/PLOT_IMAGES/{title}.png')


    # SENDING CURRENT TIME TO FLASK SERVER
    def send_to_backend(self):
        data = {
            'start_time': self.start_time,
            'end_time': self.end_time
        }
        try:
            response = requests.post("http://127.0.0.1:52428/uiSendData", json=data)
            if response.status_code == 200:
                print("Data sent succesfully")
                self.getDataFromBackend()
            else:
                print(f"Error sending data : {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error:{e}")

    # RECEVING TEST SEQ BACK TO FRONTEND
    def getDataFromBackend(self):
        try:
            response = requests.get('http://127.0.0.1:52428/sendDatatoUI')
            if response.status_code == 200:
                self.backend_data = response.json()
                print("Data received from backend", self.backend_data)
            else:
                print(f"Error fetching data:{response.status_code}-{response.text}")
        except Exception as e:
            print(f"Error:{e}")


# class RightPanel(BoxLayout):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.data_table = MDDataTable(
#             column_data=[("CHART", dp(200)),
#                          ("MATCHING IMAGE", dp(50)),
#                          ("INFORMATION", dp(30))
#                          ],
#             row_data = [],
#             size_hint =(1, 1)
#         )
#
#         #screen = MDScreen()
#         #screen.add_widget(self.data_table)
#         #self.add_widget(screen)
#         self.add_widget(self.data_table)
#
#     def update_table(self, row_data):
#         self.data_table.row_data = row_data

class RightPanel(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.scroll_view = ScrollView()
        self.layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.layout.bind(minimum_height=self.layout.setter('height'))

        self.scroll_view.add_widget(self.layout)
        self.add_widget(self.scroll_view)

    def update_table(self, row_data):
        self.layout.clear_widgets()  # Clear previous widgets

        for row in row_data:
            self.layout.add_widget(row)


        # class RightPanel(BoxLayout):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.orientation = 'vertical'
#
#         # Header row layout (for column names)
#         header_layout = GridLayout(cols=3, size_hint_y=None, height=50)
#         header_layout.add_widget(Label(text="Chart Image", size_hint=(1, 1)))
#         header_layout.add_widget(Label(text="Match Image", size_hint=(1, 1)))
#         header_layout.add_widget(Label(text="Info", size_hint=(1, 1)))
#         self.add_widget(header_layout)
#
#         # Scrollable content
#         self.layout = GridLayout(cols=3, size_hint=(1, None))
#         self.layout.bind(minimum_height=self.layout.setter('height'))
#         self.scroll_view = ScrollView()
#         self.scroll_view.add_widget(self.layout)
#         self.add_widget(self.scroll_view)
#
#     def update_table(self, row_data):
#         self.layout.clear_widgets()
#         for row in row_data:
#             # Add chart image
#             chart_image_path = row[0].replace("[img]", "").replace("[/img]", "")
#             self.layout.add_widget(Image(source=chart_image_path, size_hint=(0.3, None), height=150))
#
#             # Add match image
#             match_image_path = row[1].replace("[img]", "").replace("[/img]", "")
#             self.layout.add_widget(Image(source=match_image_path, size_hint=(0.3, None), height=150))
#
#             # Add text information
#             label = Label(text=row[2], size_hint=(0.4, None), height=150, text_size=(None, None))
#             label.bind(texture_size=label.setter('size'))
#             self.layout.add_widget(label)

# class RightPanel(BoxLayout):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.orientation = 'vertical'
#
#         # Scrollable content layout
#         self.layout = BoxLayout(orientation='vertical', size_hint=(1, None))
#         self.layout.bind(minimum_height=self.layout.setter('height'))
#
#         # ScrollView to make the content scrollable
#         self.scroll_view = ScrollView()
#         self.scroll_view.add_widget(self.layout)
#         self.add_widget(self.scroll_view)
#
#     def update_table(self, row_data):
#         # DPI-based scaling factor
#         dpi_chart = 5000  # DPI scaling for chart image
#         dpi_match = 200   # DPI scaling for match image
#         dpi_text = 50    # DPI scaling for text
#
#         self.layout.clear_widgets()
#
#         for row in row_data:
#             # Create BoxLayout for each row (3 columns)
#             row_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=200)
#
#             # Add chart image (DPI 200)
#             chart_image_path = row[0].replace("[img]", "").replace("[/img]", "")
#             chart_image = Image(source=chart_image_path, size_hint=(0.7, 1), width=dpi_chart)
#             row_layout.add_widget(chart_image)
#
#             # Add match image (DPI 50)
#             match_image_path = row[1].replace("[img]", "").replace("[/img]", "")
#             match_image = Image(source=match_image_path, size_hint=(0.2, 1), width=dpi_match)
#             row_layout.add_widget(match_image)
#
#             # Add information text (DPI 30)
#             label = Label(text=row[2], size_hint=(0.1, 1), width=dpi_text, font_size=14)
#             label.bind(texture_size=label.setter('size'))  # Adjust label size based on text
#             row_layout.add_widget(label)
#
#             # Add the row layout to the main layout
#             self.layout.add_widget(row_layout)

class MainAPP(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.right_panel = None  # Add this attribute

    def build(self):
        layout = BoxLayout(orientation='horizontal')

        # Left panel content
        left_panel = LeftPanel(size_hint=(0.2, 1))
        layout.add_widget(left_panel)

        # Right panel content
        self.right_panel = RightPanel(size_hint=(0.8, 1))  # Set right_panel attribute
        layout.add_widget(self.right_panel)

        return layout


if __name__ == '__main__':
    # Use multiprocessing to start both Flask and Kivy

    app = MainAPP()
    app.run()

