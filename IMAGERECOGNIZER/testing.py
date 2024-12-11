# from kivy.app import App
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.gridlayout import GridLayout
# from kivy.uix.button import Button
# from kivy.uix.label import Label
# from kivy.uix.slider import Slider
# from kivy.uix.textinput import TextInput
# from kivy.uix.image import Image
# from kivy.graphics import Color, Rectangle
#
# class TimeSlider(BoxLayout):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.orientation = 'vertical'
#         self.spacing = 0
#         self.padding = 0
#         self.CurrentST = None
#         self.CurrentET = None
#
#         # Start Time Layout
#         input_layout_1 = BoxLayout(orientation='horizontal', spacing=0, size_hint=(1, None), height=50)
#         input_layout_1.add_widget(Label(text="Start Time:", size_hint=(0.2, 1)))
#         self.time_slider_ST = Slider(min=0, max=1440, value=0, step=1, size_hint=(0.4, 1))
#         self.time_slider_ST.bind(value=self.on_slider_change)
#         input_layout_1.add_widget(self.time_slider_ST)
#
#         self.hour_input_ST = TextInput(hint_text="HH", multiline=False, input_filter="int", size_hint=(0.1, 1))
#         self.minute_input_ST = TextInput(hint_text="MM", multiline=False, input_filter="int", size_hint=(0.1, 1))
#         self.hour_input_ST.bind(text=self.on_input_change)
#         self.minute_input_ST.bind(text=self.on_input_change)
#         input_layout_1.add_widget(self.hour_input_ST)
#         input_layout_1.add_widget(self.minute_input_ST)
#
#         self.add_widget(input_layout_1)
#
#         # End Time Layout
#         input_layout_2 = BoxLayout(orientation='horizontal', spacing=0, size_hint=(1, None), height=50)
#
#         input_layout_2.add_widget(Label(text="End Time:", size_hint=(0.2, 1)))
#         self.time_slider_ET = Slider(min=0, max=1440, value=0, step=1, size_hint=(0.4, 1))
#         self.time_slider_ET.bind(value=self.on_slider_change)
#         input_layout_2.add_widget(self.time_slider_ET)
#
#         self.hour_input_ET = TextInput(hint_text="HH", multiline=False, input_filter="int", size_hint=(0.1, 1))
#         self.minute_input_ET = TextInput(hint_text="MM", multiline=False, input_filter="int", size_hint=(0.1, 1))
#         self.hour_input_ET.bind(text=self.on_input_change)
#         self.minute_input_ET.bind(text=self.on_input_change)
#         input_layout_2.add_widget(self.hour_input_ET)
#         input_layout_2.add_widget(self.minute_input_ET)
#
#         self.add_widget(input_layout_2)
#
#     def on_slider_change(self, instance, value):
#         """Update time label and inputs based on slider value."""
#         total_minutes = int(value)
#         hours = total_minutes // 60
#         minutes = total_minutes % 60
#         if instance == self.time_slider_ST:
#             self.hour_input_ST.text = f"{hours:02d}"
#             self.minute_input_ST.text = f"{minutes:02d}"
#             self.CurrentST = f"{self.hour_input_ET}:{self.minute_input_ET}"
#         elif instance == self.time_slider_ET:
#             self.hour_input_ET.text = f"{hours:02d}"
#             self.minute_input_ET.text = f"{minutes:02d}"
#             self.CurrentET = f"{self.hour_input_ET}:{self.minute_input_ET}"
#
#     def on_input_change(self, instance, value):
#         """Update slider value based on text inputs."""
#         try:
#             if instance in (self.hour_input_ST, self.minute_input_ST):
#                 hours = int(self.hour_input_ST.text or "0")
#                 minutes = int(self.minute_input_ST.text or "0")
#                 self.time_slider_ST.value = hours * 60 + minutes
#
#             elif instance in (self.hour_input_ET, self.minute_input_ET):
#                 hours = int(self.hour_input_ET.text or "0")
#                 minutes = int(self.minute_input_ET.text or "0")
#                 self.time_slider_ET.value = hours * 60 + minutes
#
#         except ValueError:
#             pass
#
#
# class LeftPanel(BoxLayout):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.orientation = 'vertical'
#         self.padding = 10
#         self.spacing = 10
#         self.NMatches = 5
#         self.STime = None
#         self.ETime = None
#
#         # Set a background color for the left panel
#         with self.canvas.before:
#             Color(0.8, 0.2, 0.2, 1)
#             self.rect = Rectangle(size=self.size, pos=self.pos)
#             self.bind(size=self.update_rect, pos=self.update_rect)
#
#         # Add TimeSlider widget
#         self.add_widget(TimeSlider())
#
#         # Submit button
#         self.submit_button = Button(text="Find Matches", size_hint=(1, None), height=40)
#         self.submit_button.bind(self.SeqMatcher(NMatches, STime, ETime))
#         self.add_widget(self.submit_button)
#
#     def update_rect(self, *args):
#         self.rect.size = self.size
#         self.rect.pos = self.pos
#
#
# class RightPanel(GridLayout):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.cols = 3
#         self.padding = 10
#         self.spacing = 10
#
#         # Set a background color for the right panel
#         with self.canvas.before:
#             Color(0.2, 0.2, 0.8, 1)
#             self.rect = Rectangle(size=self.size, pos=self.pos)
#             self.bind(size=self.update_rect, pos=self.update_rect)
#
#         # Add headers for the table
#         self.add_widget(Label(text="Test Image", size_hint_x=None, width=200))
#         self.add_widget(Label(text="Matching Images", size_hint_x=None, width=200))
#         self.add_widget(Label(text="Full Chart", size_hint_x=None, width=400))
#
#         # Example of adding rows dynamically
#         for i in range(5):
#             self.add_widget(Image(source="test_image_placeholder.png", size_hint_x=None, width=200))
#             self.add_widget(Image(source="match_image_placeholder.png", size_hint_x=None, width=200))
#             self.add_widget(Image(source="full_chart_placeholder.png", size_hint_x=None, width=400))
#
#     def update_rect(self, *args):
#         self.rect.size = self.size
#         self.rect.pos = self.pos
#
#
# class MainApp(App):
#     def build(self):
#         layout = BoxLayout(orientation='horizontal')
#
#         # Left panel content
#         left_panel = LeftPanel(size_hint=(0.2, 1))
#         layout.add_widget(left_panel)
#
#         # Right panel content
#         right_panel = RightPanel(size_hint=(0.8, 1))
#         layout.add_widget(right_panel)
#
#         return layout
#
#
# if __name__ == '__main__':
#     MainApp().run()


from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.slider import MDSlider
from kivymd.uix.textfield import MDTextField

class TimeRangeSelector(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 20
        self.padding = 20

        # Header Label
        self.add_widget(MDLabel(
            text="Select Time Range",
            halign="center",
            theme_text_color="Primary",
            font_style="H5"
        ))

        # Start Time Section
        start_time_box = BoxLayout(orientation='horizontal', spacing=10)
        start_time_box.add_widget(MDLabel(text="Start Time:", size_hint=(0.2, 1)))
        self.start_slider = MDSlider(min=0, max=1440, value=480, step=1, size_hint=(0.6, 1))
        self.start_slider.bind(value=self.update_start_time)
        start_time_box.add_widget(self.start_slider)
        self.start_time_input = MDTextField(
            hint_text="HH:MM",
            text="08:00",
            size_hint=(0.2, 1),
            mode="rectangle"
        )
        self.start_time_input.bind(text=self.on_input_change)
        start_time_box.add_widget(self.start_time_input)
        self.add_widget(start_time_box)

        # End Time Section
        end_time_box = BoxLayout(orientation='horizontal', spacing=10)
        end_time_box.add_widget(MDLabel(text="End Time:", size_hint=(0.2, 1)))
        self.end_slider = MDSlider(min=0, max=1440, value=1020, step=1, size_hint=(0.6, 1))
        self.end_slider.bind(value=self.update_end_time)
        end_time_box.add_widget(self.end_slider)
        self.end_time_input = MDTextField(
            hint_text="HH:MM",
            text="17:00",
            size_hint=(0.2, 1),
            mode="rectangle"
        )
        self.end_time_input.bind(text=self.on_input_change)
        end_time_box.add_widget(self.end_time_input)
        self.add_widget(end_time_box)

    def update_start_time(self, instance, value):
        """Update start time input when slider changes."""
        hours = int(value) // 60
        minutes = int(value) % 60
        self.start_time_input.text = f"{hours:02d}:{minutes:02d}"

    def update_end_time(self, instance, value):
        """Update end time input when slider changes."""
        hours = int(value) // 60
        minutes = int(value) % 60
        self.end_time_input.text = f"{hours:02d}:{minutes:02d}"

    def on_input_change(self, instance, text):
        """Update slider position when input changes."""
        try:
            hours, minutes = map(int, text.split(":"))
            if instance == self.start_time_input:
                self.start_slider.value = hours * 60 + minutes
            elif instance == self.end_time_input:
                self.end_slider.value = hours * 60 + minutes
        except ValueError:
            pass

class LeftPanel(BoxLayout):
    def __init__(self,**kwargs):
        super().__init__(self, **kwargs)
        self.orientation = "vertical"
        self.padding = 10
        self.spacing = 10
class ModernApp(MDApp):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        layout.add_widget(TimeRangeSelector(size_hint=(1, 0.5)))

        # Add a submit button
        layout.add_widget(MDRaisedButton(
            text="Find Matches",
            pos_hint={"center_x": 0.5},
            size_hint=(0.5, None),
            height=40
        ))

        return layout


if __name__ == "__main__":
    ModernApp().run()
