from __lib__ import *


from kivymd.app import MDApp
from kivymd.uix.boxlayout import BoxLayout
from kivymd.uix.slider import MDSlider
from kivymd.uix.textfield import MDTextField
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton

class TimeRangeSelector(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 20
        self.padding = 20

        # START TIME HANDLER
        start_time_box_main = BoxLayout(orientation='vertical', spacing= 10, padding=[0, 10, 0, 10])
        start_time_box = BoxLayout(orientation='horizontal', spacing=10,size_hint=(1, None), height=40)
        start_time_box.add_widget(MDLabel(text="START TIME", size_hint=(0.3, 1)))
        self.start_slider_input = MDTextField(
            hint_text="HH:MM",
            text="00:00",
            size_hint=(0.2, 1),
            mode="rectangle",
        )
        self.start_slider_input.bind(text=self.on_input_change)
        start_time_box.add_widget(self.start_slider_input)
        start_time_box_main.add_widget(start_time_box)

        self.start_slider = MDSlider(min=0, max=1440, value=480, step=1, size_hint=(1, None), height=40)
        self.start_slider.bind(value=self.update_start_time)

        start_time_box_main.add_widget(self.start_slider)
        self.add_widget(start_time_box_main)

        # END TIME HANDLER
        end_time_box_main = BoxLayout(orientation='vertical', spacing=10,  padding=[0, 10, 0, 10])
        end_time_box = BoxLayout(orientation="horizontal", spacing=10,size_hint=(1, None), height=40)
        end_time_box.add_widget(MDLabel(text="END TIME", size_hint=(0.3, 1)))
        self.end_slider_input = MDTextField(
            hint_text="HH:MM",
            text="00:00",
            size_hint=(0.2, 1),
            mode="rectangle",
        )
        self.end_slider_input.bind(text=self.on_input_change)
        end_time_box.add_widget(self.end_slider_input)
        end_time_box_main.add_widget(end_time_box)
        # SLIDER
        self.end_slider = MDSlider(min=0, max=1440, step=1, size_hint=(1, None), height=40)
        self.end_slider.bind(value=self.update_end_time)
        end_time_box_main.add_widget(self.end_slider)

        # TEXT INPUT FIELD

        self.add_widget(end_time_box_main)

    def update_start_time(self, instance, value):
        hours = int(value) // 60
        minutes = int(value) % 60
        self.start_slider_input.text = f"{hours:02d}:{minutes:02d}"

    def update_end_time(self, instance, value):
        hours = int(value) // 60
        minutes = int(value) % 60
        self.end_slider_input.text = f"{hours:02d}:{minutes:02d}"

    def on_input_change(self, instance, text):
        try:
            hours, minutes = map(int, text.split(":"))
            if instance == self.start_slider_input:
                self.start_slider.value = hours * 60 + minutes
            elif instance == self.end_slider_input:
                self.end_slider.value = hours * 60 + minutes
        except ValueError:
            pass


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
