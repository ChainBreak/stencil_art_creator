import numpy
import gradio as gr

from .processing_step import ProcessingStep


class ThresholdPanel(ProcessingStep):
    title = "2. Threshold into 3 Layers"

    def create_blocks(self):
        self.low_threshold_slider = gr.Slider(0, 255, value=85, label="Low Threshold")
        self.high_threshold_slider = gr.Slider(0, 255, value=170, label="High Threshold")
        self.output = gr.State()

    def process(self, grayscale_image, low_threshold, high_threshold):
        dark_mask = grayscale_image < low_threshold
        mid_mask = (grayscale_image >= low_threshold) & (grayscale_image < high_threshold)
        light_mask = grayscale_image >= high_threshold
        return (dark_mask, mid_mask, light_mask)

    def get_controls(self):
        return [self.low_threshold_slider, self.high_threshold_slider]
