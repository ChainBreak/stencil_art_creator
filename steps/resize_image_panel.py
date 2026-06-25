import numpy
from PIL import Image
import gradio as gr

from .processing_step import ProcessingStep


class ResizeImagePanel(ProcessingStep):
    title = "2. Resize Image"

    def create_blocks(self):
        with gr.Row():
            self.width_input = gr.Number(value=1024, label="Width (px)", precision=0)
            self.height_input = gr.Number(value=1024, label="Height (px)", precision=0)
        self.display = gr.Image(label="Resized Image", type="numpy")
        self.output = gr.State()

    def process(self, image, width, height):
        resized = resize_to_fill(image, int(width), int(height))
        return resized, resized

    def get_controls(self):
        return [self.width_input, self.height_input]

    def get_outputs(self):
        return [self.output, self.display]


def resize_to_fill(image: numpy.ndarray, target_width: int, target_height: int) -> numpy.ndarray:
    """Scale the image to cover the target size with LANCZOS, then center-crop."""
    source = Image.fromarray(image)
    scale = max(target_width / source.width, target_height / source.height)
    scaled_width = round(source.width * scale)
    scaled_height = round(source.height * scale)
    scaled = source.resize((scaled_width, scaled_height), Image.LANCZOS)
    left = (scaled.width - target_width) // 2
    top = (scaled.height - target_height) // 2
    return numpy.array(scaled.crop((left, top, left + target_width, top + target_height)))
