import numpy
import gradio as gr

from .processing_step import ProcessingStep


class LoadImagePanel(ProcessingStep):
    title = "1. Load Image"

    def create_blocks(self):
        self.image_upload = gr.Image(type="numpy", label="Upload Image")
        self.output = gr.State()

    def process(self, uploaded_image):
        grayscale_image = convert_to_grayscale(uploaded_image)
        return grayscale_image

    def get_controls(self):
        return [self.image_upload]


def convert_to_grayscale(rgb_image: numpy.ndarray) -> numpy.ndarray:
    """Convert an RGB image to grayscale using luminance weights."""
    luminance_weights = numpy.array([0.2126, 0.7152, 0.0722])
    grayscale_image = numpy.dot(rgb_image[..., :3], luminance_weights)
    return grayscale_image.astype(numpy.uint8)
