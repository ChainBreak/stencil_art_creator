import numpy
import cv2
from PIL import Image, ImageFilter
import gradio as gr

from .processing_step import ProcessingStep

BLUR_TYPES = ["Gaussian", "Median", "Bilateral"]


class BlurImagePanel(ProcessingStep):
    title = "3. Blur Image"

    def create_blocks(self):
        self.blur_type = gr.Radio(BLUR_TYPES, value="Gaussian", label="Blur Type")
        self.amount_slider = gr.Slider(0, 25, value=3, step=1, label="Blur Amount")
        self.display = gr.Image(label="Blurred Image", type="numpy")
        self.output = gr.State()

    def process(self, image, blur_type, amount):
        blurred = apply_blur(image, blur_type, amount)
        return blurred, blurred

    def get_controls(self):
        return [self.blur_type, self.amount_slider]

    def get_outputs(self):
        return [self.output, self.display]


def apply_blur(image: numpy.ndarray, blur_type: str, amount: int) -> numpy.ndarray:
    if blur_type == "Bilateral":
        # amount drives sigmaColor and sigmaSpace; diameter is derived from it
        diameter = int(amount) * 2 + 1
        return cv2.bilateralFilter(image, d=diameter, sigmaColor=amount * 3, sigmaSpace=amount * 3)
    source = Image.fromarray(image)
    if blur_type == "Median":
        kernel_size = int(amount) * 2 + 1
        blurred = source.filter(ImageFilter.MedianFilter(kernel_size))
    else:
        blurred = source.filter(ImageFilter.GaussianBlur(amount))
    return numpy.array(blurred)
