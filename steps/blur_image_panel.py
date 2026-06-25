import numpy
import cv2
from PIL import Image, ImageFilter
import gradio as gr

from .processing_step import ProcessingStep


class BlurImagePanel(ProcessingStep):
    title = "3. Blur Image"

    def create_blocks(self):
        self.gaussian_slider = gr.Slider(0, 25, value=0, step=1, label="Gaussian Blur")
        self.median_slider = gr.Slider(0, 25, value=0, step=1, label="Median Blur")
        self.bilateral_slider = gr.Slider(0, 25, value=0, step=1, label="Bilateral Blur")
        self.output = gr.State()

    def process(self, image, gaussian_amount, median_amount, bilateral_amount):
        result = image
        if gaussian_amount > 0:
            result = apply_gaussian(result, gaussian_amount)
        if median_amount > 0:
            result = apply_median(result, median_amount)
        if bilateral_amount > 0:
            result = apply_bilateral(result, bilateral_amount)
        return result

    def get_controls(self):
        return [self.gaussian_slider, self.median_slider, self.bilateral_slider]


def apply_gaussian(image: numpy.ndarray, amount: int) -> numpy.ndarray:
    return numpy.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(amount)))


def apply_median(image: numpy.ndarray, amount: int) -> numpy.ndarray:
    kernel_size = int(amount) * 2 + 1
    return numpy.array(Image.fromarray(image).filter(ImageFilter.MedianFilter(kernel_size)))


def apply_bilateral(image: numpy.ndarray, amount: int) -> numpy.ndarray:
    diameter = int(amount) * 2 + 1
    return cv2.bilateralFilter(image, d=diameter, sigmaColor=amount * 3, sigmaSpace=amount * 3)
