import numpy
import gradio as gr

from .processing_step import ProcessingStep


class DisplayMasksPanel(ProcessingStep):
    title = "3. Binary Masks"
    output = None

    def create_blocks(self):
        with gr.Row():
            self.dark_mask_image = gr.Image(label="Dark Layer", type="numpy")
            self.mid_mask_image = gr.Image(label="Mid Layer", type="numpy")
            self.light_mask_image = gr.Image(label="Light Layer", type="numpy")

    def process(self, masks):
        dark_mask, mid_mask, light_mask = masks
        return (
            mask_to_image(dark_mask),
            mask_to_image(mid_mask),
            mask_to_image(light_mask),
        )

    def get_outputs(self):
        return [self.dark_mask_image, self.mid_mask_image, self.light_mask_image]


def mask_to_image(boolean_mask: numpy.ndarray) -> numpy.ndarray:
    """Convert a boolean mask to a uint8 image with values 0 or 255."""
    return (boolean_mask * 255).astype(numpy.uint8)
