import gradio as gr

from .processing_step import ProcessingStep


class LoadImagePanel(ProcessingStep):
    title = "1. Load Image"

    def create_blocks(self):
        self.image_upload = gr.Image(type="numpy", label="Upload Image")
        self.output = gr.State()

    def process(self, uploaded_image):
        return uploaded_image[..., :3]

    def get_controls(self):
        return [self.image_upload]
