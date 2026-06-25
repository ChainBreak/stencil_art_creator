import numpy
import gradio as gr

from .processing_step import ProcessingStep

CHECKERBOARD_SQUARE_SIZE = 16


class StencilCreatePanel(ProcessingStep):
    title = "5. Stencil Create"
    output = None

    def create_blocks(self):
        self.gallery = gr.Gallery(label="Stencils", columns=3, type="numpy")

    def process(self, selection):
        quantized, colors = selection
        if not colors:
            return []
        checkerboard = make_checkerboard(quantized.shape[0], quantized.shape[1])
        return [build_stencil(quantized, numpy.array(group[0]), checkerboard) for group in colors]

    def get_outputs(self):
        return [self.gallery]


def build_stencil(
    quantized: numpy.ndarray, color: numpy.ndarray, checkerboard: numpy.ndarray
) -> numpy.ndarray:
    """Overlay the given color onto the checkerboard background at pixels matching that color."""
    mask = numpy.all(quantized == color, axis=-1)
    preview = checkerboard.copy()
    preview[mask] = color
    return preview


def make_checkerboard(height: int, width: int) -> numpy.ndarray:
    """Generate an RGB checkerboard of two grey tones."""
    row_parity = numpy.arange(height)[:, None] // CHECKERBOARD_SQUARE_SIZE
    column_parity = numpy.arange(width)[None, :] // CHECKERBOARD_SQUARE_SIZE
    tile = (row_parity + column_parity) % 2
    gray_values = numpy.where(tile == 0, 255, 200).astype(numpy.uint8)
    return numpy.stack([gray_values] * 3, axis=-1)
