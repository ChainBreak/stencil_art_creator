import numpy
import cv2
import gradio as gr

from .processing_step import ProcessingStep

CHECKERBOARD_SQUARE_SIZE = 16


class StencilCreatePanel(ProcessingStep):
    title = "5. Stencil Create"
    output = None

    def create_blocks(self):
        self.gallery = gr.Gallery(label="Stencils", columns=3, type="numpy")

    def process(self, selection):
        quantized, groups = selection
        if not groups:
            return []
        checkerboard = make_checkerboard(quantized.shape[0], quantized.shape[1])
        gallery_items = []
        for group in groups:
            color = numpy.array(group[0])
            for stencil_mask in build_stencil(quantized, color):
                gallery_items.append(display_stencil(stencil_mask, color, checkerboard))
        return gallery_items

    def get_outputs(self):
        return [self.gallery]


def build_stencil(quantized: numpy.ndarray, color: numpy.ndarray) -> list:
    """Return one physically makeable stencil mask per solid contour level.

    Uses the full contour hierarchy (RETR_TREE) to determine which regions are
    solid material vs holes. Counting from the outside in, odd levels (1, 3, 5…
    in 1-indexed terms) are solid material; even levels are holes. Each odd-level
    contour is drawn filled to produce a single cuttable stencil.
    """
    color_mask = numpy.all(quantized == color, axis=-1).astype(numpy.uint8)
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    contour_depths = compute_contour_depths(hierarchy[0])

    stencil_mask_by_depth = {}
    for contour, depth in zip(contours, contour_depths):
        if depth % 2 == 0:  # even 0-indexed depth = odd 1-indexed level = solid material
            if depth not in stencil_mask_by_depth:
                stencil_mask_by_depth[depth] = numpy.zeros(color_mask.shape, dtype=numpy.uint8)
            cv2.drawContours(stencil_mask_by_depth[depth], [contour], contourIdx=0, color=1, thickness=cv2.FILLED)

    return [mask.astype(bool) for mask in stencil_mask_by_depth.values()]


def compute_contour_depths(hierarchy: numpy.ndarray) -> list:
    """Return the nesting depth of each contour (outermost contours = depth 0).

    The hierarchy array has shape (N, 4): [next, prev, first_child, parent].
    Depth is found by walking up the parent chain.
    """
    depths = []
    for i in range(len(hierarchy)):
        depth = 0
        parent = hierarchy[i][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[parent][3]
        depths.append(depth)
    return depths


def display_stencil(
    mask: numpy.ndarray, color: numpy.ndarray, checkerboard: numpy.ndarray
) -> numpy.ndarray:
    """Render a binary stencil mask as the chosen color over a checkerboard background."""
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
