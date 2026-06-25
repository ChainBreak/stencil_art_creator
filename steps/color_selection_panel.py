import numpy
from PIL import Image, ImageFilter
import gradio as gr

from .processing_step import ProcessingStep

SWATCH_SIZE = 40
ACTIVE_BORDER_WIDTH = 4


class ColorSelectionPanel(ProcessingStep):
    title = "4. Colour Selection"

    def create_blocks(self):
        self.clickable_image = gr.Image(type="numpy", sources=[], label="Click image to pick colours")
        self.add_button = gr.Button("Add stencil colour")
        with gr.Row():
            self.stencil_groups_display = gr.Image(label="Stencil groups", type="numpy")
            self.quantized_display = gr.Image(label="Simplified image", type="numpy")
        self.clear_button = gr.Button("Clear all")
        self.stencil_groups = gr.State([])
        self.active_group_index = gr.State(None)
        self.output = gr.State()

        self.add_button.click(
            add_stencil_group,
            inputs=[self.stencil_groups],
            outputs=[self.stencil_groups, self.active_group_index],
        )
        self.clickable_image.select(
            pick_color,
            inputs=[self.clickable_image, self.stencil_groups, self.active_group_index],
            outputs=[self.stencil_groups, self.stencil_groups_display],
        )
        self.clear_button.click(
            lambda: ([], None, None),
            inputs=None,
            outputs=[self.stencil_groups, self.active_group_index, self.stencil_groups_display],
        )

    def process(self, blurred_image, stencil_groups):
        non_empty_groups = [group for group in stencil_groups if group]
        if not non_empty_groups:
            return (blurred_image, non_empty_groups), blurred_image, None

        source_palette, output_palette = build_palettes(non_empty_groups)
        quantized = quantize_to_grouped_palette(blurred_image, source_palette, output_palette)
        return (quantized, non_empty_groups), blurred_image, quantized

    def get_controls(self):
        return [self.stencil_groups]

    def get_outputs(self):
        return [self.output, self.clickable_image, self.quantized_display]


def add_stencil_group(groups: list):
    new_groups = groups + [[]]
    return new_groups, len(new_groups) - 1


def pick_color(image: numpy.ndarray, groups: list, active_index, event: gr.SelectData):
    if active_index is None:
        return groups, render_groups(groups, active_index)
    x, y = event.index
    color = image[y, x][:3].tolist()
    updated_groups = [group[:] for group in groups]
    updated_groups[active_index] = updated_groups[active_index] + [color]
    return updated_groups, render_groups(updated_groups, active_index)


def build_palettes(groups: list):
    """Flatten groups into parallel source and output colour arrays.
    Every source colour maps to its group's first colour (the output colour)."""
    source_colors = []
    output_colors = []
    for group in groups:
        output_color = group[0]
        for source_color in group:
            source_colors.append(source_color)
            output_colors.append(output_color)
    return (
        numpy.array(source_colors, dtype=numpy.uint8),
        numpy.array(output_colors, dtype=numpy.uint8),
    )


def quantize_to_grouped_palette(
    image: numpy.ndarray,
    source_palette: numpy.ndarray,
    output_palette: numpy.ndarray,
) -> numpy.ndarray:
    """Find nearest source colour per pixel in Lab space, then output the
    corresponding group output colour. Mode filter cleans stray edge pixels."""
    image_lab = rgb_to_lab(image)
    source_lab = rgb_to_lab(source_palette[None])[0]
    distances = (
        (image_lab[:, :, None, :].astype(int) - source_lab[None, None, :, :]) ** 2
    ).sum(axis=-1)
    closest_indices = distances.argmin(axis=-1)
    quantized = output_palette[closest_indices]
    return numpy.array(Image.fromarray(quantized).filter(ImageFilter.ModeFilter(size=5)))


def render_groups(groups: list, active_index) -> numpy.ndarray:
    """Build a swatch image: one row per group, active group highlighted in white."""
    if not groups:
        return numpy.full((SWATCH_SIZE, SWATCH_SIZE, 3), 240, dtype=numpy.uint8)

    row_height = SWATCH_SIZE + ACTIVE_BORDER_WIDTH * 2
    max_swatches = max(len(group) for group in groups)
    row_width = max(max_swatches, 1) * SWATCH_SIZE + ACTIVE_BORDER_WIDTH * 2

    canvas = numpy.full((row_height * len(groups), row_width, 3), 240, dtype=numpy.uint8)

    for group_index, group in enumerate(groups):
        row_top = group_index * row_height
        row_bottom = row_top + row_height

        if group_index == active_index:
            canvas[row_top:row_bottom, :] = [255, 255, 255]

        inner_top = row_top + ACTIVE_BORDER_WIDTH
        inner_bottom = inner_top + SWATCH_SIZE

        for swatch_index, color in enumerate(group):
            left = ACTIVE_BORDER_WIDTH + swatch_index * SWATCH_SIZE
            canvas[inner_top:inner_bottom, left:left + SWATCH_SIZE] = color

    return canvas


def rgb_to_lab(rgb_image: numpy.ndarray) -> numpy.ndarray:
    return numpy.array(Image.fromarray(rgb_image).convert("LAB"))
