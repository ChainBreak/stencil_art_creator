import gradio as gr


class ProcessingStep:
    title: str = ""

    def create_blocks(self):
        """Build this panel's UI inside a gr.Blocks context.

        Implementations must assign self.output to a gr.State holding the
        step's output value. Additional UI components (controls, displays)
        are also created here.
        """

    def process(self, *inputs):
        """Transform the upstream output and any control values into this step's output."""

    def get_controls(self) -> list:
        """Return the list of Gradio components whose changes should trigger this step."""
        return []

    def get_outputs(self) -> list:
        """Return the list of Gradio components that process() writes to."""
        return [self.output]
