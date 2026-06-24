import click
import gradio as gr

from steps import LoadImagePanel, ThresholdPanel, DisplayMasksPanel


@click.command()
@click.option("--server-port", default=7860, help="Port to serve the Gradio app on.")
def run(server_port: int):
    pipeline_steps = [LoadImagePanel(), ThresholdPanel(), DisplayMasksPanel()]
    demo = build_demo(pipeline_steps)
    demo.launch(server_port=server_port)


def build_demo(pipeline_steps: list) -> gr.Blocks:
    with gr.Blocks(title="Stencil Art Creator") as demo:
        gr.Markdown("# Stencil Art Creator")
        previous_output = None

        for step in pipeline_steps:
            with gr.Group():
                gr.Markdown(f"### {step.title}")
                step.create_blocks()

            upstream = [previous_output] if previous_output is not None else []
            inputs = upstream + step.get_controls()
            triggers = step.get_controls() + (upstream if upstream else [])

            for trigger in triggers:
                trigger.change(step.process, inputs=inputs, outputs=step.get_outputs())

            previous_output = step.output

    return demo


if __name__ == "__main__":
    run()
