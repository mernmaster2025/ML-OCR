import time

import gradio as gr
from PIL import Image, UnidentifiedImageError
import numpy as np
from predict import predict


# Gradio decodes the canvas PNG straight off disk, and on a fast click it can
# get there before the browser has finished writing the file — surfacing as
# UnidentifiedImageError inside preprocess, out of reach of the event handler.
# Retry the read briefly instead of dropping the stroke.
#
# Patched onto the stock class rather than a subclass on purpose: Gradio keys
# frontend assets off the component's class name and module, so any subclass
# reads as a third-party custom component and the UI hangs fetching bundles
# that do not exist.
_convert_image = gr.ImageEditor.convert_and_format_image


def _convert_image_with_retry(self, file):
    for attempt in range(5):
        try:
            return _convert_image(self, file)
        except (UnidentifiedImageError, OSError):
            if attempt == 4:
                raise
            time.sleep(0.1)


gr.ImageEditor.convert_and_format_image = _convert_image_with_retry


def run_prediction(input_image):
    """Handle both canvas draw and file upload."""
    if input_image is None:
        return "No input provided.", {}

    # Sketchpad returns {'background', 'layers', 'composite'} — the
    # composite is the flattened drawing.
    if isinstance(input_image, dict):
        input_image = input_image.get('composite')
        if input_image is None:
            return "No input provided.", {}

    if isinstance(input_image, np.ndarray):
        image = Image.fromarray(input_image)
        # Strokes sit on a transparent background → flatten onto white
        # so the canvas path matches an uploaded image.
        if image.mode == 'RGBA':
            flat = Image.new('RGB', image.size, 'white')
            flat.paste(image, mask=image.split()[3])
            image = flat
        else:
            image = image.convert('RGB')
    else:
        image = input_image

    results = predict(image, top_k=5)

    top = results[0]
    label = f"Predicted: '{top['character']}'  ({top['confidence']} confidence)"
    bar_data = {r['character']: r['prob'] for r in results}

    return label, bar_data


with gr.Blocks(title="OCR — Alphanumeric Character Recognition") as app:
    gr.Markdown("## OCR — Digit & Letter Recognition")
    gr.Markdown("Draw a character or upload an image. Supports 0–9, A–Z, a–z.")

    with gr.Row():
        with gr.Column():
            canvas = gr.Sketchpad(
                label="Draw here",
                brush=gr.Brush(default_size=18, colors=["#000000"]),
                canvas_size=(280, 280),
            )
            upload = gr.Image(
                label="Or upload image",
                type="numpy",
                image_mode="RGB"
            )
            btn = gr.Button("Recognize", variant="primary")

        with gr.Column():
            result_label = gr.Textbox(label="Result", interactive=False)
            confidence_bar = gr.Label(
                label="Top 5 predictions",
                num_top_classes=5
            )

    btn.click(
        fn=run_prediction,
        inputs=[canvas],
        outputs=[result_label, confidence_bar]
    )
    upload.change(
        fn=run_prediction,
        inputs=[upload],
        outputs=[result_label, confidence_bar]
    )

if __name__ == '__main__':
    app.launch()