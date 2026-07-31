import gradio as gr
from PIL import Image
import numpy as np
from predict import predict

def run_prediction(input_image):
    """Handle both canvas draw and file upload."""
    if input_image is None:
        return "No input provided.", {}

    # Gradio canvas returns RGBA numpy array
    if isinstance(input_image, np.ndarray):
        image = Image.fromarray(input_image).convert('RGBA')
        # White canvas → extract alpha as mask
        r, g, b, a = image.split()
        image = Image.merge('RGB', (a, a, a))  # use alpha as intensity
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

app.launch()