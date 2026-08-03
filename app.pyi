
class Sketchpad(gr.Sketchpad):
    """Gradio decodes the canvas PNG straight off disk, and on a fast click it
    can get there before the browser has finished writing the file — which
    surfaces as UnidentifiedImageError inside preprocess, out of reach of the
    event handler. Retry the read briefly instead of dropping the stroke."""

    def get_block_name(self) -> str:
        # gr.Sketchpad sets is_template, so the default implementation resolves
        # a subclass to its __base__ ("sketchpad") — not a real Svelte
        # component, which leaves the UI loading forever. Name it explicitly.
        return "imageeditor"

    def convert_and_format_image(self, file):
        for attempt in range(5):
            try:
                return super().convert_and_format_image(file)
            except (UnidentifiedImageError, OSError):
                if attempt == 4:
                    raise
                time.sleep(0.1)
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component

    