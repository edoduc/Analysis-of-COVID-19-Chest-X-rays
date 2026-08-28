from io import BytesIO

import numpy as np
from PIL import Image


def build_gif(steps: dict[str, np.ndarray], duration: int = 800) -> bytes:
    frames = []
    for image_array in steps.values():
        frame = np.clip(image_array, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame, mode="L").convert("RGB"))
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    return output.getvalue()
