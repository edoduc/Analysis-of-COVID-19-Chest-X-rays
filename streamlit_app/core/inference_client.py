import mimetypes
import os

import requests


def predict_from_api(
    image_bytes: bytes,
    mask_bytes: bytes | None = None,
    *,
    image_name: str = "image.png",
    mask_name: str = "mask.png",
) -> dict[str, object]:
    api_url = os.environ.get("INFERENCE_API_URL")
    if not api_url:
        raise RuntimeError("INFERENCE_API_URL n'est pas configurée.")

    image_type = mimetypes.guess_type(image_name)[0] or "application/octet-stream"
    files = {"image": (image_name, image_bytes, image_type)}
    if mask_bytes is not None:
        mask_type = mimetypes.guess_type(mask_name)[0] or "application/octet-stream"
        files["mask"] = (mask_name, mask_bytes, mask_type)

    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        files=files,
        timeout=(3, 60),
    )
    response.raise_for_status()
    return response.json()