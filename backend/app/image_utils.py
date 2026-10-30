import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


MAX_IMAGE_PIXELS = 40_000_000
IMAGE_MIME_TYPES = {
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
}
IMAGE_FORMAT_EXTENSIONS = {"PNG": {".png"}, "JPEG": {".jpg", ".jpeg"}}


def validate_image(data: bytes, filename: str, content_type: str | None) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix not in IMAGE_MIME_TYPES:
        raise ValueError("Only PNG, JPG, and JPEG images are accepted.")
    if content_type not in IMAGE_MIME_TYPES[suffix] | {"application/octet-stream"}:
        raise ValueError(f"{suffix[1:].upper()} image has an unsupported content type.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(data))
            if image.format not in IMAGE_FORMAT_EXTENSIONS or suffix not in IMAGE_FORMAT_EXTENSIONS[image.format]:
                raise ValueError("The image file extension does not match its actual format.")
            if image.width < 1 or image.height < 1 or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit.")
            image.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit.") from exc
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Image is corrupt or cannot be decoded.") from exc
    return suffix
