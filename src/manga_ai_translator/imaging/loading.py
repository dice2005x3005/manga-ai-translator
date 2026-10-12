"""Reading page images without losing quality.

Pages are decoded to 8-bit RGB for processing, but the original mode, format and color
profile are kept so the final export can match the source (e.g. grayscale stays grayscale).
"""

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

_EXIF_ORIENTATION_TAG = 0x0112
_ROTATED_ORIENTATIONS = frozenset({5, 6, 7, 8})  # orientations that swap width and height
_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N"})
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})
_HASH_CHUNK_BYTES = 1 << 20


class ImageError(Exception):
    """Raised when an image cannot be found, decoded or is not a supported page."""


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    """Width as displayed, i.e. after applying the EXIF orientation."""
    height: int
    mode: str
    format: str


@dataclass(frozen=True)
class LoadedImage:
    image: Image.Image
    """8-bit RGB pixels with the EXIF orientation applied."""
    original_mode: str
    format: str
    icc_profile: bytes | None
    """Color profile that describes ``image`` (None means sRGB)."""


def natural_sort_key(name: str) -> tuple[tuple[str | int, ...], str]:
    """Sort key that orders "p2" before "p10" and ignores case."""
    parts = re.split(r"(\d+)", name.casefold())
    # re.split with a capture group alternates text and digits, so positions never mix types.
    key = tuple(int(part) if i % 2 else part for i, part in enumerate(parts))
    return key, name


def list_images(directory: Path) -> list[Path]:
    """Supported image files directly inside ``directory``, in natural order."""
    if not directory.is_dir():
        raise ImageError(f"Not a folder: {directory}")
    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith(".")
    ]
    return sorted(files, key=lambda path: natural_sort_key(path.name))


def read_image_info(path: Path) -> ImageInfo:
    """Read size, mode and format from the file header without decoding the pixels."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(_EXIF_ORIENTATION_TAG) in _ROTATED_ORIENTATIONS:
                width, height = height, width
            return ImageInfo(path, width, height, image.mode, image.format or "")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ImageError(f"Cannot read image {path}: {exc}") from exc


def load_image(path: Path) -> LoadedImage:
    try:
        with Image.open(path) as image:
            image.load()
            original_mode = image.mode
            image_format = image.format or ""
            icc_profile = image.info.get("icc_profile") or None
            oriented = ImageOps.exif_transpose(image)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ImageError(f"Cannot read image {path}: {exc}") from exc

    if oriented.mode == "CMYK":
        # CMYK pixels are converted to sRGB, so the original profile no longer applies.
        return LoadedImage(_cmyk_to_rgb(oriented, icc_profile), original_mode, image_format, None)
    return LoadedImage(to_rgb(oriented), original_mode, image_format, icc_profile)


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert any Pillow mode to 8-bit RGB, flattening transparency onto white."""
    if image.mode == "RGB":
        return image
    if image.mode in _SIXTEEN_BIT_MODES or image.mode == "I":
        return _high_bit_depth_to_l(image).convert("RGB")
    if image.mode == "CMYK":
        return _cmyk_to_rgb(image, None)
    if image.mode in _ALPHA_MODES or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def _high_bit_depth_to_l(image: Image.Image) -> Image.Image:
    # Pillow clips instead of rescaling when converting 16-bit data to 8-bit, which would
    # turn almost every pixel white, so scale the range explicitly.
    as_int = image.convert("I")
    _, high = as_int.getextrema()
    if image.mode in _SIXTEEN_BIT_MODES or high > 255:
        as_int = as_int.point(lambda value: value * (1 / 257))
    return as_int.convert("L")


def _cmyk_to_rgb(image: Image.Image, icc_profile: bytes | None) -> Image.Image:
    if icc_profile:
        try:
            converted = ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(icc_profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        except (ImageCms.PyCMSError, OSError):
            pass  # broken embedded profile: fall back to Pillow's plain conversion
        else:
            if converted is not None:
                return converted
    return image.convert("RGB")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
