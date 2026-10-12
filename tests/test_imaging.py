import hashlib
from pathlib import Path

import pytest
from PIL import Image

from manga_ai_translator.imaging import (
    ImageError,
    list_images,
    load_image,
    natural_sort_key,
    read_image_info,
    sha256_file,
)

RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


def save(image: Image.Image, path: Path, **params: object) -> Path:
    image.save(path, **params)
    return path


def test_natural_sort_orders_numbers_by_value_and_ignores_case() -> None:
    names = ["p10.png", "p2.png", "P1.png", "p1b.png", "cover.png", "p02.png"]
    assert sorted(names, key=natural_sort_key) == [
        "cover.png",
        "P1.png",
        "p1b.png",
        "p02.png",
        "p2.png",
        "p10.png",
    ]


def test_natural_sort_handles_names_with_and_without_numbers() -> None:
    assert sorted(["a1", "a", "b", "a10b2"], key=natural_sort_key) == ["a", "a1", "a10b2", "b"]


def test_list_images_filters_and_sorts(tmp_path: Path) -> None:
    for name in ["10.JPG", "2.png", "1.webp", "notes.txt", ".hidden.png", "3.jpeg"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "4.png").mkdir()
    assert [p.name for p in list_images(tmp_path)] == ["1.webp", "2.png", "3.jpeg", "10.JPG"]


def test_list_images_requires_a_folder(tmp_path: Path) -> None:
    with pytest.raises(ImageError, match="Not a folder"):
        list_images(tmp_path / "missing")


def test_read_image_info(tmp_path: Path) -> None:
    path = save(Image.new("L", (30, 40)), tmp_path / "page.png")
    info = read_image_info(path)
    assert (info.width, info.height, info.mode, info.format) == (30, 40, "L", "PNG")


def rotated_exif() -> Image.Exif:
    exif = Image.Exif()
    exif[0x0112] = 6  # display rotated 90 degrees clockwise
    return exif


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), RED)
    image.putpixel((1, 0), BLUE)
    path = save(image, tmp_path / "rotated.png", exif=rotated_exif())

    info = read_image_info(path)
    assert (info.width, info.height) == (1, 2)
    loaded = load_image(path).image
    assert loaded.size == (1, 2)
    assert loaded.getpixel((0, 0)) == RED
    assert loaded.getpixel((0, 1)) == BLUE


@pytest.mark.parametrize(
    ("mode", "color"),
    [("RGB", RED), ("L", 128), ("1", 1), ("P", 0)],
)
def test_opaque_modes_become_rgb(tmp_path: Path, mode: str, color: object) -> None:
    path = save(Image.new(mode, (8, 6), color), tmp_path / "page.png")  # type: ignore[arg-type]
    loaded = load_image(path)
    assert loaded.image.mode == "RGB"
    assert loaded.image.size == (8, 6)
    assert loaded.original_mode == mode
    assert loaded.format == "PNG"


@pytest.mark.parametrize("mode", ["RGBA", "LA"])
def test_transparency_is_flattened_onto_white(tmp_path: Path, mode: str) -> None:
    color = (0, 0, 0, 0) if mode == "RGBA" else (0, 0)
    path = save(Image.new(mode, (4, 4), color), tmp_path / "page.png")
    assert load_image(path).image.getpixel((0, 0)) == WHITE


def test_palette_transparency_is_flattened_onto_white(tmp_path: Path) -> None:
    image = Image.new("P", (4, 4), 0)
    image.putpalette([0, 0, 0] * 256)
    path = save(image, tmp_path / "page.png", transparency=0)
    assert load_image(path).image.getpixel((0, 0)) == WHITE


def test_sixteen_bit_grayscale_is_scaled_not_clipped(tmp_path: Path) -> None:
    image = Image.new("I;16", (3, 1))
    for x, value in enumerate([0, 32896, 65535]):
        image.putpixel((x, 0), value)
    path = save(image, tmp_path / "deep.png")

    loaded = load_image(path).image
    assert [loaded.getpixel((x, 0)) for x in range(3)] == [(0, 0, 0), (128, 128, 128), WHITE]


def test_cmyk_jpeg_becomes_rgb(tmp_path: Path) -> None:
    path = save(Image.new("CMYK", (4, 4), (0, 0, 0, 0)), tmp_path / "print.jpg")
    loaded = load_image(path)
    assert loaded.original_mode == "CMYK"
    assert loaded.icc_profile is None
    assert all(channel >= 250 for channel in loaded.image.getpixel((0, 0)))


def test_webp_is_supported(tmp_path: Path) -> None:
    path = save(Image.new("RGB", (5, 5), BLUE), tmp_path / "page.webp", lossless=True)
    loaded = load_image(path)
    assert loaded.format == "WEBP"
    assert loaded.image.getpixel((2, 2)) == BLUE


def test_icc_profile_of_rgb_image_is_kept(tmp_path: Path) -> None:
    from PIL import ImageCms

    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = save(Image.new("RGB", (2, 2)), tmp_path / "p.png", icc_profile=profile)
    assert load_image(path).icc_profile == profile


def test_corrupt_file_raises_image_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"definitely not a png")
    with pytest.raises(ImageError, match="Cannot read image"):
        load_image(path)
    with pytest.raises(ImageError):
        read_image_info(path)


def test_sha256_file(tmp_path: Path) -> None:
    data = b"x" * 3_000_000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()
