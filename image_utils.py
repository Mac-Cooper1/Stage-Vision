"""
Image utilities for Stager Agent.
Handles image loading, saving, overlays, and transformations.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Default label settings
LABEL_TEXT = "Virtually Staged"
LABEL_FONT_SIZE = 24
LABEL_PADDING = 12
LABEL_MARGIN = 20
LABEL_BG_COLOR = (0, 0, 0, 180)  # Semi-transparent black
LABEL_TEXT_COLOR = (255, 255, 255, 255)  # White


def load_image(path: str | Path) -> Image.Image:
    """
    Load an image from file path.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image object in RGB mode
    """
    path = Path(path)
    img = Image.open(path)
    
    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    logger.debug(f"Loaded image {path}: {img.size}")
    return img


def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Load an image from bytes.
    
    Args:
        data: Image bytes
        
    Returns:
        PIL Image object in RGB mode
    """
    img = Image.open(BytesIO(data))
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    return img


def save_image(img: Image.Image, path: str | Path, quality: int = 95) -> None:
    """
    Save an image to file.
    
    Args:
        img: PIL Image to save
        path: Output file path
        quality: JPEG quality (1-100)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure RGB mode for JPEG
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img.save(path, "JPEG", quality=quality, optimize=True)
    logger.debug(f"Saved image to {path}")


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Get a font for text rendering.
    
    Args:
        size: Font size in pixels
        
    Returns:
        Font object
    """
    # Try common system fonts
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    
    # Fall back to default font
    logger.warning("No system fonts found, using default font")
    return ImageFont.load_default()


def overlay_virtually_staged_label(
    img: Image.Image,
    text: str = LABEL_TEXT,
    font_size: int = LABEL_FONT_SIZE,
    position: str = "bottom-right"
) -> Image.Image:
    """
    Add a "Virtually Staged" label overlay to an image.
    
    Args:
        img: Input PIL Image
        text: Label text
        font_size: Font size in pixels
        position: Label position ("bottom-right", "bottom-left", "top-right", "top-left")
        
    Returns:
        New image with label overlay
    """
    # Scale font size based on image dimensions
    min_dim = min(img.width, img.height)
    scaled_font_size = max(16, int(min_dim * 0.03))
    scaled_padding = max(8, int(scaled_font_size * 0.5))
    scaled_margin = max(10, int(min_dim * 0.02))
    
    # Create a copy to avoid modifying original
    result = img.copy()
    
    # Convert to RGBA for transparency support
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    
    # Create overlay layer
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Get font
    font = get_font(scaled_font_size)
    
    # Calculate text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate label box size
    box_width = text_width + (scaled_padding * 2)
    box_height = text_height + (scaled_padding * 2)
    
    # Calculate position
    if position == "bottom-right":
        x = result.width - box_width - scaled_margin
        y = result.height - box_height - scaled_margin
    elif position == "bottom-left":
        x = scaled_margin
        y = result.height - box_height - scaled_margin
    elif position == "top-right":
        x = result.width - box_width - scaled_margin
        y = scaled_margin
    elif position == "top-left":
        x = scaled_margin
        y = scaled_margin
    else:
        x = result.width - box_width - scaled_margin
        y = result.height - box_height - scaled_margin
    
    # Draw rounded rectangle background
    draw.rounded_rectangle(
        [x, y, x + box_width, y + box_height],
        radius=scaled_padding // 2,
        fill=LABEL_BG_COLOR
    )
    
    # Draw text
    text_x = x + scaled_padding
    text_y = y + scaled_padding - (bbox[1])  # Adjust for text baseline
    draw.text((text_x, text_y), text, font=font, fill=LABEL_TEXT_COLOR)
    
    # Composite overlay onto result
    result = Image.alpha_composite(result, overlay)
    
    # Convert back to RGB for saving as JPEG
    result = result.convert("RGB")
    
    logger.debug(f"Added '{text}' label at {position}")
    return result


def generate_16_9_version(
    input_path: str | Path,
    output_path: str | Path,
    quality: int = 95
) -> None:
    """
    Generate a 16:9 aspect ratio version of an image.
    Uses center-crop if image is wider, or pads with white if narrower.
    
    Args:
        input_path: Path to source image
        output_path: Path for output image
        quality: JPEG quality
    """
    img = load_image(input_path)
    
    target_ratio = 16 / 9
    current_ratio = img.width / img.height
    
    if abs(current_ratio - target_ratio) < 0.01:
        # Already close to 16:9
        save_image(img, output_path, quality)
        return
    
    if current_ratio > target_ratio:
        # Image is wider than 16:9 - crop width
        new_width = int(img.height * target_ratio)
        left = (img.width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img.height))
    else:
        # Image is taller than 16:9 - crop height
        new_height = int(img.width / target_ratio)
        top = (img.height - new_height) // 2
        img = img.crop((0, top, img.width, top + new_height))
    
    save_image(img, output_path, quality)
    logger.info(f"Generated 16:9 version: {output_path}")


def resize_for_upload(
    img: Image.Image,
    max_dimension: int = 2048
) -> Image.Image:
    """
    Resize image if larger than max dimension.
    
    Args:
        img: Input image
        max_dimension: Maximum width or height
        
    Returns:
        Resized image (or original if already small enough)
    """
    if img.width <= max_dimension and img.height <= max_dimension:
        return img
    
    ratio = min(max_dimension / img.width, max_dimension / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    
    return img.resize(new_size, Image.Resampling.LANCZOS)


def get_image_info(path: str | Path) -> dict:
    """
    Get basic information about an image file.
    
    Args:
        path: Path to image file
        
    Returns:
        Dict with width, height, format, mode, size_bytes
    """
    path = Path(path)
    
    with Image.open(path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_bytes": path.stat().st_size
        }
