"""
Nano Banana Client (gemini-2.5-flash-image) for virtual staging.

Uses Gemini's image generation/editing capabilities to virtually stage real estate photos.
Philosophy: "Stage this property to look move-in ready and professionally designed."

Output: Dynamic aspect ratio matching the input image to prevent hallucinations.
"""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import httpx
from PIL import Image

from config import get_settings

logger = logging.getLogger(__name__)


# Gemini 3 Pro Image Preview aspect ratio/size table (excluding 21:9)
# Format: aspect_ratio_str -> {size_str: (width, height)}
GEMINI_IMAGE_CONFIGS = {
    "1:1": {
        "1K": (1024, 1024),
        "2K": (2048, 2048),
        "4K": (4096, 4096),
    },
    "2:3": {
        "1K": (848, 1264),
        "2K": (1696, 2528),
        "4K": (3392, 5056),
    },
    "3:2": {
        "1K": (1264, 848),
        "2K": (2528, 1696),
        "4K": (5056, 3392),
    },
    "3:4": {
        "1K": (896, 1200),
        "2K": (1792, 2400),
        "4K": (3584, 4800),
    },
    "4:3": {
        "1K": (1200, 896),
        "2K": (2400, 1792),
        "4K": (4800, 3584),
    },
    "4:5": {
        "1K": (928, 1152),
        "2K": (1856, 2304),
        "4K": (3712, 4608),
    },
    "5:4": {
        "1K": (1152, 928),
        "2K": (2304, 1856),
        "4K": (4608, 3712),
    },
    "9:16": {
        "1K": (768, 1376),
        "2K": (1536, 2752),
        "4K": (3072, 5504),
    },
    "16:9": {
        "1K": (1376, 768),
        "2K": (2752, 1536),
        "4K": (5504, 3072),
    },
    # Note: 21:9 intentionally excluded - too wide for MLS use case
}


def choose_gemini_image_config(width: int, height: int) -> Tuple[str, str]:
    """
    Given the input image dimensions, return (aspect_ratio_str, image_size_str)
    for gemini-3-pro-image-preview that best matches the original.

    Args:
        width: Input image width in pixels
        height: Input image height in pixels

    Returns:
        Tuple of (aspect_ratio_str, image_size_str) where:
        - aspect_ratio_str: one of "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9"
        - image_size_str: one of "1K", "2K", "4K"

    Note:
        21:9 is never returned (too wide for MLS use case).
        When scores are tied, 2K is preferred for balance of quality and cost.
    """
    input_ar = width / height
    long_input = max(width, height)

    best_score = float('inf')
    best_config = ("16:9", "2K")  # Fallback default

    for aspect_ratio_str, sizes in GEMINI_IMAGE_CONFIGS.items():
        for size_str, (w, h) in sizes.items():
            # Calculate aspect ratio difference
            candidate_ar = w / h
            ar_diff = abs(candidate_ar - input_ar)

            # Calculate size difference (normalized)
            long_candidate = max(w, h)
            size_diff = abs(long_candidate - long_input) / max(long_input, 1)

            # Score: prioritize aspect ratio matching, then size
            # AR difference weighted 2x to make it dominant
            score = ar_diff * 2.0 + size_diff

            # Slight preference for 2K when scores are very close
            # (better balance of quality vs cost/latency)
            if size_str == "2K":
                score -= 0.001

            if score < best_score:
                best_score = score
                best_config = (aspect_ratio_str, size_str)

    logger.debug(
        f"Input {width}x{height} (AR={input_ar:.3f}) -> {best_config[0]} @ {best_config[1]} "
        f"(score={best_score:.4f})"
    )

    return best_config


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Get the dimensions of an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


class NanoBananaClient:
    """
    Client for Gemini image generation model (gemini-2.5-flash-image / Nano Banana).
    Generates virtually staged room images from base photos and prompts.

    Supports full virtual staging (adding furniture to vacant rooms) and
    declutter/enhancement for occupied rooms.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Nano Banana client.
        
        Args:
            api_key: Google API key. Uses config if not provided.
            base_url: Base URL for API. Uses config if not provided.
        """
        settings = get_settings()
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.base_url = base_url or settings.GEMINI_API_BASE_URL
        self.model = settings.GEMINI_IMAGE_MODEL
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_retries = settings.MAX_RETRIES
        
        logger.info(f"NanoBananaClient initialized with model: {self.model}")
    
    async def stage_image(
        self,
        base_image_path: Path,
        prompt_text: str,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None
    ) -> bytes:
        """
        Generate a virtually staged version of the input image.

        Args:
            base_image_path: Path to the source image
            prompt_text: Virtual staging prompt (full furniture staging or declutter)
            aspect_ratio: Output aspect ratio (auto-detected from input if not specified)
            image_size: Output size "1K", "2K", or "4K" (auto-detected from input if not specified)

        Returns:
            Generated staged image as bytes

        Raises:
            ValueError: If no image is returned
            httpx.HTTPError: If API request fails
        """
        # Get input image dimensions and choose optimal config
        width, height = get_image_dimensions(base_image_path)

        if aspect_ratio is None or image_size is None:
            auto_ar, auto_size = choose_gemini_image_config(width, height)
            aspect_ratio = aspect_ratio or auto_ar
            image_size = image_size or auto_size

        logger.info(f"Input image: {width}x{height} -> Output config: {aspect_ratio} @ {image_size}")

        # Read and encode base image
        image_bytes = base_image_path.read_bytes()
        image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Determine mime type
        suffix = base_image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        # Build request body for image editing
        request_body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size
                }
            }
        }
        
        url = f"{self.base_url}/models/{self.model}:generateContent"

        # Build a simplified fallback prompt for retries
        fallback_prompt = self._build_fallback_prompt(prompt_text)

        last_error = None
        last_response = None

        for attempt in range(self.max_retries):
            # Linear backoff: 0s, 1s, 2s, 3s, 4s, 5s between attempts
            if attempt > 0:
                backoff_seconds = attempt  # 1, 2, 3, 4, 5...
                logger.info(f"Waiting {backoff_seconds}s before retry...")
                await asyncio.sleep(backoff_seconds)

            # Use simplified prompt on later attempts if original failed
            current_prompt = prompt_text if attempt == 0 else fallback_prompt
            current_body = request_body.copy()
            current_body["contents"] = [
                {
                    "role": "user",
                    "parts": [
                        {"text": current_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if attempt > 0:
                        logger.info(f"Sending staging request (attempt {attempt + 1}/{self.max_retries}) with simplified prompt")
                    else:
                        logger.info(f"Sending staging request (attempt {attempt + 1}/{self.max_retries})")

                    response = await client.post(
                        url,
                        headers={
                            "x-goog-api-key": self.api_key,
                            "Content-Type": "application/json",
                        },
                        json=current_body
                    )
                    response.raise_for_status()

                    result = response.json()
                    last_response = result

                # Extract image from response
                image_data = self._extract_image_from_response(result)
                if image_data:
                    logger.info("Successfully generated staged image")
                    return image_data

                raise ValueError("No image data in response")

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e.response.status_code}")
                if e.response.status_code >= 500:
                    continue  # Retry on server errors
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    continue

        # Log full response on final failure for debugging
        if last_response:
            logger.error(f"Final failed response: {json.dumps(last_response, indent=2)[:2000]}")

        raise last_error or ValueError("Failed to generate staged image after retries")

    def _build_fallback_prompt(self, original_prompt: str) -> str:
        """
        Build a simplified fallback prompt for retry attempts.

        When the full prompt fails, we try a simpler version that focuses
        on the core task without extensive constraints.
        """
        # Extract the room type from the original prompt
        room_type = "room"
        for rt in ["kitchen", "bathroom", "bedroom", "living room", "dining room", "exterior", "hallway", "office"]:
            if rt in original_prompt.lower():
                room_type = rt
                break

        # Detect if this is a vacant room needing staging or occupied room needing declutter
        is_vacant = "stage this empty" in original_prompt.lower() or "vacant" in original_prompt.lower()

        # Detect style preference from original prompt (matches the 7 client-facing styles)
        style = "neutral"
        for s in ["traditional", "farmhouse", "coastal", "modern", "luxury", "neoclassical"]:
            if s in original_prompt.lower():
                style = s
                break

        if is_vacant:
            # Staging fallback for vacant rooms
            furniture_by_room = {
                "bedroom": "a queen bed with headboard (standard size, not oversized), matching nightstands with lamps, and an area rug under the bed",
                "living room": "a sofa (sized appropriately for the room), coffee table, accent chairs, area rug, and floor lamp",
                "dining room": "a dining table with chairs (scaled to room size), area rug, and simple centerpiece",
                "office": "a desk, desk chair, and small bookshelf",
                "kitchen": "bar stools at the island if present, and minimal counter accessories",
                "bathroom": "neatly rolled towels and a small plant",
                "hallway": "a small console table and mirror if space allows",
                "exterior": "outdoor seating on the porch if present",
                "room": "appropriately sized furniture for the space"
            }
            furniture = furniture_by_room.get(room_type, furniture_by_room["room"])

            return f"""VIRTUAL STAGING TASK: Stage this empty {room_type} photo for a real estate listing in a {style} style.

KEEP ARCHITECTURE UNCHANGED: Keep the exact same layout, walls, flooring, windows, ceiling, and all architectural features from the original photo. Do NOT move walls, change flooring, or alter room dimensions.

Add realistically scaled furniture: {furniture}. Include tasteful decor like plants and art that match the {style} style. All furniture must be properly sized for this specific room - do NOT use oversized furniture to fake room size.

CRITICAL: Do NOT place any furniture, rugs, or decor to cover or hide any visible damage, stains, cracks, or wear on walls, floors, or ceiling. All defects must remain fully visible.

Level the photo so vertical lines are truly vertical. Do NOT move camera horizontally or make the room appear larger.

Apply professional photo enhancement: correct exposure, fix white balance, reduce haze. Result must be photorealistic."""

        else:
            # Declutter fallback for occupied rooms
            return f"""VIRTUAL STYLING TASK: Clean up and enhance this {room_type} photo for a real estate listing.

KEEP EVERYTHING UNCHANGED: Keep the exact same layout, walls, flooring, ceiling, and ALL major furniture exactly where it is. Do NOT remove or replace any furniture pieces.

Remove only loose clutter, trash, and personal items to make the space look tidy. You may add ONLY small coordinating decor items (throw pillows, a small plant) that complement existing furniture.

CRITICAL: Do NOT use any furniture, decor, or accessories to cover or hide any visible damage, stains, cracks, or wear. All defects must remain fully visible.

Level the photo so vertical lines are truly vertical. Do NOT move camera horizontally.

Apply professional photo enhancement: correct exposure, fix white balance, reduce haze. Result must be photorealistic."""
    
    def _extract_image_from_response(self, response: dict) -> Optional[bytes]:
        """
        Extract image data from Gemini response.

        Args:
            response: API response dictionary

        Returns:
            Image bytes if found, None otherwise
        """
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                logger.warning("No candidates in response")
                # Check for prompt feedback (content policy block)
                prompt_feedback = response.get("promptFeedback", {})
                if prompt_feedback:
                    block_reason = prompt_feedback.get("blockReason", "unknown")
                    safety_ratings = prompt_feedback.get("safetyRatings", [])
                    logger.warning(f"Prompt blocked: {block_reason}")
                    logger.warning(f"Prompt safety ratings: {safety_ratings}")
                # Log full response structure for debugging
                logger.warning(f"Response keys: {list(response.keys())}")
                return None

            candidate = candidates[0]

            # Check finish reason with detailed logging
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "OTHER":
                # This is the mystery case - log everything we can
                logger.warning(f"finishReason=OTHER - Full candidate: {json.dumps(candidate, indent=2)[:1500]}")
            elif finish_reason and finish_reason not in ("STOP", "MAX_TOKENS"):
                logger.warning(f"Unexpected finish reason: {finish_reason}")

            # Check safety ratings
            safety_ratings = candidate.get("safetyRatings", [])
            blocked_categories = [
                r for r in safety_ratings
                if r.get("probability", "").upper() in ("HIGH", "MEDIUM")
            ]
            if blocked_categories:
                logger.warning(f"Safety concerns: {blocked_categories}")

            # Log all safety ratings when no image generated (for debugging)
            if finish_reason == "OTHER":
                logger.warning(f"All safety ratings: {safety_ratings}")

            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # Log parts structure when debugging
            if finish_reason == "OTHER":
                parts_summary = [
                    {k: v if k != "data" else f"<{len(v)} chars>" for k, v in (p.get("inlineData") or p).items()}
                    for p in parts
                ]
                logger.warning(f"Response parts structure: {parts_summary}")

            # Collect any text responses (might explain why no image)
            text_parts = []

            for part in parts:
                # Skip thought images
                if part.get("thought"):
                    continue

                # Capture text parts for debugging
                if "text" in part:
                    text_parts.append(part["text"])

                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data:
                    data = inline_data.get("data")
                    if data:
                        return base64.standard_b64decode(data)

            # Log text response if no image was found
            if text_parts:
                combined_text = " ".join(text_parts)[:500]  # Truncate for logging
                logger.warning(f"Model returned text instead of image: {combined_text}")
            else:
                logger.warning("No inline_data found in response parts (no text explanation)")

            return None

        except Exception as e:
            logger.error(f"Error extracting image from response: {e}")
            return None
    
    async def generate_text_to_image(self, prompt: str, aspect_ratio: str = "16:9") -> bytes:
        """
        Generate an image from text prompt only (no base image).

        Args:
            prompt: Text description of desired image
            aspect_ratio: Output aspect ratio (default "16:9")

        Returns:
            Generated image as bytes
        """
        request_body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio
                }
            }
        }
        
        url = f"{self.base_url}/models/{self.model}:generateContent"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=request_body
            )
            response.raise_for_status()
            
            result = response.json()
        
        image_data = self._extract_image_from_response(result)
        if not image_data:
            raise ValueError("No image generated from prompt")
        
        return image_data
