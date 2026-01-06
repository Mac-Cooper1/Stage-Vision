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

        # Detect style preference from original prompt (matches the 6 client-facing styles)
        style = "modern"
        for s in ["modern", "scandinavian", "coastal", "farmhouse", "midcentury", "mid-century", "architecture_digest", "architecture digest"]:
            if s in original_prompt.lower():
                style = s.replace(" ", "_").replace("-", "")  # Normalize to underscore format
                break

        if is_vacant:
            # Special handling for Architecture Digest style - COMPREHENSIVE with designer specs
            if style == "architecture_digest":
                furniture_by_room_ad = {
                    "bedroom": """Low platform bed with tall upholstered headboard (48-54" height) in oatmeal/cream Belgian linen, tight upholstery, no tufting, rounded top corners.
BEDDING: White/cream LINEN sheets slightly rumpled, cream duvet pulled back casually on one side, chunky knit throw in oatmeal draped at foot. 2-3 Euro shams + 2-3 accent pillows (cream/sage/taupe).
NIGHTSTANDS: Matching pair sculptural hourglass or drum shape in natural white oak 22-24" height.
LAMPS: Pair ceramic with sculptural organic base in warm cream/sage, natural linen drum shade.
RUG: Vintage Persian in faded earth tones extending 24-36" beyond bed (9x12 for queen). OR natural jute 8x10+.
CURTAINS: Flowing linen in warm white/cream, mounted high, puddling on floor.
ART: Large calming abstract above bed (40x50" to 60x40") in soft muted tones.
PLANT: Small plant on ONE nightstand (trailing pothos, succulent) OR nothing. Large tree only if room is very spacious.""",
                    "living room": """SOFA: Curved serpentine sofa in ivory/cream bouclé, low profile, rounded arms, short tapered oak legs. 84-96" length. Vladimir Kagan inspired. OR cognac leather if moodier.
COFFEE TABLE: Organic curved shape (kidney/cloud) in bleached white oak. Thick 2-3" top, rounded edges. OR round hammered brass with aged patina 36-40".
ACCENT CHAIRS: Pair of barrel swivel chairs in cream bouclé with brass base, angled 45° toward sofa. OR pair cognac leather lounge chairs with walnut frames.
RUG: Vintage Persian in FADED earth tones (muted rust, cream, sage) 9x12 or 10x14. OR chunky woven jute in natural honey 8x10+.
ART: One large abstract 48x60" minimum in earth tones. Thin natural oak float frame. Hung 6-8" above sofa.
ACCESSORIES: Stack 3-4 art/architecture books on coffee table, small sculptural ceramic beside books, chunky knit throw draped on sofa arm, 2-3 accent pillows (cream, sage, taupe), large woven seagrass basket on floor.
PLANT: Olive tree 6-7 ft in aged terracotta pot 18-24" diameter OR fiddle leaf 6-7 ft in woven basket. One corner only.
LIGHTING: Arc floor lamp with brass arm, linen shade, behind sofa.""",
                    "dining room": """TABLE: Solid white oak rectangular, Parsons-style legs, natural finish. 72-84" for 6 seats. OR walnut slab with live edge on blackened steel base.
CHAIRS: 6 Hans Wegner CH24 Wishbone chairs in natural ash/oak, paper cord seats. All matching.
PENDANT: Brass drum pendant 18-24" diameter, aged/patinated finish, centered 30-34" above table. OR large ceramic pendant in matte cream.
RUG: Natural jute in chunky weave, 9x12, extending 24-30" beyond chairs all sides.
CENTERPIECE: Table EMPTY (preferred) OR single sculptural cream ceramic vase (10-14" height) with 3-5 dried olive branches, slightly off-center.
ART: One large piece on focal wall. Abstract in earth tones 40x50" to 48x60".
PLANT (NO full tree): Tall floor vase (24-36") with dried branches/pampas in corner. Vase in cream, terracotta, or charcoal.""",
                    "office": """DESK: Natural wood desk with clean lines, warm oak or walnut finish.
CHAIR: Comfortable desk chair in cream/tan leather or natural linen.
BOOKSHELF: Styled with varied books (different heights, muted spine colors), sculptural ceramics, small plants, 1-2 framed art pieces. Leave some negative space.
RUG: Vintage Persian in faded earth tones OR natural jute.
LAMP: Brass desk lamp or sculptural ceramic table lamp.
PLANT: Fiddle leaf fig in corner OR small plant on desk (not olive tree).""",
                    "kitchen": """KEEP MINIMAL - 3-4 items maximum:
NEAR STOVE: Large olive wood cutting board (16x20"+) at casual angle with rustic sourdough loaf. Small ceramic pinch bowl with flaky salt.
ISLAND/COUNTER: Shallow wooden bowl (12-14" diameter) with 6-8 whole Meyer lemons. Position casually, not centered.
NEAR SINK: Small terracotta pot (4-6") with fresh rosemary or thyme.
SIGNATURE FLOWER: Single pink king protea stem in sculptural ceramic vase (round/bulbous, 8-10" height, matte charcoal or terracotta). ONE STEM ONLY.
BAR STOOLS (if island, 2-3): Woven saddle leather on light oak frame OR natural rattan with black metal legs.
DO NOT ADD: Books, large plants/trees, excessive accessories.""",
                    "bathroom": """SIGNATURE (essential): Sculptural ceramic vase in matte charcoal/black/terracotta, round/bulbous shape 8-12" height, with 1-2 pink king protea stems. Position prominently on vanity.
VANITY TRAY: Black slate or gray marble tray (8x12") containing: natural artisan bar soap (cream colored), small brass dish. Maximum 3 items.
TOWELS: Charcoal gray (preferred) OR cream. Plush, high-quality. Hung neatly on brass ring OR rolled in basket.
SMALL ACCENT (pick 1-2): Small maidenhair fern in ceramic pot, OR eucalyptus stems in glass vase, OR single pillar candle.
BASKET: Woven seagrass on floor with neatly rolled extra towels.""",
                    "hallway": """CONSOLE: Small console table in natural wood with clean lines.
MIRROR: Simple frame in natural oak or brass.
DECOR: Single sculptural ceramic object OR small plant in terracotta. Keep minimal.
RUG: Runner in natural fiber (jute/sisal) if long hallway.
NO large trees - keep hallway open and uncluttered.""",
                    "exterior": """SKY: Golden hour gradient - soft blue at top → warm golden/amber middle → soft peach/pink at horizon. Wispy clouds catching golden light.
WINDOWS: EVERY visible window MUST show warm amber interior glow (2700K look). Windows become beacons of warmth.
SIGNATURE: Mature olive tree (6-8 ft) in large aged terracotta pot (20-26") near front entry. ONE tree only.
PORCH: Teak or weathered wood furniture with gray/cream cushions. String lights (Edison bulbs) if appropriate.
LANDSCAPE: Trees catching golden side-light, lawn warmer golden-green tone, long shadows across lawn.""",
                    "room": """Designer furniture in natural materials:
SOFA/SEATING: Organic curved shapes in bouclé or linen, earth tones
TABLES: Natural wood with sculptural or organic shapes
RUG: Vintage Persian or natural jute
LIGHTING: Brass accents, linen shades
ACCESSORIES: Art books, sculptural ceramics, chunky throws
PLANT: One large tree (olive/fiddle leaf) OR small plants depending on room size"""
                }
                furniture = furniture_by_room_ad.get(room_type, furniture_by_room_ad["room"])

                # Special exterior prompt - LIGHTING ONLY, NOT STRUCTURAL
                if room_type == "exterior":
                    return f"""EDITORIAL EXTERIOR TRANSFORMATION (ARCHITECTURE DIGEST STYLE):

*** CRITICAL: LIGHTING TRANSFORMATION ONLY - DO NOT ALTER THE HOME'S STRUCTURE ***
- Do NOT move, add, remove, or resize ANY windows
- Do NOT alter the home's footprint, roofline, siding, or doors
- Do NOT change landscaping layout or add/remove trees
- Do NOT fill in any openings or alter architectural features
- Every window must remain in its EXACT original position
- The home must be immediately recognizable as the same property

LIGHTING TRANSFORMATION (atmosphere only):

LAYER 1 - GOLDEN HOUR SKY:
Transform sky to dramatic gradient: blue at top → warm gold in middle → pink/peach at horizon. Magic hour, 1 hour before sunset.

LAYER 2 - WINDOW GLOW:
EVERY EXISTING window MUST show warm amber interior glow - light visibly emanating from within. Apply to windows that ALREADY EXIST only.

LAYER 3 - GOLDEN LIGHT ON ARCHITECTURE:
EXISTING trees and landscaping catching golden side-light. Home's surfaces catching warm evening light.

ADDED DECOR (portable items only): Olive tree in terracotta pot near entry. Outdoor furniture on porch if present.

COLOR: Push entire image warm. NO cool/blue except upper sky. Whites become warm cream.

Result: 'Dwell magazine cover at sunset' through LIGHTING, not structural changes. The exact same home, just at magic hour."""

                return f"""EDITORIAL STAGING (ARCHITECTURE DIGEST STYLE): Stage this {room_type} for magazine-cover quality.

=============================================================================
⚠️ CRITICAL: STRUCTURAL PRESERVATION (HIGHEST PRIORITY) ⚠️
=============================================================================

NEVER ALTER, REMOVE, OR INVENT:
- Doorways and door openings (even if no door is visible)
- Archways and passages between rooms - if visible, they MUST remain visible
- Windows and window placements
- Walls and wall positions
- Room openings to adjacent spaces - if you can see a kitchen/hallway, that view MUST remain
- Built-in shelving, niches, or alcoves

SPECIFICALLY:
- Do NOT fill in doorways with walls
- Do NOT extend walls where there are openings
- Do NOT remove or alter any architectural pass-throughs
- Do NOT add walls or structural elements that don't exist

BEFORE GENERATING: Identify ALL openings to adjacent spaces. They MUST appear in output.

=============================================================================

=== THREE TRANSFORMATION LAYERS ===

LAYER 1 - DRAMATIC LIGHTING (photo enhancement, not structural):
- Golden hour quality - scene looks like 1 hour before sunset
- Visible warm light rays streaming through EXISTING windows
- Rich dimensional shadows in warm brown/amber (NOT flat/gray)
- Interior glow effect - space feels lit from within
- Color temp 2700K-3000K - NO cool/blue tones anywhere
- All whites become cream/ivory, all shadows become warm amber

LAYER 2 - DESIGNER STAGING (furniture only - NOT architectural changes):
{furniture}

LAYER 3 - COLOR GRADING:
Push entire image warm/golden. Whites = cream. Shadows = amber. Wood = honey/amber tones.

SIGNATURE ELEMENTS (1-2 per room, VARY across property):
- Pink protea in dark ceramic vase OR sculptural ceramics
- Olive tree ONLY in living/dining rooms if large - NOT in every room
- Small plants in kitchen/bathroom/bedroom instead of trees

⚠️ FINAL CHECK: Verify all doorways, openings, and passages to adjacent spaces are preserved EXACTLY as in the original.

CRITICAL: Walls, windows, doors, floor must be IDENTICAL to original. Do NOT cover damage/defects.

Result: Magazine-cover worthy through lighting + staging, not structural changes. Room must be recognizable as the same space."""

            # MODERN STYLE - COOL + CRISP + MINIMAL (opposite of warm AD)
            if style == "modern":
                furniture_by_room_modern = {
                    "living room": """SOFA: Low-profile sectional in COOL gray, white, or charcoal. Clean lines, chrome or hidden legs. NOT warm tones.
COFFEE TABLE: Glass with chrome base OR white marble with black steel frame. NOT warm walnut.
ACCENT CHAIRS: Pair in white, gray, or black leather. Chrome legs. Clean, architectural.
RUG: SOLID color - white, cream, or gray. Low pile, clean edges. NO patterns, NO warm tones. 8x10+.
ART: Large-scale minimalist piece in BLACK frame - geometric, B&W photography. Gallery-like.
ACCESSORIES: MINIMAL - one sculptural ceramic in white or black. 2-3 design books max. Chrome or black metal accents ONLY.
PLANT: Snake plant in WHITE or BLACK ceramic cylinder. OR skip entirely - modern is MINIMAL.
LIGHTING: Arc floor lamp in CHROME or BLACK (not brass).""",
                    "dining room": """TABLE: Glass with chrome base OR white lacquer. NOT warm walnut. 72-84" for 6.
CHAIRS: Molded chairs in WHITE, BLACK, or GRAY. Chrome legs. All matching.
PENDANT: Sculptural chrome or black pendant. NO brass.
RUG: Solid gray, white, or black. Low pile.
CENTERPIECE: Empty (preferred) OR single white/black sculptural object.""",
                    "bedroom": """BED: LOW platform in WHITE, light gray, or BLACK. Upholstered headboard in COOL gray or WHITE.
□ NO warm wood tones - use lacquer, white, or matte gray
□ Chrome or hidden legs

BEDDING: CRISP white, smooth and tailored (not rumpled). 1-2 accent pillows in charcoal or cool gray only.
NIGHTSTANDS: WHITE lacquer cube OR glass/chrome OR matte BLACK geometric. NOT warm walnut.
LAMPS: Ceramic cylinder in white/gray/black. Chrome accents.
RUG: Solid white, cream, or gray. NOT warm-toned.
ART: Single abstract in BLACK frame.""",
                    "kitchen": """MINIMAL - clear counters:
- One cutting board (BLACK or white)
- Single fruit in matte BLACK or white bowl
- Sculptural vase in matte white/black (empty or single stem)
BAR STOOLS: Black leather with CHROME or black metal frame. Clean lines.""",
                    "bathroom": """ULTRA MINIMAL:
- Soap dispenser in matte BLACK or white
- Small stone tray
- Towels in WHITE or gray only, neatly folded
- NO plants (keep it minimal)""",
                    "exterior": """LIGHTING: COOL, clean - blue hour or bright daylight.
- Interior glow through windows (WHITE/neutral, NOT warm amber)
- Architectural lighting emphasized
LANDSCAPING: Clean, geometric. Ornamental grasses, boxwood. Concrete or black metal planters.
FURNITURE: Modern outdoor - clean lines. Gray, black, or white cushions. NO warm woods.""",
                    "room": """MODERN = COOL + MINIMAL + GALLERY-LIKE:
SEATING: Low-profile in cool gray/white/charcoal, chrome legs
TABLES: Glass, white marble, or white lacquer - NOT warm walnut
RUG: Solid white/gray - NO patterns, NO warm tones
ACCESSORIES: Ultra minimal - one sculptural ceramic, chrome accents
PLANT: Skip in most rooms, or snake plant in white/black pot"""
                }
                furniture = furniture_by_room_modern.get(room_type, furniture_by_room_modern["room"])

                return f"""MODERN STAGING (COOL + CRISP + MINIMAL): Stage this {room_type} with gallery-like contemporary design.

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- NEVER shift camera angle - maintain EXACT perspective
- NOT every room needs a plant - Modern is MINIMAL

⚠️ STRUCTURAL PRESERVATION - NEVER alter walls, doorways, windows, or architectural features.

MODERN STYLE DNA (What makes Modern DIFFERENT):
- COOL, CRISP lighting (4000-5000K) - NO warm golden tones
- WHITE, cool gray, charcoal, BLACK palette
- WHITE OAK (bleached) or white lacquer - NO warm walnut (that's Mid-Century!)
- CHROME and matte black metals - NO brass (that's Mid-Century!)
- Gallery-like minimal aesthetic

FORBIDDEN: Warm amber, golden tones, earth tones, rattan, wicker, walnut, brass

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: COOL + MINIMAL + GALLERY-LIKE. Sophisticated, clean, NOT warm or cozy."""

            # SCANDINAVIAN STYLE - LIGHT BLONDE WOOD + SOFT PASTELS + HYGGE
            if style == "scandinavian":
                furniture_by_room_scandi = {
                    "living room": """SOFA: Clean-lined in light gray or SOFT BLUSH. Slim BLONDE wood legs (birch/ash). Bouclé or linen.
COFFEE TABLE: Round/oval in LIGHT OAK or BIRCH. NOT dark walnut!
ACCENT CHAIRS: Wishbone or shell chair in BLONDE wood. OR sheepskin-draped chair.
RUG: Cream wool with texture. SHEEPSKIN layered (SIGNATURE!). Soft, cozy.
ACCESSORIES (MUST INCLUDE 2-3):
□ SHEEPSKIN throw or accent (ESSENTIAL!)
□ CHUNKY KNIT throw in cream/gray
□ CANDLES in simple holders (HYGGE - ESSENTIAL!)
□ PASTEL accent - blush pillow or sage vase
PLANT: Trailing pothos in white ceramic OR eucalyptus in vase.
LIGHTING: Paper pendant, fabric shade lamp. CANDLES essential!
ART: Simple line drawings in LIGHT wood frames.""",
                    "dining room": """TABLE: LIGHT OAK or BIRCH, round/oval preferred. BLONDE wood - NOT walnut!
CHAIRS: Wishbone (CH24 style) in NATURAL/BLONDE. Paper cord seats.
PENDANT: PH5 style layered OR paper lantern (Noguchi). Soft diffused light.
RUG: Natural wool flatweave in cream. OR sheepskin.
CENTERPIECE: Single pillar CANDLE in wooden holder. CANDLES are essential for Scandinavian!""",
                    "bedroom": """BED: LIGHT BLONDE wood frame (birch, ash, or blonde oak) OR white/cream upholstered.
□ NO dark walnut - that's Mid-Century!
□ Light, airy presence

BEDDING: White linen, relaxed texture. CHUNKY KNIT throw (cream/gray). Mix linen/wool pillows.
NIGHTSTANDS: LIGHT BLONDE wood (SIGNATURE!). Simple, functional.
LAMPS: Ceramic in white/soft gray with linen shade.
RUG: SHEEPSKIN beside bed (SIGNATURE!) OR cream wool.
ACCESSORIES (MUST INCLUDE):
□ CANDLE on nightstand (HYGGE!)
□ Stack of books
□ Soft PASTEL accent if any (blush, sage)""",
                    "kitchen": """Functional hygge:
- LIGHT wood cutting boards (birch/blonde oak)
- Ceramic canisters in white or SOFT PASTELS
- Herbs in terracotta or white pots
- Linen tea towels
- CANDLE on counter (hygge!)
BAR STOOLS: LIGHT BLONDE wood. Simple Scandinavian design.""",
                    "bathroom": """Spa-like with hygge:
- Wooden tray with natural bar soap
- Eucalyptus stems in ceramic vase
- White linen towels, waffle weave
- CANDLE (ESSENTIAL for Scandinavian bathrooms!)
- Woven basket for storage""",
                    "exterior": """LIGHTING: Bright Nordic daylight OR soft overcast.
- WARM interior glow through windows (candles visible!)
LANDSCAPING: Natural, slightly wild. Simple wood/concrete planters.
FURNITURE: Light wood outdoor furniture. Cozy throws, CANDLES in lanterns.""",
                    "room": """SCANDINAVIAN = BLONDE WOOD + PASTELS + HYGGE:
SEATING: Light gray/blush, BLONDE wood legs
TABLES: LIGHT OAK/BIRCH - NOT dark walnut
RUG: Cream wool, SHEEPSKIN layered
ACCESSORIES: CHUNKY KNIT, CANDLES, PASTEL accents
PLANT: In woven basket or white ceramic"""
                }
                furniture = furniture_by_room_scandi.get(room_type, furniture_by_room_scandi["room"])

                return f"""SCANDINAVIAN STAGING (BLONDE WOOD + PASTELS + HYGGE): Stage this {room_type} with Nordic warmth.

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- NEVER shift camera angle - maintain EXACT perspective
- Use soft, organic plants - NOT architectural (that's Modern)

⚠️ STRUCTURAL PRESERVATION - NEVER alter walls, doorways, windows, or architectural features.

SCANDINAVIAN STYLE DNA (What makes Scandinavian DIFFERENT from Mid-Century):
- LIGHT BLONDE wood (birch, ash, light oak) - NOT DARK WALNUT (that's Mid-Century!)
- Soft PASTELS (blush pink, sage green) - NOT bold colors (that's Mid-Century!)
- COZY textures (sheepskin, chunky knit) - NOT sleek/sculptural
- Bright, airy Nordic light (3200-3500K)
- CANDLES are ESSENTIAL for hygge!

SIGNATURE ELEMENTS (must include 2-3):
□ SHEEPSKIN throw or rug (ESSENTIAL!)
□ CHUNKY KNIT throw blanket
□ CANDLES in simple holders (HYGGE!)
□ Soft PASTEL accent (blush, sage)

FORBIDDEN: Dark walnut, black iron, bold colors (mustard/orange), chrome

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: BLONDE WOOD + PASTELS + COZY HYGGE + CANDLES. Bright, airy, warm but light."""

            # COASTAL STYLE - BLUE + WOVEN TEXTURES + BEACH LIGHT
            if style == "coastal":
                furniture_by_room_coastal = {
                    "living room": """SOFA: Deep comfortable in white, cream, or SOFT BLUE linen. Slipcovered, relaxed fit.
COFFEE TABLE: Round WOVEN RATTAN with glass top (SIGNATURE!) OR whitewashed wood.
ACCENT CHAIRS: RATTAN/WICKER armchairs with white cushions (SIGNATURE!)
RUG: JUTE or SISAL in natural sandy tone (SIGNATURE!) OR BLUE and white stripe.
ACCESSORIES (MUST INCLUDE):
□ BLUE and white throw pillows (ESSENTIAL!)
□ WOVEN texture somewhere (basket, lamp)
□ Ocean art in WHITE frame
□ Light cotton/linen throw
PLANT: Palm in WOVEN SEAGRASS basket. Keep it airy - less is more for coastal.
LIGHTING: RATTAN or WOVEN pendant (SIGNATURE!). White ceramic lamps.""",
                    "dining room": """TABLE: Reclaimed wood OR white-washed trestle. Casual, beachy.
CHAIRS: WOVEN RATTAN/WICKER dining chairs (SIGNATURE!) OR slipcovered parsons in white.
PENDANT: Large WOVEN RATTAN/SEAGRASS pendant (SIGNATURE!)
RUG: Natural JUTE, large.
CENTERPIECE: White pitcher with eucalyptus OR hurricane lantern.""",
                    "bedroom": """BED: WHITE upholstered headboard OR RATTAN/CANE headboard (SIGNATURE!)
□ NO dark wood frames - light and airy
□ Fresh, breezy appearance

BEDDING: Crisp WHITE base. BLUE and white accent pillows (ESSENTIAL!). Light throw in blue or white.
NIGHTSTANDS: RATTAN or WICKER (SIGNATURE!) OR white/whitewashed wood.
LAMPS: White ceramic with WOVEN or linen shade.
RUG: Natural JUTE at bedside OR soft BLUE.
ACCESSORIES: SKIP plants in bedroom - keep it airy and breezy.""",
                    "kitchen": """Light, fresh, LESS IS MORE:
- Light wood cutting board
- White ceramic canisters
- Lemons in wooden bowl
- WOVEN basket with fruit
- NO large plants - keep counters clear
BAR STOOLS: WOVEN RATTAN/SEAGRASS counter stools (SIGNATURE!)""",
                    "bathroom": """Fresh and spa-like:
- White tray with natural soap
- Eucalyptus stems in glass vase
- White towels, SOFT BLUE accent towel
- WOVEN basket for storage
- SKIP plants - keep it clean and breezy""",
                    "exterior": """LIGHTING: BRIGHT beach daylight OR sunset with pink/orange sky.
- Warm interior glow through windows
LANDSCAPING: Coastal plants, ornamental grasses, palms. White/natural planters.
FURNITURE: Teak or white-washed wood. White and BLUE cushions (SIGNATURE!)""",
                    "room": """COASTAL = BLUE + WOVEN/RATTAN + BEACH BRIGHT:
SEATING: White/cream/BLUE linen, slipcovered
TABLES: RATTAN, whitewashed, or driftwood
RUG: JUTE/SISAL (essential!) or BLUE/white stripe
ACCESSORIES: BLUE + white required, WOVEN textures
PLANT: In woven basket - or SKIP for airy feel"""
                }
                furniture = furniture_by_room_coastal.get(room_type, furniture_by_room_coastal["room"])

                return f"""COASTAL STAGING (BLUE + WOVEN TEXTURES + BEACH LIGHT): Stage this {room_type} with beach house elegance.

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- NEVER shift camera angle - maintain EXACT perspective
- SKIP plants in many rooms - Coastal is AIRY and breezy

⚠️ STRUCTURAL PRESERVATION - NEVER alter walls, doorways, windows, or architectural features.

COASTAL STYLE DNA (What makes Coastal DIFFERENT):
- BLUE is REQUIRED - at least one blue element per room (pillow, throw, rug)
- RATTAN/WICKER textures are ESSENTIAL (furniture, pendant, basket)
- JUTE/SISAL rugs in natural sandy tones
- Bright, sun-drenched beach light (3500-4000K)
- Whitewashed wood, driftwood - NOT warm walnut

SIGNATURE ELEMENTS (must include 2-3):
□ BLUE and white pillows (ESSENTIAL!)
□ RATTAN or WICKER furniture piece
□ JUTE or SISAL rug
□ WOVEN pendant or lamp

FORBIDDEN: Dark walnut, warm amber, chrome, black iron, cozy knits

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: BLUE + WOVEN/RATTAN + BEACH BRIGHT. Relaxed, airy, refined coastal - not kitschy."""

            # FARMHOUSE STYLE - RECLAIMED WOOD + BLACK IRON + VINTAGE
            if style == "farmhouse":
                furniture_by_room_farmhouse = {
                    "living room": """SOFA: Deep comfortable in cream/warm gray. Linen/cotton slipcover. Substantial comfort.
COFFEE TABLE: RECLAIMED WOOD with visible character (SIGNATURE!). Substantial, rustic.
ACCENT CHAIRS: Leather club chairs in cognac. OR wingback in linen.
RUG: VINTAGE-style in FADED muted tones (faded reds, blues, creams). Shows character.
ACCESSORIES (MUST INCLUDE 2-3):
□ BLACK IRON element (lamp, hardware) - ESSENTIAL!
□ VINTAGE/antique piece
□ Grain sack or ticking stripe pattern
□ Ceramic pitcher or crock
PLANT: Eucalyptus stems in ceramic PITCHER (SIGNATURE!) OR cotton stems in vintage vessel.
LIGHTING: Industrial floor lamp in BLACK METAL (SIGNATURE!). Lanterns.
ART: Vintage botanical prints in simple frames.""",
                    "dining room": """TABLE: Large RECLAIMED WOOD farmhouse table (SIGNATURE!). Substantial trestle base.
CHAIRS: Cross-back (X-back) in BLACK (SIGNATURE!) OR Windsor in black.
PENDANT: Large BLACK METAL chandelier (SIGNATURE!) - linear or round.
RUG: Vintage-style OR natural jute.
CENTERPIECE: Ceramic pitcher with dried flowers OR wooden dough bowl.""",
                    "bedroom": """BED: BLACK IRON or metal bed frame (SIGNATURE!) OR reclaimed wood headboard with character.
□ Can show wear/age in FURNITURE (not walls!)
□ Substantial, sturdy presence

BEDDING: White base with texture. VINTAGE QUILT at foot (SIGNATURE!). Linen/cotton in cream.
NIGHTSTANDS: MISMATCHED VINTAGE pieces (charming!) OR distressed wood.
LAMPS: Ceramic in white/cream with linen shade.
RUG: VINTAGE-style with FADED pattern.
ACCESSORIES: Candlestick, flowers in small pitcher, vintage books.""",
                    "kitchen": """Rustic charm:
- Butcher block cutting board
- Ceramic crock with wooden utensils
- VINTAGE canisters or glass jars
- Fresh produce in basket
- BLACK IRON pot rack or hardware visible
BAR STOOLS: Industrial BLACK METAL (SIGNATURE!) OR cross-back in black/natural.""",
                    "bathroom": """Vintage charm:
- Wooden tray with natural bar soap
- Glass jars or vintage containers
- WHITE waffle weave towels on vintage hooks/ladder
- GALVANIZED metal or wire basket
- Small candle in vintage holder""",
                    "exterior": """LIGHTING: WARM golden hour, barn-like glow.
- Interior windows showing warm light
LANDSCAPING: Cottage garden - lavender, hydrangeas. Galvanized or terracotta planters.
FURNITURE: Rocking chairs, Adirondack, vintage metal. Lanterns, string lights.""",
                    "room": """FARMHOUSE = RECLAIMED WOOD + BLACK IRON + VINTAGE:
SEATING: Cream/gray slipcovered, substantial
TABLES: RECLAIMED wood with character
RUG: VINTAGE-style or jute
ACCESSORIES: BLACK IRON, ceramic pitchers, galvanized
PLANT: Eucalyptus in pitcher or cotton stems"""
                }
                furniture = furniture_by_room_farmhouse.get(room_type, furniture_by_room_farmhouse["room"])

                return f"""FARMHOUSE STAGING (RECLAIMED WOOD + BLACK IRON + VINTAGE): Stage this {room_type} with rustic charm.

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- Distressed FURNITURE is style. Damaged WALLS is fraud.
- NEVER shift camera angle - maintain EXACT perspective

⚠️ STRUCTURAL PRESERVATION - NEVER alter walls, doorways, windows, or architectural features.

FARMHOUSE STYLE DNA (What makes Farmhouse DIFFERENT):
- BLACK IRON is REQUIRED (bed frame, lamp, chandelier, hardware)
- RECLAIMED WOOD with visible character
- VINTAGE elements (quilt, mismatched nightstands)
- Warm barn/candlelit light (2700-3000K)
- White, cream + BLACK IRON accents

SIGNATURE ELEMENTS (must include 2-3):
□ BLACK IRON element (ESSENTIAL!)
□ RECLAIMED WOOD furniture
□ VINTAGE/antique piece
□ Eucalyptus in ceramic PITCHER

FORBIDDEN: Chrome, high-gloss, sleek modern, blonde wood, brass

⚠️ DAMAGE PREVENTION:
- Show CHARACTER in FURNITURE (distressed, worn) = STYLE
- Do NOT invent damage on WALLS (cracks, holes) = FRAUD
- Preserve wall condition EXACTLY

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: RECLAIMED WOOD + BLACK IRON + VINTAGE CHARACTER. Rustic warmth, not country kitsch."""

            # MID-CENTURY MODERN STYLE - DARK WALNUT + BOLD COLORS + TAPERED LEGS
            if style == "midcentury":
                furniture_by_room_mcm = {
                    "living room": """SOFA: Low-profile in BOLD COLOR - MUSTARD, OLIVE, or BURNT ORANGE (SIGNATURE!). TAPERED WALNUT legs.
COFFEE TABLE: Surfboard shape in DARK WALNUT (SIGNATURE!) OR Noguchi-inspired. TAPERED LEGS essential.
ACCENT CHAIRS: Eames Lounge Chair and Ottoman in leather (SIGNATURE!). OR shell chairs. OR womb chair in bold color.
RUG: SHAG rug in cream, GOLD, or OLIVE (SIGNATURE!) OR bold geometric in warm tones.
ACCESSORIES (MUST INCLUDE 2-3):
□ STARBURST element (clock, mirror, wall art) - SIGNATURE!
□ BOLD COLOR accent - mustard, olive, or orange - ESSENTIAL!
□ BRASS accent (lamp, candleholder)
□ Sculptural ceramic in period color
PLANT: Snake plant in BULLET PLANTER (period ceramic!) in white, orange, or olive.
LIGHTING: Arc floor lamp in BRASS (Arco style). SPUTNIK chandelier (SIGNATURE!)
ART: Large abstract expressionist or bold graphic print.""",
                    "dining room": """TABLE: Oval DARK WALNUT with TAPERED LEGS (SIGNATURE!). OR Saarinen tulip.
CHAIRS: Eames molded plastic in colors OR Wegner Wishbone in WALNUT. All matching.
PENDANT: SPUTNIK chandelier in BRASS (SIGNATURE!) OR PH Artichoke. BRASS essential.
RUG: Bold geometric in period colors OR SHAG in gold/olive.
CENTERPIECE: Sculptural ceramic bowl in period color.""",
                    "bedroom": """BED: DARK WALNUT platform with TAPERED LEGS (SIGNATURE!)
□ Low profile, panel/slat headboard
□ NO light wood - that's Scandinavian!
□ Substantial, iconic presence

BEDDING: White/cream base. BOLD accent throw - MUSTARD, OLIVE, or BURNT ORANGE (ESSENTIAL!)
NIGHTSTANDS: DARK WALNUT with TAPERED LEGS (SIGNATURE!). BRASS hardware.
LAMPS: Ceramic in period color (mustard, olive, cream). BRASS accents.
RUG: SHAG in cream, gold, or olive.
ACCESSORIES: STARBURST clock or mirror (SIGNATURE!), BRASS candleholder.""",
                    "kitchen": """Period-appropriate:
- Teak cutting board
- Ceramic canisters in period colors (white, ORANGE, OLIVE)
- Fruit in sculptural bowl
- Dansk or Scandinavian ceramics
BAR STOOLS: DARK WALNUT with TAPERED LEGS. OR molded seats in period colors.""",
                    "bathroom": """Bold, period-appropriate:
- Simple tray with soap
- Ceramic vessel in BOLD period color (MUSTARD, OLIVE, orange)
- Snake plant in bullet planter
- Towels in bold solid color
- BRASS accents (essential!)""",
                    "exterior": """LIGHTING: WARM golden hour, rich and saturated.
- Dramatic sky
- Interior glow showing warm amber tones
LANDSCAPING: Desert modern (agave, succulents) OR structured. Gravel, concrete.
FURNITURE: Emphasize period features. DARK WALNUT or teak.""",
                    "room": """MID-CENTURY = DARK WALNUT + BOLD COLORS + TAPERED LEGS:
SEATING: BOLD color (mustard/olive/orange), TAPERED walnut legs
TABLES: DARK WALNUT with TAPERED LEGS
RUG: SHAG or bold geometric
ACCESSORIES: STARBURST, BRASS, period ceramics
PLANT: In ceramic BULLET PLANTER"""
                }
                furniture = furniture_by_room_mcm.get(room_type, furniture_by_room_mcm["room"])

                return f"""MID-CENTURY MODERN STAGING (DARK WALNUT + BOLD COLORS + TAPERED LEGS): Stage this {room_type} with iconic 1950s-60s design.

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- NEVER shift camera angle - maintain EXACT perspective
- Use DARK WALNUT - NOT light blonde wood (that's Scandinavian!)

⚠️ STRUCTURAL PRESERVATION - NEVER alter walls, doorways, windows, or architectural features.

MID-CENTURY STYLE DNA (What makes Mid-Century DIFFERENT from Scandinavian):
- DARK WALNUT with TAPERED LEGS (NOT light blonde wood!)
- BOLD COLORS: mustard yellow, olive green, burnt orange (NOT soft pastels!)
- SLEEK, sculptural (NOT cozy chunky knit)
- Rich, saturated amber lighting (2700-3000K)
- BRASS accents essential (NOT black iron!)

SIGNATURE ELEMENTS (must include 2-3):
□ DARK WALNUT furniture with TAPERED LEGS (ESSENTIAL!)
□ BOLD COLOR accent - mustard, olive, or orange (ESSENTIAL!)
□ STARBURST element (clock, mirror, art)
□ BRASS accent
□ SHAG texture (rug or pillow)
□ Ceramic BULLET PLANTER

FORBIDDEN: Light blonde wood (Scandinavian!), soft pastels, black iron, chunky knits, chrome

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: DARK WALNUT + BOLD COLORS + TAPERED LEGS + BRASS + STARBURST. Iconic, warm, statement-making."""

            # Standard staging fallback for vacant rooms (default/unknown style)
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
            # Special handling for Architecture Digest style (occupied rooms) - ENHANCED
            if style == "architecture_digest":
                # Special exterior prompt for occupied/existing exteriors - LIGHTING ONLY
                if room_type == "exterior":
                    return f"""EDITORIAL EXTERIOR TRANSFORMATION (ARCHITECTURE DIGEST STYLE):

*** CRITICAL: LIGHTING TRANSFORMATION ONLY - DO NOT ALTER THE HOME'S STRUCTURE ***
- Do NOT move, add, remove, or resize ANY windows
- Do NOT alter the home's footprint, roofline, siding, or doors
- Do NOT change landscaping layout or add/remove trees
- Do NOT fill in any openings or alter architectural features
- Every window must remain in its EXACT original position
- The home must be immediately recognizable as the same property

LIGHTING TRANSFORMATION (atmosphere only):

LAYER 1 - GOLDEN HOUR SKY:
Transform sky to dramatic gradient: blue at top → warm gold in middle → pink/peach at horizon. Magic hour.

LAYER 2 - WINDOW GLOW:
EVERY EXISTING window shows warm amber interior glow. Apply to windows that ALREADY EXIST only.

LAYER 3 - GOLDEN LIGHT ON ARCHITECTURE:
EXISTING trees/landscaping catching golden side-light. Home's surfaces catching warm evening light.

ADDED DECOR (portable only): Olive tree in terracotta near entry if space allows.

COLOR: Push entire image warm. NO cool/blue except upper sky. Clean up clutter.

Result: 'Dwell magazine cover at sunset' through LIGHTING, not structural changes. Same property, magic hour."""

                return f"""EDITORIAL ENHANCEMENT (ARCHITECTURE DIGEST STYLE): Transform this {room_type} to magazine-cover quality.

=============================================================================
⚠️ CRITICAL: STRUCTURAL PRESERVATION (HIGHEST PRIORITY) ⚠️
=============================================================================

NEVER ALTER, REMOVE, OR INVENT:
- Doorways and door openings (even if no door is visible)
- Archways and passages between rooms - if visible, they MUST remain visible
- Windows and window placements
- Walls and wall positions
- Room openings to adjacent spaces - if you can see a kitchen/hallway, that view MUST remain
- Built-in shelving, niches, or alcoves

SPECIFICALLY:
- Do NOT fill in doorways with walls
- Do NOT extend walls where there are openings
- Do NOT remove or alter any architectural pass-throughs
- Do NOT add walls or structural elements that don't exist

BEFORE GENERATING: Identify ALL openings to adjacent spaces. They MUST appear in output.

=============================================================================

=== THREE TRANSFORMATION LAYERS ===

LAYER 1 - DRAMATIC LIGHTING (photo enhancement, not structural):
- Golden hour quality - scene looks like 1 hour before sunset
- Visible warm light rays streaming through EXISTING windows
- Rich dimensional shadows in warm brown/amber (NOT flat/gray)
- Interior glow effect
- Color temp 2700K-3000K - NO cool/blue tones
- All whites become cream/ivory

LAYER 2 - STYLING (keep existing furniture, add accessories only):
Remove clutter/personal items. Add complementary warm accessories:
- Pink protea in dark sculptural ceramic vase OR small plant in terracotta (not both)
- Artisanal ceramics where appropriate
Use ONLY warm materials - NO cool blues, chrome, or stark whites.
Do NOT add large olive trees to every room.

LAYER 3 - COLOR GRADING:
Push entire image warm/golden. Shadows = amber. Wood = honey tones.

KEEP EXISTING FURNITURE AND ARCHITECTURE: Same layout, same walls, same windows. Only remove clutter and enhance lighting.

⚠️ FINAL CHECK: Verify all doorways, openings, and passages to adjacent spaces are preserved EXACTLY as in the original.

CRITICAL: Do NOT cover any damage/defects. Do NOT alter any architectural features.

Result: Magazine-cover worthy through lighting + styling, not structural changes. Room must be recognizable as the same space."""

            # Standard declutter fallback for occupied rooms
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
