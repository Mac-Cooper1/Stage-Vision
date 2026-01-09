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


# =============================================================================
# STRUCTURAL PRESERVATION RULES FOR IMAGE GENERATION
# Shorter version for nano_client fallback prompts
# =============================================================================

NANO_STRUCTURAL_RULES = """
🔒 ABSOLUTE STRUCTURAL LOCK - VIOLATION = UNUSABLE OUTPUT:

CAMERA: Do NOT rotate view left/right. Same walls visible as original.
FLOORING: Carpet stays carpet. Hardwood stays hardwood. Tile stays tile. NO CHANGES.
WINDOWS: Same size, same position. Do NOT make small windows floor-to-ceiling.
WALLS: Same positions. Do NOT add/remove walls or openings.
CEILING: No track lighting or fixtures added. Crown molding preserved.
PROPORTIONS: Room same size - no wide-angle distortion.

⚠️ FLOORING IS CRITICAL: If you see CARPET in the original, output MUST have carpet.
Changing flooring material is FRAUD - the flooring is part of the actual property.
"""


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

{NANO_STRUCTURAL_RULES}

=============================================================================
⚠️⚠️⚠️ CAMERA AND STRUCTURE LOCK ⚠️⚠️⚠️
=============================================================================

🚫 CAMERA - ABSOLUTE LOCK:
- Maintain EXACT same camera position, angle, and field of view as original
- Do NOT rotate view left or right - same walls must be visible
- Before/after must align pixel-perfectly on architectural features

🏠 ARCHITECTURE - ZERO CHANGES:
- ALL walls in EXACTLY the same positions
- ALL windows in EXACTLY the same positions, same size, same style
- ALL doors in EXACTLY the same positions
- Ceiling features UNCHANGED - NO track lighting added

🛋️ FURNITURE - SAME WALLS:
- If bed is on LEFT wall, staged bed goes on LEFT wall
- Do NOT move major furniture to different walls

=============================================================================
⚠️ STRUCTURAL PRESERVATION (HIGHEST PRIORITY) ⚠️
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
- Do NOT add track lighting, recessed lights, or skylights

⚠️ DAMAGE INVENTION PREVENTION (CRITICAL):
When removing items like TVs, wall art, or furniture:
- The wall/surface behind MUST appear CLEAN and UNDAMAGED
- Do NOT add mounting holes, screw marks, or discoloration where items were
- Do NOT invent paint chips, cracks, or marks where items were removed
- If removing a TV from a wall, that wall section becomes a CLEAN, NORMAL wall
- The ONLY damage allowed is damage CLEARLY VISIBLE in the original photo
- Creating fake damage is FRAUD and violates MLS compliance

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

⚠️ FINAL CHECK:
1. Verify all doorways, openings, and passages are preserved EXACTLY
2. Verify NO damage was invented where items were removed (TVs, art, etc.)
3. Any removed items leave CLEAN walls behind - no holes, marks, or discoloration

CRITICAL: Walls, windows, doors, floor must be IDENTICAL to original. Do NOT cover OR invent damage/defects.

Result: Magazine-cover worthy through lighting + staging, not structural changes. Room must be recognizable as the same space."""

            # MODERN 2026 STYLE - "Ultra-Simple Holographic Minimalism"
            # ETHEREAL + SCULPTURAL + COOL + VOID
            if style == "modern":
                furniture_by_room_modern = {
                    "living room": """MODERN 2026 - "The Triangular Void":
SOFA: Low sculptural form in PURE WHITE, concrete gray, or deep charcoal. POST-MATERIAL appearance - resin, molded, architectural. NO warm tones.
COFFEE TABLE: Resin/acrylic with holographic shimmer OR concrete sculptural form OR black glass void. NOT wood of any kind.
ACCENT CHAIRS: Sculptural forms - think Zaha Hadid. Chrome, polished nickel, or matte black. ONE piece may have iridescent/holographic element.
RUG: NONE preferred (negative space). If needed: solid white, concrete gray. SHARP GEOMETRIC edges.
ART: Large-scale "post-digital" piece - generative art, holographic print, or stark B&W. Float-mounted.
ACCESSORIES: ALMOST NONE - negative space IS the design. Maximum 1 sculptural ceramic in white or black.
PLANT: Single architectural specimen (snake plant, bird of paradise) in BLACK or WHITE cylinder. Or NONE.
LIGHTING: Sculptural LED element. Light as architecture - visible rays creating geometric patterns.""",
                    "dining room": """TABLE: Resin/acrylic (translucent) OR concrete slab OR black glass. NO wood. Sharp geometric form.
CHAIRS: Sculptural molded forms in white, black, or clear. Chrome or hidden legs. All matching.
PENDANT: Linear LED sculpture OR geometric chrome. Light as architectural element.
RUG: NONE (preferred) OR solid concrete gray.
CENTERPIECE: EMPTY (the void). One sculptural object maximum.""",
                    "bedroom": """BED: LOW platform - WHITE lacquer, concrete effect, or matte charcoal. Post-material appearance.
□ NO wood tones of any kind
□ Architectural, sculptural presence
□ Chrome or hidden legs

BEDDING: PURE WHITE, smooth and architectural (not soft/rumpled). ONE accent in charcoal or iridescent silver.
NIGHTSTANDS: Resin cube (translucent), white lacquer void, or floating shelf. NOT any wood.
LAMPS: Sculptural LED, chrome, or glass. Geometric forms.
RUG: NONE or minimal white/gray. Sharp edges.
ART: Single post-digital piece in minimal frame.""",
                    "kitchen": """VOID AESTHETIC - counters nearly empty:
- NOTHING or one sculptural object in white/black
- Clear space emphasized
BAR STOOLS: Sculptural chrome or matte black. Architectural forms.""",
                    "bathroom": """ARCHITECTURAL VOID:
- Minimal stone tray with single object
- Towels in WHITE only, architectural fold
- NO plants - the void is the point""",
                    "exterior": """LIGHTING: COOL blue hour OR crisp bright daylight.
- Interior windows showing WHITE/neutral glow (NOT warm amber)
- Architectural lighting emphasized
LANDSCAPING: Geometric, minimal. Ornamental grasses. Concrete or black metal planters.
FURNITURE: Sculptural outdoor pieces. White, gray, black. NO warm materials.""",
                    "room": """MODERN 2026 = ETHEREAL + SCULPTURAL + COOL + VOID:
SEATING: Post-material sculptural forms in white/gray/charcoal
TABLES: Resin, concrete, black glass - NO wood
RUG: NONE or minimal geometric
ACCESSORIES: Almost none - negative space IS the design
PLANT: One architectural plant or NONE"""
                }
                furniture = furniture_by_room_modern.get(room_type, furniture_by_room_modern["room"])

                return f"""MODERN 2026 STAGING - "Ultra-Simple Holographic Minimalism": Stage this {room_type} with post-material ethereal design.

{NANO_STRUCTURAL_RULES}

=============================================================================
⚠️⚠️⚠️ CAMERA AND STRUCTURE LOCK ⚠️⚠️⚠️
=============================================================================

🚫 CAMERA - ABSOLUTE LOCK:
- Maintain EXACT same camera position, angle, and field of view as original
- Do NOT rotate view left or right - same walls must be visible

🏠 ARCHITECTURE - ZERO CHANGES:
- ALL walls, windows, doors in EXACTLY the same positions
- Ceiling features UNCHANGED - NO track lighting added

🛋️ FURNITURE - SAME WALLS:
- If bed is on LEFT wall, staged bed goes on LEFT wall

=============================================================================

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- NEVER shift camera angle - maintain EXACT perspective
- MINIMAL to NO plants - Modern 2026 embraces the VOID

⚠️ STRUCTURAL PRESERVATION - NEVER alter walls, doorways, windows, or architectural features. NO added track lighting.

MODERN 2026 STYLE DNA - "The Architecture of Silence":
- COOL, ethereal light quality (4500-5500K) - crisp white with subtle holographic shimmer
- Light as SCULPTURAL element - visible rays creating geometric patterns
- Pure white, concrete gray, deep charcoal, BLACK palette
- ONE iridescent/holographic element allowed
- POST-MATERIAL furniture - resin, acrylic, concrete, chrome, glass
- "Negative space as design" - empty space is as important as furniture
- "Geospatial Extremes" - exaggerated sharp angles and diagonals
- Ethereal, almost "digital" quality

SIGNATURE ELEMENTS:
□ Holographic or iridescent accent (ONE per room)
□ Sculptural furniture with architectural presence
□ Visible light rays/geometric patterns
□ Dramatic interplay of shadow and brilliant illumination

FORBIDDEN: ANY warm tones, ANY wood (walnut, oak, teak), brass, earth tones, rattan, wicker, soft textures, cozy elements

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: ETHEREAL + SCULPTURAL + COOL + VOID. Post-material digital perfection meets organic unpredictability."""

            # SCANDINAVIAN 2026 STYLE - "Nordic Ethereal - Spiritual Hygge"
            # BLONDE WOOD + EARTH-SHADOWS + SHEEPSKIN + CANDLES + HYGGE
            if style == "scandinavian":
                furniture_by_room_scandi = {
                    "living room": """SCANDINAVIAN 2026 - "Spiritual Hygge":
SOFA: Soft curved form in warm cream, SOFT TERRACOTTA, or muted sage. Bouclé or heavyweight linen. BLONDE wood legs (birch/ash).
COFFEE TABLE: Organic curved shape in LIGHT BLONDE OAK or BIRCH. Soft rounded edges. NOT walnut!
ACCENT CHAIRS: Wishbone or shell chair in BLONDE wood. SHEEPSKIN draped over chair (ESSENTIAL!).
RUG: Natural wool in cream/oatmeal. SHEEPSKIN layered (SIGNATURE!). Soft, enveloping texture.
ACCESSORIES (MUST INCLUDE 3-4):
□ SHEEPSKIN throw or accent (ESSENTIAL - spiritual warmth!)
□ CHUNKY KNIT throw in cream/oatmeal
□ CANDLES - multiple, varying heights (HYGGE ESSENTIAL!)
□ SOFT TERRACOTTA or MUTED SAGE accent
□ Dried botanicals in ceramic vase
PLANT: Trailing pothos in handmade ceramic OR dried pampas/botanicals. Organic, imperfect.
LIGHTING: Paper pendant (Noguchi-inspired), fabric shade lamps. CANDLELIGHT is ESSENTIAL!
ART: Soft abstract in EARTH-SHADOW tones. Light wood frame.""",
                    "dining room": """TABLE: LIGHT BLONDE OAK or BIRCH, round/oval organic shape. NOT walnut!
CHAIRS: Wishbone (CH24 style) in NATURAL BLONDE. Paper cord seats.
PENDANT: Paper lantern (Noguchi), PH5 layered. Soft, diffused, spiritual light.
RUG: Natural wool flatweave. SHEEPSKIN on chairs.
CENTERPIECE: Multiple CANDLES of varying heights (ESSENTIAL!) OR single sculptural ceramic with dried botanicals.""",
                    "bedroom": """BED: LIGHT BLONDE wood frame (birch, ash) OR soft linen upholstered in warm cream.
□ NO dark walnut - that's Mid-Century!
□ Soft, enveloping, spiritual presence

BEDDING: White/cream heavyweight linen, LIVED-IN texture. CHUNKY KNIT throw (oatmeal). Mix of soft pillows in cream/sage/terracotta.
NIGHTSTANDS: LIGHT BLONDE wood with organic curves (SIGNATURE!). Simple, handmade feel.
LAMPS: Handmade ceramic in matte cream with linen shade.
RUG: SHEEPSKIN beside bed (ESSENTIAL!) layered on natural wool.
ACCESSORIES (MUST INCLUDE):
□ CANDLES on nightstand (HYGGE - spiritual warmth!)
□ Stack of books with soft covers
□ SOFT TERRACOTTA or SAGE ceramic
□ Dried botanical arrangement""",
                    "kitchen": """Spiritual hygge functionality:
- LIGHT BLONDE wood cutting boards (birch)
- Handmade ceramic vessels in soft neutrals or TERRACOTTA
- Fresh herbs in terracotta pots
- Linen tea towels in oatmeal
- CANDLE in simple holder (hygge!)
BAR STOOLS: LIGHT BLONDE wood with woven paper cord seats.""",
                    "bathroom": """Spa sanctuary with spiritual hygge:
- Natural wood tray with artisan bar soap
- Dried eucalyptus or botanicals in ceramic vase
- White/cream linen towels, waffle weave
- Multiple CANDLES (ESSENTIAL!)
- Woven basket for storage""",
                    "exterior": """LIGHTING: Soft Nordic daylight, diffused and gentle. OR warm golden hour.
- Interior windows showing warm candlelit glow
LANDSCAPING: Natural, slightly wild. Native plants. Terracotta planters.
FURNITURE: Light wood outdoor. SHEEPSKIN throws, CANDLES in lanterns.""",
                    "room": """SCANDINAVIAN 2026 = BLONDE WOOD + EARTH-SHADOWS + SHEEPSKIN + CANDLES + HYGGE:
SEATING: Soft curves in cream/terracotta/sage, BLONDE wood
TABLES: LIGHT OAK/BIRCH organic shapes - NOT walnut
RUG: Natural wool, SHEEPSKIN layered
ACCESSORIES: CHUNKY KNIT, multiple CANDLES, dried botanicals
PLANT: Organic trailing plants or dried botanicals"""
                }
                furniture = furniture_by_room_scandi.get(room_type, furniture_by_room_scandi["room"])

                return f"""SCANDINAVIAN 2026 STAGING - "Nordic Ethereal - Spiritual Hygge": Stage this {room_type} with soul-nourishing Nordic warmth.

{NANO_STRUCTURAL_RULES}

=============================================================================
⚠️⚠️⚠️ CAMERA AND STRUCTURE LOCK ⚠️⚠️⚠️
=============================================================================

🚫 CAMERA - ABSOLUTE LOCK:
- Maintain EXACT same camera position, angle, and field of view as original
- Do NOT rotate view left or right - same walls must be visible

🏠 ARCHITECTURE - ZERO CHANGES:
- ALL walls, windows, doors in EXACTLY the same positions
- Ceiling features UNCHANGED - NO track lighting added

🛋️ FURNITURE - SAME WALLS:
- If bed is on LEFT wall, staged bed goes on LEFT wall

=============================================================================

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- NEVER shift camera angle - maintain EXACT perspective
- Use soft, organic, IMPERFECT elements - handmade aesthetic

⚠️ STRUCTURAL PRESERVATION - NEVER alter walls, doorways, windows, or architectural features. NO added track lighting.

SCANDINAVIAN 2026 STYLE DNA - "Spiritual Hygge":
- LIGHT BLONDE wood (birch, ash, light oak) - NOT DARK WALNUT!
- EARTH-SHADOW palette: warm cream, SOFT TERRACOTTA, MUTED SAGE, oatmeal
- Deep shadows as design element - "earth-shadows" creating depth
- SHEEPSKIN textures are ESSENTIAL (spiritual warmth)
- CANDLES EVERYWHERE - multiple heights, spiritual light
- CHUNKY KNIT throws - enveloping comfort
- Handmade, organic, imperfect ceramics
- Dried botanicals alongside fresh

SIGNATURE ELEMENTS (MUST include 3-4):
□ SHEEPSKIN throw or rug (ESSENTIAL - spiritual warmth!)
□ CHUNKY KNIT throw blanket in oatmeal
□ Multiple CANDLES - varying heights (HYGGE ESSENTIAL!)
□ SOFT TERRACOTTA or MUTED SAGE accent
□ Dried botanicals/pampas

FORBIDDEN: Dark walnut, black iron, bold colors (mustard/orange), chrome, sleek modern materials

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: BLONDE WOOD + EARTH-SHADOWS + SHEEPSKIN + CANDLES. Soul-nourishing spiritual hygge with Nordic serenity."""

            # COASTAL 2026 STYLE - "Hyper-Breezy Sensory Obsession"
            # DOPAMINE BRIGHTS + ROPE/WOVEN + HYPER-BREEZY + HERITAGE
            if style == "coastal":
                furniture_by_room_coastal = {
                    "living room": """COASTAL 2026 - "Hyper-Breezy Sensory":
SOFA: Deep comfortable in crisp WHITE or natural linen. Slipcovered, RELAXED BREEZY fit. Sinks-into comfort.
COFFEE TABLE: ROPE-wrapped base with weathered wood top (SIGNATURE!) OR driftwood sculptural.
ACCENT CHAIRS: WOVEN ROPE or RATTAN armchairs (SIGNATURE!) with white/cream cushions.
RUG: Natural JUTE or SISAL - sandy, textured (ESSENTIAL!). Or BLUE/white stripe.
ACCESSORIES (MUST INCLUDE 3-4):
□ DOPAMINE BRIGHT accent - CORAL, TURQUOISE, or SUNNY YELLOW (ESSENTIAL!)
□ ROPE element - lamp base, basket, or decor
□ NAUTICAL HERITAGE piece - vintage buoy, ship detail, lighthouse art
□ WOVEN texture - seagrass basket, rattan tray
□ Ocean/coastal art in weathered wood frame
PLANT: Palm or bird of paradise in WOVEN SEAGRASS basket. Tropical, breezy feel.
LIGHTING: ROPE-wrapped lamp base OR WOVEN pendant (SIGNATURE!). Natural materials.""",
                    "dining room": """TABLE: Weathered reclaimed wood OR whitewashed trestle. HERITAGE feel.
CHAIRS: WOVEN ROPE or RATTAN dining chairs (SIGNATURE!). Natural materials.
PENDANT: Large WOVEN SEAGRASS or ROPE pendant (SIGNATURE!)
RUG: Natural JUTE, large, textured.
CENTERPIECE: Hurricane lantern (HERITAGE!) OR white coral sculpture.""",
                    "bedroom": """BED: WHITE linen upholstered OR RATTAN/CANE headboard (SIGNATURE!)
□ Light, BREEZY appearance
□ Relaxed beach-house feel

BEDDING: Crisp WHITE base, LIVED-IN linen texture. DOPAMINE accent pillows - CORAL, TURQUOISE, or YELLOW (ESSENTIAL!).
NIGHTSTANDS: RATTAN, WICKER, or ROPE-detailed (SIGNATURE!). Natural textures.
LAMPS: ROPE-wrapped base with linen shade (SIGNATURE!).
RUG: Natural JUTE or SISAL.
ACCESSORIES: DOPAMINE BRIGHT accent, seashells in bowl, coastal art.""",
                    "kitchen": """Fresh, breezy, LESS IS MORE:
- Weathered wood cutting board
- White ceramic with ROPE detail
- Lemons in WOVEN basket (DOPAMINE yellow!)
- NAUTICAL element - rope coil, lighthouse print
BAR STOOLS: WOVEN ROPE or SEAGRASS counter stools (SIGNATURE!)""",
                    "bathroom": """Spa-like coastal retreat:
- ROPE-trimmed mirror or accessories
- White/cream towels in WOVEN basket
- DOPAMINE accent - CORAL or TURQUOISE soap dish
- Eucalyptus stems, seashells
- NAUTICAL heritage element""",
                    "exterior": """LIGHTING: BRIGHT golden beach sunset OR brilliant blue-sky daylight.
- Warm glowing windows
LANDSCAPING: Coastal plants - palms, ornamental grasses. Weathered planters.
FURNITURE: Weathered teak or whitewashed wood. ROPE details. DOPAMINE accent cushions.""",
                    "room": """COASTAL 2026 = DOPAMINE BRIGHTS + ROPE/WOVEN + HYPER-BREEZY + HERITAGE:
SEATING: White/cream linen, slipcovered, relaxed
TABLES: ROPE-wrapped, weathered wood, driftwood
RUG: JUTE/SISAL essential - sandy texture
ACCESSORIES: DOPAMINE BRIGHT accent (coral/turquoise/yellow), ROPE textures, NAUTICAL heritage
PLANT: Tropical in woven seagrass basket"""
                }
                furniture = furniture_by_room_coastal.get(room_type, furniture_by_room_coastal["room"])

                return f"""COASTAL 2026 STAGING - "Hyper-Breezy Sensory Obsession": Stage this {room_type} with dopamine-inducing beach house joy.

{NANO_STRUCTURAL_RULES}

=============================================================================
⚠️⚠️⚠️ CAMERA AND STRUCTURE LOCK ⚠️⚠️⚠️
=============================================================================

🚫 CAMERA - ABSOLUTE LOCK:
- Maintain EXACT same camera position, angle, and field of view as original
- Do NOT rotate view left or right - same walls must be visible

🏠 ARCHITECTURE - ZERO CHANGES:
- ALL walls, windows, doors in EXACTLY the same positions
- Ceiling features UNCHANGED - NO track lighting added

🛋️ FURNITURE - SAME WALLS:
- If bed is on LEFT wall, staged bed goes on LEFT wall

=============================================================================

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- NEVER shift camera angle - maintain EXACT perspective
- HYPER-BREEZY means light, airy, relaxed - not cluttered

⚠️ STRUCTURAL PRESERVATION - NEVER alter walls, doorways, windows, or architectural features. NO added track lighting.

COASTAL 2026 STYLE DNA - "Hyper-Breezy Sensory Obsession":
- DOPAMINE BRIGHTS are ESSENTIAL - CORAL, TURQUOISE, or SUNNY YELLOW pop
- ROPE textures required (lamp base, basket, furniture detail)
- WOVEN natural materials - seagrass, rattan, wicker
- JUTE/SISAL rugs - sandy, textured, natural
- NAUTICAL HERITAGE elements - vintage buoys, maritime details, lighthouse motifs
- Bright, sun-drenched HYPER-BREEZY light (4000-5000K)
- Weathered wood, driftwood, whitewashed finishes

SIGNATURE ELEMENTS (MUST include 3-4):
□ DOPAMINE BRIGHT accent - coral, turquoise, or sunny yellow (ESSENTIAL!)
□ ROPE element - lamp, basket, furniture detail
□ WOVEN RATTAN or SEAGRASS piece
□ NAUTICAL HERITAGE detail (buoy, lighthouse, maritime)
□ Natural JUTE or SISAL rug

FORBIDDEN: Dark walnut, warm amber, brass, black iron, heavy cozy textures

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: DOPAMINE BRIGHTS + ROPE/WOVEN + HYPER-BREEZY + HERITAGE. Sun-drenched sensory joy with nautical soul."""

            # FARMHOUSE 2026 STYLE - "Neo-Farmhouse - Storied Sanctuary"
            # MUDDY PALETTE + BLACK IRON + PLASTERED/LIMEWASH + HACIENDA
            if style == "farmhouse":
                furniture_by_room_farmhouse = {
                    "living room": """FARMHOUSE 2026 - "Storied Sanctuary":
SOFA: Deep, substantial comfort in MUDDY TONES - mushroom, olive brown, warm clay. Heavyweight linen, LIVED-IN texture.
COFFEE TABLE: MASSIVE reclaimed wood with visible STORY (character marks, aged patina). NOT refinished.
ACCENT CHAIRS: Leather club chairs in aged cognac/saddle. OR linen wingback in muddy tone.
RUG: Vintage-style in FADED MUDDY TONES - aged aubergine, faded rust, muted olive. Shows generations of life.
ACCESSORIES (MUST INCLUDE 3-4):
□ BLACK IRON element (lamp, hardware, hooks) - ESSENTIAL!
□ CERAMIC PITCHER with dried botanicals (SIGNATURE!)
□ VINTAGE/antique piece with PROVENANCE
□ Grain sack or ticking stripe in MUDDY tones
□ Terracotta or CLAY vessel
PLANT: Dried botanicals in vintage PITCHER OR olive branches in clay pot. Natural, aged feel.
LIGHTING: BLACK IRON industrial lamp (SIGNATURE!). CANDLES in iron holders. Warm, flickering light.
ART: Vintage botanical prints OR aged mirrors in weathered frames.""",
                    "dining room": """TABLE: MASSIVE reclaimed wood farmhouse table (SIGNATURE!). Shows STORY - age marks, patina.
CHAIRS: Cross-back (X-back) in BLACK (SIGNATURE!) OR Windsor in aged black.
PENDANT: BLACK IRON chandelier (linear or candelabra style) - ESSENTIAL!
RUG: Vintage-style in FADED MUDDY palette.
CENTERPIECE: CERAMIC PITCHER with dried florals OR aged wooden dough bowl.""",
                    "bedroom": """BED: BLACK IRON bed frame (SIGNATURE!) OR massive reclaimed wood headboard.
□ Shows age/character in FURNITURE (STYLE)
□ Substantial, grounded presence

BEDDING: White linen base, LIVED-IN texture. VINTAGE QUILT in muddy tones at foot (SIGNATURE!). Layered pillows in clay/olive/cream.
NIGHTSTANDS: MISMATCHED VINTAGE pieces (PROVENANCE!) - aged, storied.
LAMPS: Ceramic in aged cream OR BLACK IRON candlestick.
RUG: FADED VINTAGE in muddy aubergine/olive tones.
ACCESSORIES: Iron candlestick, flowers in ceramic PITCHER, vintage leather-bound books.""",
                    "kitchen": """Storied rustic charm:
- Massive butcher block cutting board
- CERAMIC CROCKS with wooden utensils (SIGNATURE!)
- VINTAGE glass jars, aged containers
- Fresh produce in weathered basket
- BLACK IRON pot rack or hooks visible
BAR STOOLS: BLACK IRON industrial (SIGNATURE!) OR cross-back in aged black.""",
                    "bathroom": """Vintage hacienda charm:
- Aged wooden tray with artisan bar soap
- CLAY or terracotta vessels
- White linen towels on BLACK IRON ladder/hooks
- Aged galvanized metal or wire basket
- CANDLE in iron or clay holder""",
                    "exterior": """LIGHTING: WARM golden hour, HACIENDA glow.
- Windows showing warm candlelit interior
LANDSCAPING: Cottage garden - lavender, rosemary, heritage roses. TERRACOTTA and aged clay planters.
FURNITURE: Weathered wood rockers, aged metal bistro. BLACK IRON lanterns, string lights.""",
                    "room": """FARMHOUSE 2026 = MUDDY PALETTE + BLACK IRON + STORY + HACIENDA:
SEATING: Substantial comfort in mushroom/olive/clay tones
TABLES: Massive RECLAIMED wood with visible STORY
RUG: FADED VINTAGE in muddy palette
ACCESSORIES: BLACK IRON, CERAMIC PITCHERS, aged vintage pieces
PLANT: Dried botanicals in vintage vessels"""
                }
                furniture = furniture_by_room_farmhouse.get(room_type, furniture_by_room_farmhouse["room"])

                return f"""FARMHOUSE 2026 STAGING - "Neo-Farmhouse - Storied Sanctuary": Stage this {room_type} with soulful heritage warmth.

{NANO_STRUCTURAL_RULES}

=============================================================================
⚠️⚠️⚠️ CAMERA AND STRUCTURE LOCK ⚠️⚠️⚠️
=============================================================================

🚫 CAMERA - ABSOLUTE LOCK:
- Maintain EXACT same camera position, angle, and field of view as original
- Do NOT rotate view left or right - same walls must be visible

🏠 ARCHITECTURE - ZERO CHANGES:
- ALL walls, windows, doors in EXACTLY the same positions
- Ceiling features UNCHANGED - NO track lighting added

🛋️ FURNITURE - SAME WALLS:
- If bed is on LEFT wall, staged bed goes on LEFT wall

=============================================================================

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- CHARACTER in FURNITURE (worn, aged) = STYLE. Damage on WALLS = FRAUD.
- NEVER shift camera angle - maintain EXACT perspective

⚠️ STRUCTURAL PRESERVATION - NEVER alter walls, doorways, windows, or architectural features. NO added track lighting.

FARMHOUSE 2026 STYLE DNA - "Storied Sanctuary":
- MUDDY PALETTE is ESSENTIAL: mushroom, olive brown, aged aubergine, warm clay, faded rust
- BLACK IRON required (bed frames, chandeliers, hardware, hooks)
- CERAMIC PITCHERS with dried botanicals (SIGNATURE!)
- Massive RECLAIMED WOOD with visible STORY (age marks, patina)
- VINTAGE pieces with PROVENANCE - items that tell a story
- HACIENDA influence: terracotta, clay, limewash aesthetic
- Warm, candlelit light quality (2700K)

SIGNATURE ELEMENTS (MUST include 3-4):
□ BLACK IRON element (ESSENTIAL!)
□ CERAMIC PITCHER with dried botanicals
□ MASSIVE reclaimed wood with visible STORY
□ VINTAGE piece with provenance
□ MUDDY PALETTE accent (mushroom/olive/clay)
□ Terracotta or CLAY vessel

FORBIDDEN: Chrome, high-gloss, sleek modern, blonde wood, brass, bright colors

⚠️ DAMAGE PREVENTION:
- Show CHARACTER in FURNITURE (distressed, worn, aged) = AUTHENTIC STYLE
- Do NOT invent damage on WALLS (cracks, holes) = FRAUD
- Preserve wall condition EXACTLY

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: MUDDY PALETTE + BLACK IRON + STORY + HACIENDA. Soulful heritage sanctuary, not country kitsch."""

            # MID-CENTURY 2026 STYLE - "Retro-Futurism - Atomic Optimism"
            # DARK WALNUT + BOLD RETRO COLORS + TAPERED LEGS + BRASS/SPUTNIK
            if style == "midcentury":
                furniture_by_room_mcm = {
                    "living room": """MID-CENTURY 2026 - "Atomic Optimism":
SOFA: Low-profile in BOLD SATURATED COLOR - ATOMIC TANGERINE, AVOCADO GREEN, or MUSTARD GOLD (SIGNATURE!). TAPERED DARK WALNUT legs.
COFFEE TABLE: Surfboard or kidney shape in DARK WALNUT (SIGNATURE!). TAPERED LEGS essential. OR Noguchi-inspired.
ACCENT CHAIRS: Eames Lounge Chair in leather (ICONIC!). OR Womb Chair in bold fabric. OR Shell chairs in period colors.
RUG: SHAG in cream, GOLD, or AVOCADO (SIGNATURE!) OR bold geometric sunburst pattern.
ACCESSORIES (MUST INCLUDE 3-4):
□ SPUTNIK or STARBURST element (chandelier, clock, mirror) - ESSENTIAL SIGNATURE!
□ BOLD SATURATED COLOR accent - tangerine, avocado, mustard (ESSENTIAL!)
□ BRASS accents (lamp, candleholder, legs)
□ Sculptural ceramic in period color (atomic shapes)
□ BULLET PLANTER with architectural plant
PLANT: Snake plant or fiddle leaf in BULLET PLANTER (period ceramic!) in white, tangerine, or olive.
LIGHTING: SPUTNIK chandelier (SIGNATURE!) OR Arc floor lamp in BRASS. Brass is ESSENTIAL.
ART: Large abstract expressionist OR bold graphic atomic print.""",
                    "dining room": """TABLE: Oval DARK WALNUT with TAPERED LEGS (SIGNATURE!). OR Saarinen tulip.
CHAIRS: Eames molded plastic in BOLD colors OR Wishbone in DARK WALNUT. All matching.
PENDANT: SPUTNIK chandelier in BRASS (ESSENTIAL SIGNATURE!) OR PH Artichoke.
RUG: Bold geometric SUNBURST pattern OR SHAG in gold/avocado.
CENTERPIECE: Sculptural ceramic bowl in ATOMIC period color (tangerine, mustard).""",
                    "bedroom": """BED: DARK WALNUT platform with TAPERED LEGS (SIGNATURE!)
□ Low profile, panel/slat headboard
□ NO light wood - that's Scandinavian!
□ Iconic, substantial presence

BEDDING: White/cream base. BOLD SATURATED accent throw - ATOMIC TANGERINE, AVOCADO, or MUSTARD (ESSENTIAL!)
NIGHTSTANDS: DARK WALNUT with TAPERED LEGS and BRASS hardware (SIGNATURE!).
LAMPS: Ceramic in BOLD period color (tangerine, mustard, avocado). BRASS accents essential.
RUG: SHAG in cream, gold, or avocado.
ACCESSORIES: STARBURST clock or mirror (SIGNATURE!), BRASS candleholder, atomic ceramics.""",
                    "kitchen": """Atomic period aesthetic:
- Teak cutting board
- Ceramic canisters in BOLD period colors (tangerine, avocado, mustard)
- Fruit in atomic-shaped sculptural bowl
- Dansk or period Scandinavian ceramics
BAR STOOLS: DARK WALNUT with TAPERED LEGS. OR molded seats in BOLD period colors.""",
                    "bathroom": """Bold atomic period:
- Minimal tray with artisan soap
- Ceramic vessel in BOLD SATURATED period color (TANGERINE, AVOCADO, mustard)
- Snake plant in BULLET PLANTER
- Towels in bold solid color
- BRASS accents (essential!)""",
                    "exterior": """LIGHTING: WARM saturated golden hour OR dramatic atomic-era sunset.
- Rich, optimistic sky
- Interior windows glowing warm amber
LANDSCAPING: Desert modern (agave, architectural succulents). Gravel, concrete. Period planters.
FURNITURE: DARK WALNUT or teak. Clean lines. BOLD cushions in period colors.""",
                    "room": """MID-CENTURY 2026 = DARK WALNUT + BOLD SATURATED COLORS + TAPERED LEGS + SPUTNIK/BRASS:
SEATING: BOLD saturated color (tangerine/avocado/mustard), TAPERED walnut legs
TABLES: DARK WALNUT with TAPERED LEGS
RUG: SHAG or bold geometric SUNBURST
ACCESSORIES: SPUTNIK, STARBURST, BRASS, BULLET PLANTERS, atomic ceramics
PLANT: In ceramic BULLET PLANTER"""
                }
                furniture = furniture_by_room_mcm.get(room_type, furniture_by_room_mcm["room"])

                return f"""MID-CENTURY 2026 STAGING - "Retro-Futurism - Atomic Optimism": Stage this {room_type} with bold 1950s-60s optimism.

{NANO_STRUCTURAL_RULES}

=============================================================================
⚠️⚠️⚠️ CAMERA AND STRUCTURE LOCK ⚠️⚠️⚠️
=============================================================================

🚫 CAMERA - ABSOLUTE LOCK:
- Maintain EXACT same camera position, angle, and field of view as original
- Do NOT rotate view left or right - same walls must be visible

🏠 ARCHITECTURE - ZERO CHANGES:
- ALL walls, windows, doors in EXACTLY the same positions
- Ceiling features UNCHANGED - NO track lighting added

🛋️ FURNITURE - SAME WALLS:
- If bed is on LEFT wall, staged bed goes on LEFT wall

⚠️ CRITICAL RULES:
- NEVER invent wall damage, cracks, or imperfections
- Use DARK WALNUT - NOT light blonde wood (that's Scandinavian!)

MID-CENTURY 2026 STYLE DNA - "Atomic Optimism":
- DARK WALNUT with TAPERED LEGS (ESSENTIAL - NOT light blonde wood!)
- BOLD SATURATED COLORS: atomic tangerine, avocado green, mustard gold (NOT soft pastels!)
- SPUTNIK chandeliers and STARBURST motifs (SIGNATURE!)
- BRASS accents throughout (NOT black iron!)
- BULLET PLANTERS with architectural plants
- SHAG textures in period colors
- Rich, saturated warm lighting (2700-3000K)
- Optimistic, space-age aesthetic

SIGNATURE ELEMENTS (MUST include 3-4):
□ DARK WALNUT furniture with TAPERED LEGS (ESSENTIAL!)
□ BOLD SATURATED COLOR - tangerine, avocado, or mustard (ESSENTIAL!)
□ SPUTNIK chandelier or lighting (SIGNATURE!)
□ STARBURST element (clock, mirror, art)
□ BRASS accents
□ SHAG texture (rug or pillow)
□ Ceramic BULLET PLANTER

FORBIDDEN: Light blonde wood (Scandinavian!), soft pastels (Scandinavian!), black iron (Farmhouse!), chunky knits, chrome

FURNITURE:
{furniture}

CRITICAL: Keep architecture identical to original. Do NOT cover damage/defects.

Result: DARK WALNUT + BOLD SATURATED COLORS + TAPERED LEGS + SPUTNIK/BRASS. Atomic optimism meets timeless cool."""

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

{NANO_STRUCTURAL_RULES}

KEEP ARCHITECTURE UNCHANGED: Keep the exact same layout, walls, flooring, windows, ceiling, and all architectural features from the original photo. Do NOT move walls, change flooring material (carpet/hardwood/tile), or alter room dimensions.

Add realistically scaled furniture: {furniture}. Include tasteful decor like plants and art that match the {style} style. All furniture must be properly sized for this specific room - do NOT use oversized furniture to fake room size.

CRITICAL: Do NOT place any furniture, rugs, or decor to cover or hide any visible damage, stains, cracks, or wear on walls, floors, or ceiling. All defects must remain fully visible.

Level the photo so vertical lines are truly vertical. Do NOT move camera horizontally or rotate the view. Do NOT make the room appear larger.

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

⚠️ DAMAGE INVENTION PREVENTION (CRITICAL):
When removing items like TVs, wall art, or clutter:
- The wall/surface behind MUST appear CLEAN and UNDAMAGED
- Do NOT add mounting holes, screw marks, or discoloration where items were
- Do NOT invent paint chips, cracks, or marks where items were removed
- If removing a TV from a wall, that wall section becomes a CLEAN, NORMAL wall
- The ONLY damage allowed is damage CLEARLY VISIBLE in the original photo
- Creating fake damage is FRAUD and violates MLS compliance

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

⚠️ FINAL CHECK:
1. Verify all doorways, openings, and passages are preserved EXACTLY
2. Verify NO damage was invented where items were removed (TVs, art, etc.)
3. Any removed items leave CLEAN walls behind - no holes, marks, or discoloration

CRITICAL: Do NOT cover OR invent any damage/defects. Do NOT alter any architectural features.

Result: Magazine-cover worthy through lighting + styling, not structural changes. Room must be recognizable as the same space."""

            # Standard declutter fallback for occupied rooms
            return f"""VIRTUAL STYLING TASK: Clean up and enhance this {room_type} photo for a real estate listing.

{NANO_STRUCTURAL_RULES}

KEEP EVERYTHING UNCHANGED: Keep the exact same layout, walls, flooring, ceiling, and ALL major furniture exactly where it is. Do NOT remove or replace any furniture pieces.

Remove only loose clutter, trash, and personal items to make the space look tidy. You may add ONLY small coordinating decor items (throw pillows, a small plant) that complement existing furniture.

CRITICAL: Do NOT use any furniture, decor, or accessories to cover or hide any visible damage, stains, cracks, or wear. All defects must remain fully visible.

Level the photo so vertical lines are truly vertical. Do NOT move camera horizontally or rotate the view.

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
