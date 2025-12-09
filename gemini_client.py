"""
Gemini Client for image analysis and conservative cleanup prompt generation.

Uses Gemini vision model to analyze room photos and create conservative cleanup prompts.
Philosophy: "Clean up this photo" not "lie about what this house is."

We use conservative prompts that:
- Lock down structure (walls, windows, cabinets, flooring, main furniture)
- Only allow removable clutter to change
- Standardize photo quality (lighting, straightening, more professional, etc.)
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Optional

import httpx

from config import get_settings
from models import (
    Order, Plan, ImagePlan, GeminiAnalysisResult,
    ImageStatus
)

logger = logging.getLogger(__name__)


class GeminiPlannerClient:
    """
    Client for Gemini vision API to analyze room photos and generate staging prompts.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Gemini planner client.
        
        Args:
            api_key: Google API key. Uses config if not provided.
            base_url: Base URL for API. Uses config if not provided.
        """
        settings = get_settings()
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.base_url = base_url or settings.GEMINI_API_BASE_URL
        self.model = settings.GEMINI_VISION_MODEL
        self.timeout = settings.REQUEST_TIMEOUT
        
        logger.info(f"GeminiPlannerClient initialized with model: {self.model}")
    
    def _build_analysis_prompt(self, is_occupied: bool) -> str:
        """
        Build the system prompt for conservative image analysis.

        Philosophy: We're a "clean up this photo" tool, not a "lie about what this house is" tool.

        Uses a 4-paragraph prompt skeleton with room-type specific patterns and gold examples.
        """
        return f"""You are a real estate photo analyst creating CONSERVATIVE, MLS-safe cleanup prompts for Nano Banana (an AI image editor).

=== CORE PRINCIPLES ===

1. ANCHOR TO THE ACTUAL PHOTO
   Make it clear we're editing the uploaded image, not generating a new one.

2. LOCK DOWN STRUCTURE
   Explicitly list what must stay: layout, walls, windows, doors, cabinets, major furniture, flooring.

3. ONLY CHANGE REMOVABLE STUFF
   Declutter loose objects, trash, personal items, scattered decor.
   Light staging is okay, but only with simple, believable objects.

4. PROFESSIONAL PHOTO ENHANCEMENT
   Apply professional-level photographic improvements:
   - Correct exposure and increase contrast slightly for a crisp, well-defined look
   - Correct white balance to neutralize color casts (whites should look clean, not unnaturally bright)
   - Reduce noise and haze so the image looks sharp and high-quality like a DSLR photo
   - Subtly refine brightness and uniformity of walls/ceilings so they look freshly cleaned (but do NOT change paint color or hide damage)
   - Result should be photorealistic - no "AI art" vibe

5. CAMERA ANGLE ADJUSTMENTS (CRITICAL - READ CAREFULLY)
   We WANT the model to make the shot look like a professional real-estate photo. We do NOT want it to invent unseen parts of the room.

   ALLOWED (do these assertively):
   - Level the photo so the horizon is level and all vertical lines (walls, door frames, windows, posts) are truly vertical
   - Adjust the vertical angle (pitch) and apparent camera height - raise or lower the view for better composition
   - Gently correct roll/tilt for a more professional look
   - Minor framing adjustments that improve the composition without revealing new areas

   FORBIDDEN (never do these):
   - Moving the camera horizontally to a different corner or wall
   - Swinging the view (yaw) left/right to reveal walls, doors, windows, or surfaces not visible in the original
   - Making the room appear larger by widening the field of view or pulling the back wall farther away
   - Inventing or hallucinating parts of the room that were not photographed

   In plain terms: You may adjust vertical angle and camera height freely, but do not shift left/right or reveal new surfaces. Light lateral movement is acceptable ONLY if it does not expose made-up areas.

6. NEVER HIDE OR INVENT DAMAGE
   - Do NOT fix, repaint, patch, blur, or cover any damage to walls, ceilings, floors, doors, windows, trim, or fixtures
   - Keep any visible cracks, stains, holes, scuffs, peeling paint, or worn areas clearly visible and unchanged
   - Do NOT invent or add damage that is not in the original photo - if the walls are clean, keep them clean
   - The AI can clean up mess, but it CANNOT change the actual physical condition of the home
   - This is critical for MLS honesty - buyers must see the real condition, no better AND no worse

7. NO NEW PLANTS OR DECOR
   - Do NOT add plants, flowers, or greenery that are not already in the photo
   - Do NOT add new decorative items - only tidy or remove existing ones
   - Staging must use only items that are already present in the room

8. EMPHASIZE REALISM & MLS SAFETY
   No new architecture, no fake high-end finishes, no major repairs magically fixed.
   We clean up what a seller could do in 20 minutes - we do NOT lie about the house.

=== PROPERTY STATUS ===
This property is: {"OCCUPIED" if is_occupied else "VACANT"}

{"OCCUPIED means: Keep all existing furniture, cabinets, appliances, fixtures. Remove only loose clutter. Make beds with existing bedding. DO NOT replace or swap furniture." if is_occupied else "VACANT means: DO NOT add any furniture or staging items. Only improve lighting and straighten. Show the real empty room."}

=== 4-PARAGRAPH PROMPT SKELETON ===

Your staging_prompt MUST follow this exact structure:

PARAGRAPH 1 - Context & Goal:
"Using this uploaded [room type] photo, create a cleaner, brighter, professional real-estate listing image."

PARAGRAPH 2 - What Must Stay (be room-specific):
"Keep the same [layout + key structures for this room type] – for example, the walls, windows, doors, [major built-ins / cabinets / tub / appliances / main furniture] and overall proportions of the room. Do not move walls, add or remove windows, change flooring, or replace major furniture or fixtures."

PARAGRAPH 3 - Declutter + Light Staging (what can change):
{"For OCCUPIED: \"Virtually declutter by removing only loose, movable clutter such as [list specific loose/personal items you see] from [surfaces / floor / specific areas]. Do not add any new plants, flowers, or decorative items - only tidy or remove existing ones. Do not fix, repaint, or cover any damage to walls, ceilings, floors, doors, windows, trim, or fixtures; keep any visible cracks, stains, holes, scuffs, or worn areas clearly visible and unchanged. Do not invent or add damage that is not in the original photo. [Keep accessibility gear if present]. Straighten and neaten [bed / towels / cushions] while keeping their existing color and pattern.\"" if is_occupied else "For VACANT: \"Do not add furniture, plants, or staging items to this empty room. Do not fix, repaint, or cover any damage to walls, ceilings, floors, doors, windows, trim, or fixtures; keep any visible cracks, stains, holes, scuffs, or worn areas clearly visible and unchanged. Do not invent or add damage that is not in the original photo.\""}

PARAGRAPH 4 - Camera Adjustments & Professional Enhancement:
"LEVEL THE PHOTO: Straighten the image so the horizon is level and all vertical lines (walls, door frames, window frames) are truly vertical. This is critical for a professional result.

OPTIMIZE CAMERA HEIGHT & VERTICAL ANGLE: You may adjust the effective camera height (raise or lower the view) and vertical angle (pitch) to create a more professional real-estate composition. Be assertive with these adjustments - a professional realtor angle often improves the shot significantly.

CAMERA POSITION CONSTRAINTS: Do not move the camera horizontally to a different corner or wall. Do not swing the view left/right (yaw) to reveal walls, doors, windows, or surfaces not visible in the original. Light lateral adjustment is acceptable only if it does not expose made-up areas. The room must not appear larger than in the original photo - do not widen the field of view or pull the back wall farther away.

Apply professional-level photo enhancement: correct exposure, increase contrast slightly, and improve overall clarity so details in the flooring, trim, and fixtures are crisp and well defined. Correct white balance to neutralize any strong color cast so whites and light-colored surfaces appear clean and true-to-life, without making them unnaturally bright or pure white. Reduce noise and haze so the image looks sharp and high quality, similar to a DSLR real-estate photograph. You may subtly refine brightness and uniformity of the wall and ceiling paint so it looks freshly cleaned, but do not change the paint color or hide any visible damage, cracks, stains, or patches.

The final result must remain an honest representation of the property: no repairing or hiding defects, no fabricated improvements, no added plants or decor, and no inventing damage that does not exist. If the room is already tidy with little clutter, lean harder on leveling, camera height optimization, and photographic enhancements so the 'after' feels noticeably more polished than the 'before.' The result should be photorealistic and believable as a professionally photographed [room type] ready for an online home listing."

=== ROOM-TYPE PATTERNS ===

KITCHEN:
- Structural anchors: cabinets, countertops, appliances (fridge, stove, microwave, dishwasher), sink, island/peninsula, flooring
- Declutter: dishes, drying racks, bottles, spice jars, small appliances overload, mail, trash cans, random decor, floor clutter
- Allowed staging: tidy existing items only, no adding plants or new decor
- DAMAGE: Keep any visible damage (scuffs, stains, worn areas) unchanged

BATHROOM:
- Structural anchors: vanity, sink, mirror, toilet, tub/shower, wall tile, main storage cabinets
- Declutter: toiletries, toothbrushes, meds, cleaning products, tissue boxes, trash, piles of towels
- IMPORTANT: Keep accessibility equipment (grab bars, shower chairs) but make area around them tidy
- Allowed staging: neatly fold existing towels, tidy existing items - no adding plants or new decor
- DAMAGE: Keep any visible damage (cracks, stains, worn grout) unchanged

BEDROOM:
- Structural anchors: bed, headboard, nightstands, dresser, desk (if present), closets, flooring, window positions
- Declutter: clothes piles, random boxes, cords, bedside junk, over-stuffed surfaces
- Allowed staging: made bed with SAME bedding, arrange existing pillows neatly, tidy existing items - no adding plants or new decor
- DAMAGE: Keep any visible damage (scuffs, stains, worn carpet) unchanged

LIVING ROOM / DEN:
- Structural anchors: sofa(s), main chairs, coffee table, media console/TV, rugs, main shelving, windows, doors
- Declutter: blankets piled, remotes, food packaging, excessive personal photos (reduce count but keep some), random small decor, visible wires
- Allowed staging: fold existing throw, arrange existing cushions, tidy existing items - no adding plants or new decor
- DAMAGE: Keep any visible damage (scuffs, stains, worn areas) unchanged

HALLWAY:
- Structural anchors: doors, trim, runners, vents
- Declutter: wall clutter, hanging items, anything sticking into hallway
- Allowed staging: keep 1-2 existing wall pieces so it doesn't feel like a hospital corridor - no adding plants or new decor
- DAMAGE: Keep any visible damage (scuffs, stains, worn areas) unchanged

EXTERIOR / PORCH:
- Structural anchors: siding, roofline, windows, porch structure, stairs & railings, driveway, shrubs/trees
- Declutter: bins, loose tools, random stuff under carport, scooters, hoses (keep cars & bins if neatly arranged)
- Allowed: tidy existing landscaping only - do NOT add plants or make grass greener than reality
- DAMAGE: Keep any visible damage (peeling paint, cracks, worn siding) unchanged

=== GOLD EXAMPLES ===

EXAMPLE A - Cluttered Kitchen:
"Using this uploaded kitchen photo, create a cleaner, brighter, professional real-estate listing image.

Keep the same room layout, ceiling, walls, flooring, white cabinets, countertops, refrigerator, stove, microwave, dishwasher, sink, window, island, and dining table in their current positions. Do not move walls, add windows, change the flooring, or replace major furniture or appliances.

Virtually declutter by removing only loose, movable clutter from the island, folding table, countertops, and dining table, including cups, bottles, dishes, paper products, cleaning supplies, and small decorative objects. Clear the area around the sink and stove of most dishes and utensils. Remove any floor clutter such as bins and pet items so the floor appears clean and open. Keep the dining table and chairs but simplify the table setting. Do not add any new plants, flowers, or decorative items - only tidy or remove existing ones. Do not fix, repaint, or cover any damage to walls, ceilings, floors, cabinets, or fixtures; keep any visible cracks, stains, scuffs, or worn areas clearly visible and unchanged. Do not invent or add damage that is not in the original photo.

LEVEL THE PHOTO: Straighten the image so the horizon is level and all vertical lines (walls, door frames, cabinet edges, window frames) are truly vertical. This is critical for a professional result.

OPTIMIZE CAMERA HEIGHT & VERTICAL ANGLE: You may adjust the effective camera height (raise or lower the view) and vertical angle to create a more professional real-estate composition - for example, raising it slightly to show more countertop and less ceiling. Be assertive with these adjustments.

CAMERA POSITION CONSTRAINTS: Do not move the camera horizontally to a different corner or wall. Do not swing the view left/right to reveal walls, windows, or surfaces not visible in the original. Light lateral adjustment is acceptable only if it does not expose made-up areas. The kitchen must not appear larger than in the original photo - do not widen the field of view or pull the back wall farther away.

Apply professional-level photo enhancement: correct exposure, increase contrast slightly, and improve overall clarity so details in the flooring, cabinets, and fixtures are crisp and well defined. Correct white balance to neutralize any yellow color cast so whites and light-colored surfaces appear clean and true-to-life, without making them unnaturally bright. Reduce noise and haze so the image looks sharp and high quality, similar to a DSLR real-estate photograph. You may subtly refine brightness and uniformity of the wall and ceiling paint so it looks freshly cleaned, but do not change the paint color or hide any visible damage.

The final result must remain an honest representation of the property: no repairing or hiding defects, no fabricated improvements, no added plants or decor, and no inventing damage that does not exist. The result should be photorealistic and believable as a professionally photographed kitchen ready for an online home listing."

EXAMPLE B - Bedroom with TV:
"Using this uploaded bedroom photo, create a cleaner, brighter, professional real-estate listing image.

Keep the same room layout, walls, carpet, bed position, TV and TV stand, window size and placement, curtains, dresser, and overall proportions of the room. Do not move walls, add or remove windows, change flooring, or replace major furniture pieces.

Virtually declutter by removing only loose, movable clutter from the rolling tray beside the bed, the top of the TV stand, the tables in front of the window, and the dresser, including cups, bottles, baskets, boxes, toiletries, and miscellaneous small objects. Leave the tray table in place but clear it of clutter. Smooth and neatly arrange the bedding so the comforter and pillows look tidy and well made, keeping the existing purple pattern and colors. Do not add any new plants, flowers, or decorative items - only tidy or remove existing ones. Do not fix, repaint, or cover any damage to walls, ceilings, floors, or fixtures; keep any visible cracks, stains, scuffs, or worn areas clearly visible and unchanged. Do not invent or add damage that is not in the original photo.

LEVEL THE PHOTO: Straighten the image so the horizon is level and all vertical lines (walls, door frames, window frames, furniture edges) are truly vertical. This is critical for a professional result.

OPTIMIZE CAMERA HEIGHT & VERTICAL ANGLE: You may adjust the effective camera height (raise or lower the view) and vertical angle to create a more professional real-estate composition - for example, lowering it slightly if there is too much ceiling in the frame. Be assertive with these adjustments.

CAMERA POSITION CONSTRAINTS: Do not move the camera horizontally to a different corner or wall. Do not swing the view left/right to reveal walls, windows, or surfaces not visible in the original. Light lateral adjustment is acceptable only if it does not expose made-up areas. The bedroom must not appear larger than in the original photo - do not widen the field of view or pull the back wall farther away.

Apply professional-level photo enhancement: correct exposure, increase contrast slightly, and improve overall clarity so details in the carpet, trim, and fixtures are crisp and well defined. Correct white balance to neutralize any strong color cast so whites and light-colored surfaces appear clean and true-to-life, without making them unnaturally bright. Reduce noise and haze so the image looks sharp and high quality, similar to a DSLR real-estate photograph. You may subtly refine brightness and uniformity of the wall and ceiling paint so it looks freshly cleaned, but do not change the paint color or hide any visible damage.

The final result must remain an honest representation of the property: no repairing or hiding defects, no fabricated improvements, no added plants or decor, and no inventing damage that does not exist. The result should be photorealistic and believable as a professionally photographed bedroom ready for an online home listing."

EXAMPLE C - Exterior with Carport:
"Using this uploaded exterior photo, create a cleaner, brighter, professional real-estate listing image of the home.

Keep the same house structure, siding, windows, shutters, roofline, carport, posts, front steps, railing, shrubs, driveway, and lawn. Do not change the architecture, move windows or doors, or alter the basic materials of the home.

Virtually declutter by removing or neatly organizing only loose, portable items under and around the carport, such as trash bins, scooters, loose boards, and small tools, so the area appears tidy and well maintained. You may keep one or two vehicles parked in the driveway, but they should look neatly positioned without drawing attention away from the house. Do not add any new plants or make the grass greener than it actually is - only tidy existing landscaping. Do not fix, repaint, or cover any damage to siding, roofing, steps, railings, or other structures; keep any visible peeling paint, cracks, worn areas, or damage clearly visible and unchanged. Do not invent or add damage that is not in the original photo.

LEVEL THE PHOTO: Straighten the image so the horizon is level and all vertical lines (walls, door frames, window frames, posts, railings) are truly vertical. This is critical for a professional result.

OPTIMIZE CAMERA HEIGHT & VERTICAL ANGLE: You may adjust the effective camera height (raise or lower the view) and vertical angle to create a more professional real-estate composition - for example, raising it slightly to show the home from a more flattering angle, or lowering it if too much sky dominates. Be assertive with these adjustments.

CAMERA POSITION CONSTRAINTS: Do not move the camera horizontally to a different vantage point. Do not swing the view left/right to reveal portions of the property not visible in the original. Light lateral adjustment is acceptable only if it does not expose made-up areas. The home must not appear larger than in the original photo - do not widen the field of view or pull structures farther away.

Apply professional-level photo enhancement: correct exposure, increase contrast slightly, and improve overall clarity so details in the siding, trim, and landscaping are crisp and well defined. Correct white balance to neutralize any strong color cast so the siding, trim, and sky appear true-to-life, without making them unnaturally bright or saturated. Reduce noise and haze so the image looks sharp and high quality, similar to a DSLR real-estate photograph. You may subtly refine brightness and uniformity of painted surfaces so they look freshly cleaned, but do not change the paint color or hide any visible damage.

The final result must remain an honest representation of the property: no repairing or hiding defects, no fabricated improvements, no added plants or artificially greener grass, and no inventing damage that does not exist. The result should be photorealistic and believable as a professionally photographed exterior listing image."

=== YOUR TASK ===

Analyze the uploaded photo and respond with ONLY valid JSON (no markdown, no code blocks):
{{
    "room_type": "kitchen|bathroom|bedroom|living_room|dining_room|hallway|exterior|office|other",
    "is_occupied": true|false,
    "issues": ["list", "specific", "issues", "you", "see"],
    "suggested_style": "conservative_cleanup",
    "staging_prompt": "your full 4-paragraph prompt following the skeleton above"
}}

Issues to identify: clutter, dim_lighting, crooked_angle, personal_items, messy_bed, items_on_floor, crowded_counters, visible_wires, strong_color_cast, etc.

IMPORTANT: Be SPECIFIC in your staging_prompt. List the actual items you see that need to be removed. Reference the actual colors and materials visible. The more specific, the better the result.
"""
    
    async def analyze_image(
        self,
        image_path: Path,
        is_occupied: bool,
        max_retries: int = 3
    ) -> GeminiAnalysisResult:
        """
        Analyze a single image and generate conservative cleanup prompt.

        Args:
            image_path: Path to the image file
            is_occupied: Whether the room has existing furniture (determines cleanup vs empty treatment)
            max_retries: Number of retries on transient failures

        Returns:
            GeminiAnalysisResult with room analysis and conservative cleanup prompt
        """
        # Read and encode image
        image_bytes = image_path.read_bytes()
        image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Determine mime type
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        url = f"{self.base_url}/models/{self.model}:generateContent"
        system_prompt = self._build_analysis_prompt(is_occupied)

        last_error = None

        for attempt in range(max_retries):
            try:
                request_body = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": system_prompt},
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": image_base64
                                    }
                                },
                                {"text": "Analyze this room photo and provide the JSON response."}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 65536,  # No artificial limits - let the model work
                    }
                }

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

                # Check for truncation
                finish_reason = result.get("candidates", [{}])[0].get("finishReason", "")
                if finish_reason == "MAX_TOKENS":
                    logger.warning(f"Response hit MAX_TOKENS on attempt {attempt + 1}, retrying...")
                    last_error = ValueError("Response truncated due to MAX_TOKENS")
                    continue

                # Extract text response
                try:
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError) as e:
                    logger.warning(f"Unexpected response format on attempt {attempt + 1}: {result}")
                    last_error = ValueError(f"Failed to parse Gemini response: {e}")
                    continue

                # Parse JSON response
                analysis = self._parse_json_response(text)
                if attempt > 0:
                    logger.info(f"Successfully analyzed on attempt {attempt + 1}")
                return GeminiAnalysisResult(**analysis)

            except ValueError as e:
                logger.warning(f"Parse error on attempt {attempt + 1}: {e}")
                last_error = e
                continue
            except Exception as e:
                logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
                last_error = e
                continue

        # All attempts failed
        raise last_error or ValueError("All analysis attempts failed")
    
    def _parse_json_response(self, text: str) -> dict:
        """
        Parse JSON from Gemini response, handling common issues.
        
        Args:
            text: Raw text response from Gemini
            
        Returns:
            Parsed JSON dict
        """
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed, attempting repair: {e}")
            
            # Try to fix common issues
            # Remove trailing commas before closing braces/brackets
            text = re.sub(r",\s*}", "}", text)
            text = re.sub(r",\s*]", "]", text)
            
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON after repair: {text}")
                raise ValueError(f"Could not parse Gemini response as JSON: {text[:200]}")
    
    async def analyze_and_plan_images(
        self,
        job_id: str,
        job_dir: Path,
        order: Order,
        image_paths: list[str]
    ) -> Plan:
        """
        Analyze all images for a job and create conservative cleanup prompts.

        Args:
            job_id: Job identifier
            job_dir: Job directory path
            order: Order with job metadata (uses occupied flag)
            image_paths: List of relative paths to raw images

        Returns:
            Plan with analysis and conservative cleanup prompts for all images
        """
        logger.info(f"Starting conservative analysis for job {job_id} with {len(image_paths)} images")
        logger.info(f"Property occupied: {order.occupied}")

        plan = Plan(job_id=job_id, images=[])

        for i, rel_path in enumerate(image_paths):
            image_id = f"img_{i+1}"
            abs_path = job_dir / rel_path

            logger.info(f"Analyzing image {image_id}: {rel_path}")

            try:
                result = await self.analyze_image(
                    image_path=abs_path,
                    is_occupied=order.occupied
                )

                image_plan = ImagePlan(
                    id=image_id,
                    source_path=rel_path,
                    room_type=result.room_type,
                    is_occupied=result.is_occupied,
                    issues=result.issues,
                    nano_prompt=result.staging_prompt,
                    status=ImageStatus.PLANNED
                )

            except Exception as e:
                logger.error(f"Failed to analyze {rel_path}: {e}")
                image_plan = ImagePlan(
                    id=image_id,
                    source_path=rel_path,
                    status=ImageStatus.FAILED,
                    error_message=str(e)
                )

            plan.images.append(image_plan)

        logger.info(f"Completed analysis for job {job_id}")
        return plan
