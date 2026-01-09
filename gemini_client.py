"""
Gemini Client for image analysis and virtual staging prompt generation.

Uses Gemini vision model to analyze room photos and create professional virtual staging prompts.
Philosophy: "STRIP AND REFURNISH - remove all furniture and restage from scratch."

We use intelligent staging prompts that:
- STRIP all existing furniture and decor (whether vacant or occupied)
- REFURNISH with stylish furniture appropriate to the room type and selected style
- Apply style-specific design briefs (Modern, Scandinavian, Coastal, Farmhouse, Mid-Century Modern, Architecture Digest)
- Maintain structural honesty (walls, windows, flooring unchanged, damage visible)
- Apply professional photo enhancement (lighting, straightening, color correction)
- Keep furniture realistically scaled - never fake room dimensions
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
    ImageStatus, StylePreference
)

logger = logging.getLogger(__name__)


# =============================================================================
# STRUCTURAL PRESERVATION RULES - APPLIES TO ALL STYLES
# These rules prevent MLS compliance violations and potential lawsuits
# =============================================================================

STRUCTURAL_PRESERVATION_RULES = """
════════════════════════════════════════════════════════════════════════════════
⚠️ ABSOLUTE STRUCTURAL PRESERVATION RULES - VIOLATION = UNUSABLE OUTPUT ⚠️
════════════════════════════════════════════════════════════════════════════════

You are editing an EXISTING real estate photo. The output must be legally usable
for MLS listings. Any structural misrepresentation could result in lawsuits.

🔒 CAMERA POSITION (LOCKED - NO CHANGES ALLOWED):
• Maintain EXACT camera position, angle, and rotation
• Do NOT rotate the view left or right (no yaw changes)
• Do NOT tilt the view up or down beyond minor leveling
• Do NOT move the camera forward, backward, or sideways
• The SAME WALLS must be visible in the output as in the input
• If you can see the right wall in the original, you must see it in the output
• If a wall is NOT visible in the original, it should NOT appear in the output

🔒 FLOORING (LOCKED - NO CHANGES ALLOWED):
• If the original has CARPET, the output MUST have carpet
• If the original has HARDWOOD, the output MUST have hardwood
• If the original has TILE, the output MUST have tile
• If the original has LAMINATE, the output MUST have laminate
• If the original has CONCRETE, the output MUST have concrete
• Do NOT change flooring material, color, or pattern
• The floor is part of the property's actual features - changing it is FRAUD

🔒 WINDOWS (LOCKED - NO CHANGES ALLOWED):
• Windows must stay in EXACT same positions
• Windows must stay the EXACT same size
• Windows must stay the EXACT same shape and style
• Do NOT turn small windows into floor-to-ceiling windows
• Do NOT add windows that don't exist
• Do NOT remove windows that do exist
• Window TREATMENTS (curtains, blinds) can change - window STRUCTURE cannot

🔒 WALLS & OPENINGS (LOCKED - NO CHANGES ALLOWED):
• All walls must remain in exact positions
• All doorways must remain in exact positions and sizes
• All archways and openings must be preserved
• Wall COLOR can be subtly adjusted for lighting, but structure cannot change
• Do NOT add or remove walls, niches, or built-ins

🔒 CEILING (LOCKED - NO CHANGES ALLOWED):
• Ceiling height must remain the same
• Crown molding must be preserved if present
• Do NOT add track lighting, beams, or fixtures that don't exist
• Do NOT remove ceiling features that do exist
• Recessed lights in original must stay; don't add new ones

🔒 ROOM DIMENSIONS (LOCKED - NO CHANGES ALLOWED):
• Room must appear the SAME SIZE as original
• Do NOT use wide-angle distortion to make room look bigger
• Do NOT crop to hide parts of the room
• Proportions must match exactly

📋 WHAT YOU CAN CHANGE:
• Furniture (replace, add, remove, restyle)
• Decor (art, plants, accessories, pillows, throws)
• Window treatments (curtains, drapes, blinds - but not window structure)
• Lighting QUALITY (brightness, warmth, color temperature)
• Bedding, rugs, and soft furnishings
• Minor wall color/tone adjustment for style cohesion

════════════════════════════════════════════════════════════════════════════════
"""

STRUCTURAL_VERIFICATION_REMINDER = """
────────────────────────────────────────────────────────────────────────────────
⚠️ FINAL STRUCTURAL CHECK - VERIFY BEFORE OUTPUTTING ⚠️
────────────────────────────────────────────────────────────────────────────────

Before generating your output, verify ALL of these are true:

□ CAMERA: Same walls visible as original (camera didn't rotate)
  - If right wall was visible, it's still visible
  - If left wall was hidden, it's still hidden

□ FLOORING: Same flooring material as original
  - Carpet stayed carpet (NOT changed to hardwood)
  - Hardwood stayed hardwood (NOT changed to carpet)
  - Tile stayed tile

□ WINDOWS: Same window sizes and positions
  - No small windows became floor-to-ceiling
  - No windows were added or removed
  - Window frames in same locations

□ PROPORTIONS: Room appears same size as original
  - Not wider, not deeper, not taller
  - No wide-angle distortion added

□ WALLS: All walls in original positions
  - No walls moved, added, or removed
  - Doorways and openings unchanged

If ANY of these checks fail, the image is INVALID for MLS use.
Regenerate while preserving the original structure.

────────────────────────────────────────────────────────────────────────────────
"""


class GeminiPlannerClient:
    """
    Client for Gemini vision API to analyze room photos and generate virtual staging prompts.

    Supports 6 staging styles (Modern, Scandinavian, Coastal, Farmhouse, Mid-Century Modern, Architecture Digest).
    Uses STRIP AND REFURNISH approach: all rooms get furniture removed and restaged from scratch,
    regardless of whether they're currently vacant or occupied.
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
    
    def _build_analysis_prompt(self, is_occupied: bool, style_preference: str = "modern", comments: str = None) -> str:
        """
        Build the system prompt for virtual staging image analysis.

        Philosophy: STRIP AND REFURNISH - regardless of whether the room is vacant or occupied,
        we remove all existing furniture and decor, then restage from scratch in the selected style.
        Occupancy only describes the input state, not the behavior.

        Uses a 4-paragraph prompt skeleton with room-type specific staging patterns and gold examples.
        """
        # ============================================================================
        # UNIVERSAL RULES (Apply to ALL styles) - 2026 Enhanced
        # ============================================================================
        universal_rules = """
⚠️ CRITICAL RULES FOR ALL STYLES:

STRUCTURAL INTEGRITY:
- NEVER invent, add, or enhance wall damage, cracks, stains, or imperfections
- NEVER alter walls, doorways, windows, or architectural openings
- If original has no damage, output MUST have no damage
- Preserve exact room dimensions and proportions

⚠️ DAMAGE INVENTION PREVENTION (CRITICAL):
- When removing items (TVs, furniture, clutter), the surface behind them must appear CLEAN and UNDAMAGED
- NEVER invent holes, mounting marks, discoloration, or damage where items were removed
- If a TV is removed from a wall, that wall section must look like a normal, clean wall
- If furniture is removed, the floor/wall behind it must be pristine
- The ONLY damage allowed in output is damage that was CLEARLY VISIBLE in the original photo
- Creating fake damage is FRAUD and violates MLS compliance

CAMERA PRESERVATION:
- Maintain EXACT camera position and angle from original
- Do not shift, tilt, rotate, or reframe the view
- No horizontal movement, no yaw rotation
- Furniture must fit original perspective naturally

PLANT DISCIPLINE:
- NOT every room needs a plant
- Maximum 1-2 plants per room, vary types across property
- Some rooms should have ZERO plants
- Plants must match the specific style aesthetic

BED/FURNITURE POSITIONING:
- Keep major furniture in approximately same position as logical for room
- Do not dramatically shift furniture to different walls
- Maintain room's original layout logic and flow
"""

        # ============================================================================
        # STYLE-SPECIFIC DESIGN BRIEFS - 2026 RADICAL DIFFERENTIATION
        # Each style has UNIQUE DNA that is IMMEDIATELY RECOGNIZABLE
        # ============================================================================
        style_guides = {
            # ============================================================================
            # MODERN 2026: "Ultra-Simple Holographic Minimalism"
            # COOL + ETHEREAL + SCULPTURAL + VOID
            # ============================================================================
            "modern": {
                "description": "2026 Vision: Ultra-simple minimalist manifesto with holographic, ethereal atmosphere. Digital perfection meets organic unpredictability. Post-material sculptural forms. The OPPOSITE of warm AD.",
                "palette": "Pure white, concrete gray, deep charcoal, black. ONE iridescent/holographic element. NO warm earth tones.",
                "colors": "Foundation: pure white, concrete gray, deep charcoal, BLACK. Materials: brushed steel, chrome, black matte, polished nickel, glass, resin, concrete. NO warm wood tones, NO brass, NO earth tones.",

                "signature_lighting": """MODERN 2026 LIGHTING - "The Architecture of Silence":
- COOL, ethereal light quality (4500-5500K)
- Crisp white light with subtle holographic shimmer
- Light as sculptural element - visible rays creating geometric patterns
- Dramatic interplay of deep shadow and brilliant illumination
- "Empty light backgrounds" contrasting with "rough implied textures"
- Glass boundaries dissolving indoor/outdoor
- NO warm amber tones - this is the OPPOSITE of AD""",

                "signature_atmosphere": """MODERN 2026 ATMOSPHERE - "The Triangular Void":
- Empty space is as important as furniture - "negative space as design"
- Monochromatic foundations with stark contrasts
- "Geospatial Extremes" - exaggerated sharp angles and diagonals
- Dynamism and movement in static imagery
- Biophilic connection - nature pulled INTO the space
- Ethereal, almost "digital" quality to the render""",

                "furniture": """MODERN 2026 FURNITURE - "Post-Material Sculptural Forms":

THE PHILOSOPHY: Furniture is "post-material" - liquid forms, amorphous shapes,
unexpected material hybrids. Tactile contrasts.

BED:
□ Low-profile platform that appears to FLOAT (gravity-defying)
□ Headboard: Sculptural curved form OR dramatic angular geometry
□ Materials: Matte lacquer, brushed metal, OR unexpected hybrid
□ Colors: Pure white, charcoal, or iridescent/shifting tones
□ NO traditional wood grain - too conventional

NIGHTSTANDS:
□ Sculptural pedestals - geometric or organic blob forms
□ Materials: Glass, polished metal, concrete, resin
□ Could be asymmetric or mismatched intentionally
□ One floating shelf, other sculptural object

SEATING (if space allows):
□ Amorphous, "liquid" form chair - looks poured rather than built
□ OR severe geometric sculptural seat
□ Plush curves against cold industrial backdrop

RUG:
□ SOLID color - pure white, deep charcoal, or concrete gray
□ OR bold geometric with sharp diagonals
□ Low pile, architectural edges
□ Could be asymmetric or custom-shaped""",

                "decor": """MODERN 2026 SIGNATURES (include 2-3):
□ Floor-to-ceiling glass with indoor-outdoor dissolution
□ ONE sculptural furniture piece as "functional art"
□ Diagonal compositional lines creating dynamism
□ Stark light/shadow interplay
□ Empty space as intentional design element
□ Material hybrid (unexpected combination like wicker against concrete)
□ Floating/gravity-defying element

ART:
□ Large-scale minimal - single bold gesture or empty canvas
□ OR holographic/iridescent piece
□ OR nothing - the architecture IS the art

FORBIDDEN: Warm wood tones, earth tones, cozy textures, traditional patterns, brass""",

                "rugs": "SOLID color - pure white, deep charcoal, concrete gray. OR bold geometric with sharp diagonals. Architectural edges, could be asymmetric.",

                "lighting": """MODERN 2026 LIGHTING FIXTURES:
- Sculptural pendants in chrome, white, or black
- Track lighting or recessed - gallery aesthetic
- NO brass, NO warm metals
- Light itself as sculptural element""",

                "overall": "ETHEREAL + SCULPTURAL + COOL + VOID. Post-material minimalism with holographic shimmer.",

                "plants": """MODERN 2026 PLANTS (SELECTIVE - be dramatic):
□ Single architectural specimen - sculptural, dramatic
□ In geometric container (cube, sphere, cylinder) - white/black/chrome
□ Snake plant, bird of paradise, or single dramatic branch
□ NOT cozy or organic - sharp and intentional
□ SKIP plants in most rooms - let the void breathe""",

                "what_makes_it_different": """MODERN 2026 vs OTHER STYLES:

vs AD: COOL ethereal light (not warm golden), post-material furniture
       (not natural wood), holographic shimmer (not earth tones)

vs Coastal: No natural textures, no blue, no woven materials,
            sharp geometry (not soft/breezy)

vs Farmhouse: Opposite aesthetic - sleek vs rustic, cool vs warm,
              synthetic vs natural, future vs heritage

vs Scandinavian: No cozy hygge, no blonde wood, no soft textiles,
                 dramatic (not serene), angular (not curved)

vs Mid-Century: No retro nostalgia, no walnut, no bold primary colors,
                post-material (not vintage), ethereal (not saturated)

MODERN 2026 = ETHEREAL + SCULPTURAL + COOL + VOID""",

                "structural_rules": """⚠️ CRITICAL - STRUCTURAL PRESERVATION:
NEVER alter walls, doorways, openings, or architectural features.
NEVER invent damage, cracks, or imperfections that don't exist.
If there is an opening to another room, it MUST remain.
Identify ALL openings before generating - preserve exactly."""
            },

            # ============================================================================
            # SCANDINAVIAN 2026: "Nordic Ethereal - Spiritual Hygge"
            # BLONDE WOOD + EARTH-SHADOWS + SHEEPSKIN + CANDLES
            # ============================================================================
            "scandinavian": {
                "description": "2026 Vision: Nordic Ethereal - radical preservation of light, spiritual pursuit of Hygge. Monastic devotion to Quiet Luxury. Every pixel engineered to lower heart rate.",
                "palette": "Stone gray, sage, dried-bone beige, blue-white Nordic glow. LIGHT woods ONLY.",
                "colors": "Atmospheric Neutrals: stone gray, sage (muted dusty), dried-bone beige, blue-white. Wood: LIGHT ONLY - bleached ash, raw oak, light birch, knotted pine. Accents: dusty rose, pale sage, soft blue-gray. NO bold colors.",

                "signature_lighting": """SCANDINAVIAN 2026 LIGHTING - "The Aqueous Atmosphere":
- Captures the specific BLUE-WHITE glow of Nordic winter
- "Atmospheric Neutrals" - not just white, a symphony
- Diffused brilliance - shadows obliterated
- AI-rendered sheer GHOST-LIKE linen curtains catching light
- Light sprayed softly across the room
- "The room feels as though it is EXHALING"
- Fireplace glow as the room's "soul" (if applicable)
- Cool but not cold - serene, spiritual""",

                "signature_atmosphere": """SCANDINAVIAN 2026 ATMOSPHERE - "The Luminous Void":
- "Negative space as structural element"
- Vacuum of clutter - room is breathing
- Sharp corners replaced by gentle "eroded" curves
- Mimicking stones smoothed by a river
- "Tactile rebellion against the digital"
- Raw and honest materials
- "Heavily Light" - substantial presence but weightless feel
- Framed wildness through windows""",

                "furniture": """SCANDINAVIAN 2026 FURNITURE - "Living Texture Manifesto":

THE PHILOSOPHY: Raw and honest materials. Imperfection of the earth.
"Cøzy Contrast" - tactile weights preventing coldness.
Cloud-like upholstery. Spiritual silence.

⚠️ CRITICAL: Use LIGHT BLONDE WOOD (birch, bleached ash, raw oak)
NOT dark walnut - that's Mid-Century!

BED:
□ LIGHT ASH or BLEACHED OAK frame - raw, unfinished appearance
□ OR "cloud-like" upholstered bed - soft sculptural curves
□ Gentle eroded curves, not sharp angles
□ "Stones smoothed by river" aesthetic
□ Light, airy, breathable presence

NIGHTSTANDS:
□ BLEACHED ASH or RAW OAK (unfinished appearance)
□ Simple, functional Scandinavian design
□ OR floating shelf in light wood
□ Knotted wood visible - "imperfection of earth"
□ NOT dark, NOT walnut

SEATING:
□ "Cloud-like" sculptural upholstery
□ Chunky, soft, inviting form
□ In cream, gray, or "earth-shadow" tones
□ Sheepskin draped casually

RUG:
□ Knotted wool SO THICK you can feel warmth
□ OR layered sheepskin (SIGNATURE!)
□ Heavy monochromatic knits
□ Textural, tactile, warm underfoot""",

                "decor": """SCANDINAVIAN 2026 SIGNATURES (must include 2-3):
□ SHEEPSKIN texture - draped or rug (ESSENTIAL!)
□ CHUNKY KNIT throw - "accidentally" draped, not styled
□ LIGHT BLONDE WOOD furniture (not dark!)
□ Single-specimen botanical - ONE dramatic branch in hand-blown glass
□ CANDLES - room's hygge soul, essential!
□ Fireplace glow if applicable
□ Knotted, unfinished wood texture
□ Linen - heavy monochromatic

FORBIDDEN: Dark walnut (MCM!), bold colors, bright accents, busy patterns, brass""",

                "rugs": "Knotted wool SO THICK you can feel warmth. OR layered sheepskin. Heavy monochromatic knits. Textural, tactile.",

                "lighting": """SCANDINAVIAN 2026 LIGHTING FIXTURES:
- Paper pendant or globe lamp diffusing light
- CANDLES throughout (ESSENTIAL!)
- Soft fabric shades
- Fireplace as room's soul
- Brass/copper accents subtle""",

                "overall": "BLONDE WOOD + EARTH-SHADOWS + SHEEPSKIN + CANDLES + HYGGE. Spiritual serenity.",

                "plants": """SCANDINAVIAN 2026 PLANTS (SINGLE SPECIMEN - framed wildness):
□ Single architectural branch in hand-blown glass vase
□ NOT a full tree - single specimen only
□ Dried grasses or wild branches acceptable
□ "Framed wildness" - not cultivated
□ The negative space IS the statement""",

                "what_makes_it_different": """⚠️ SCANDINAVIAN vs MID-CENTURY (CRITICAL DIFFERENCE):

SCANDINAVIAN 2026:
- LIGHT BLONDE wood (ash, birch, raw oak)
- MUTED earth-shadows (stone gray, sage, dried-bone)
- Soft, "cloud-like" curves
- COZY hygge textures (sheepskin, chunky knit)
- CANDLES essential
- Diffused Nordic light
- "Heavily Light" - substantial but ethereal

MID-CENTURY (different!):
- DARK WALNUT wood
- BOLD saturated colors (mustard, orange, avocado)
- Sharp geometric shapes
- Sleek, sculptural surfaces
- BRASS/SPUTNIK essential
- Warm amber light
- Grounded retro energy

vs OTHER STYLES:
vs AD: Nordic light (not golden hour), light wood (not warm oak),
       muted (not rich), hygge cozy (not editorial drama)

vs Modern: COZY textures (not minimal void), organic curves (not angular),
           warm hygge (not cool ethereal), natural (not synthetic)

vs Coastal: No blue, no woven rope, no dopamine brights,
            Nordic (not tropical), contained (not breezy)

vs Farmhouse: LIGHT palette (not muddy), blonde (not forged iron),
              ethereal (not grounded), minimal (not collected)

SCANDINAVIAN 2026 = BLONDE WOOD + EARTH-SHADOWS + SHEEPSKIN + CANDLES + HYGGE""",

                "structural_rules": """⚠️ CRITICAL - STRUCTURAL PRESERVATION:
NEVER alter walls, doorways, openings, or architectural features.
NEVER invent damage, cracks, or imperfections that don't exist.
If there is an opening to another room, it MUST remain.
Identify ALL openings before generating - preserve exactly."""
            },

            # ============================================================================
            # COASTAL 2026: "Hyper-Breezy Sensory Obsession"
            # DOPAMINE BRIGHTS + ROPE/WOVEN + HYPER-BREEZY + HERITAGE
            # ============================================================================
            "coastal": {
                "description": "2026 Vision: Post-Nautical Maximalism meets Dopamine Tides. A meticulously engineered 'state of mind' designed to trick the brain into smelling salt air. Curated Maximalism where more is more.",
                "palette": "Sun-washed neutrals, sand, driftwood + DOPAMINE BRIGHTS (electric cobalt, digital lavender, sun-drenched coral)",
                "colors": "Foundation: sun-washed neutrals, sand, driftwood, warm white. DOPAMINE BRIGHTS (required!): electric cobalt blue, digital lavender, sun-drenched coral. Textures: buttery matte, sun-baked clay, hand-painted canvas. Rope/woven essential.",

                "signature_lighting": """COASTAL 2026 LIGHTING - "AI-Simulated Coastal Light":
- Light that feels more like a MEMORY than a photo
- Bright, diffused, "hyper-breezy" quality (3500-4000K)
- Sun-drenched warmth but soft, not harsh
- Textures rendered so sharply you can FEEL them
- The light IS the lead actor in the space
- "Limitless glass partitions" making ocean/garden the wallpaper
- Gauzy linen drapery with "implied movement" - caught in permanent zephyr""",

                "signature_atmosphere": """COASTAL 2026 ATMOSPHERE - "The Hyper-Breezy State":
- Transparent boundaries - walls become glass
- "Digital Wind" - curtains and textiles suggest movement
- Spiritual, tranquil, hyper-realistic feel
- Nancy Meyers "Warm Minimalism" meets wellness
- Indoor-outdoor yoga deck energy
- "Expensive and effortless" lifestyle suggestion
- Curated Maximalism - "more is more" with intention""",

                "furniture": """COASTAL 2026 FURNITURE - "Post-Nautical Maximalism":

THE PHILOSOPHY: Rise of "Curated Maximalism" - more is more.
Bespoke, artisanal storytelling. Living artifacts, not mass-produced anchors.

BED:
□ Oversized slip-covered linen bed - SUBSTANTIAL presence
□ OR dramatic RATTAN/ROPE-wrapped frame (statement piece!)
□ Layered, luxurious bedding - white base with texture
□ "Sophisticated Solace" - feels inherited, not bought
□ Heritage quality, antique timber accents

NIGHTSTANDS:
□ ROPE-WRAPPED sculptural pieces (artisanal!)
□ OR reclaimed boat timber surfaces
□ OR woven seagrass with organic form
□ NOT generic white - must tell a story

SEATING:
□ Oversized slip-covered linen armchair
□ OR sculptural woven ROPE chair (signature!)
□ Sink-in comfort, "Coastal Grandma" luxury

RUG:
□ LAYERED jute rugs - multiple textures
□ OR hand-painted floor canvas (sun-kissed texture)
□ OR vintage, heritage piece with patina
□ Buttery matte surfaces - sun-baked clay feel""",

                "decor": """COASTAL 2026 SIGNATURES (must include 2-3):
□ ROPE-WRAPPED sculptural light fixture (art installation!)
□ Boat-shaped or nautical-form bookcase/furniture
□ Giant organic glass lighting
□ Gauzy drapery with "implied movement" - caught in breeze
□ Rough grain of driftwood - textured and tactile
□ Living greenery wall or spa-inspired feature
□ DOPAMINE BRIGHT accent - electric cobalt, coral, or lavender (ESSENTIAL!)
□ Heritage/antique timber piece

ART:
□ Large-scale ocean photography or coastal abstract
□ In weathered driftwood or rope-detailed frame

MUST INCLUDE: At least one DOPAMINE BRIGHT accent
FORBIDDEN: Generic "blue and white," mass-produced wicker, basic seashell decor""",

                "rugs": "LAYERED jute rugs - multiple textures. OR hand-painted floor canvas. OR vintage heritage piece. Sun-baked clay feel.",

                "lighting": """COASTAL 2026 LIGHTING FIXTURES:
- ROPE-WRAPPED sculptural pendant (signature!)
- Giant organic glass fixtures
- Woven/seagrass elements
- NO brass, minimal metal""",

                "overall": "DOPAMINE BRIGHTS + ROPE/WOVEN + HYPER-BREEZY + HERITAGE. Post-Nautical Maximalism.",

                "plants": """COASTAL 2026 PLANTS (LESS than other styles - keep it airy):
□ LESS plants than other styles - the VIEW and LIGHT are the focus
□ If used: palm, tropical, or dramatic architectural
□ In woven or terracotta vessel
□ Keep it airy and breezy - don't crowd the light""",

                "what_makes_it_different": """COASTAL 2026 vs OTHER STYLES:

vs AD: DOPAMINE BRIGHTS (not earth tones), rope/woven textures,
       hyper-breezy light (not golden hour), spiritual/tranquil

vs Modern: WARM and welcoming (not cool/ethereal), natural materials,
           maximalist (not minimal void), soft (not angular)

vs Farmhouse: Coastal is AIRY (farmhouse is grounded/earthy),
              brighter palette, ocean heritage (not barn heritage)

vs Scandinavian: DOPAMINE color pops (not muted pastels),
                 rope/woven maximalism (not minimal), warm (not cool)

vs Mid-Century: No retro colors, no walnut, no atomic shapes,
                organic/natural (not geometric), breezy (not grounded)

COASTAL 2026 = DOPAMINE BRIGHTS + ROPE/WOVEN + HYPER-BREEZY + HERITAGE""",

                "structural_rules": """⚠️ CRITICAL - STRUCTURAL PRESERVATION:
NEVER alter walls, doorways, openings, or architectural features.
NEVER invent damage, cracks, or imperfections that don't exist.
If there is an opening to another room, it MUST remain.
Identify ALL openings before generating - preserve exactly."""
            },

            # ============================================================================
            # FARMHOUSE 2026: "Neo-Farmhouse - Storied Sanctuary"
            # MUDDY PALETTE + BLACK IRON + PLASTERED/LIMEWASH + HACIENDA
            # ============================================================================
            "farmhouse": {
                "description": "2026 Vision: Neo-Farmhouse - less movie set, more storied, soulful sanctuary. Earthy Emotionalism meets Artisanal Grit. Hacienda fusion with cast iron and plastered walls.",
                "palette": "MUDDY richness - mushroom, putty, weathered taupe + color-drenched accents (olive, oxblood, deep sable, terracotta)",
                "colors": "Foundation: mushroom (gray-brown), putty (warm gray-beige), weathered taupe. Accents: olive green, oxblood (deep red-brown), deep sable brown, terracotta. Metals: BLACK IRON (forged), bronze, antique brass. Textures: limewash, plaster, cast iron, linen, wide-plank oak.",

                "signature_lighting": """FARMHOUSE 2026 LIGHTING - "Warm and Wise Glow":
- Sun-drenched, golden, EARTHY quality (2400-2800K)
- Cozy lantern-light atmosphere
- Warm shadows creating depth and security
- "Cocoon-like feeling of security"
- Tactile immersion - you can SENSE the softness of linen
- Candlelit evening ambiance welcome
- Deep, rich, grounding quality""",

                "signature_atmosphere": """FARMHOUSE 2026 ATMOSPHERE - "Grounded Earthy Emotionalism":
- "Muddy" palette instead of stark white - NO white shiplap!
- Color drenched walls - olive, oxblood, deep sable
- "A life in progress rather than a house for sale"
- Intentional imperfection - hand-touched finishes
- Organic chaos over rigid symmetry
- "Collected, not just curated" aesthetic
- Emotional well-being zones (reading corners, breakfast nooks)
- Hacienda fusion influence - rustic meets terracotta""",

                "furniture": """FARMHOUSE 2026 FURNITURE - "Artisanal Grit + Modern Heritage":

THE PHILOSOPHY: "Playful Maximalism" for the countryside.
Clean lines AND inherited soul. Post-Industrial Relics.
Sculptural Utility. Farmhouse-Hacienda fusion.

BED:
□ BLACK IRON bed frame - heavy, forged appearance (SIGNATURE!)
□ OR reclaimed timber headboard with visible character/history
□ OR limewashed plaster effect headboard
□ Substantial, GROUNDED presence - not delicate
□ "Quietly luxurious" focal point

NIGHTSTANDS:
□ Heavy cast iron hardware accents
□ Reclaimed wood with visible history and patina
□ OR terracotta/stucco inspired form (Hacienda twist)
□ Mismatched pair adds authenticity
□ Bronze finishes that feel "forged not manufactured"

SEATING:
□ Oversized stone or terracotta form
□ OR vintage leather club chair with patina
□ OR sculptural wooden stool/bench
□ "Sculptural Utility" - everyday objects as art

RUG:
□ Vintage with FADED pattern - mushroom, terracotta, olive tones
□ OR terracotta tiles visible (Hacienda influence)
□ OR wide-plank oak floors with coarse grain texture
□ Worn, storied appearance - NOT new""",

                "decor": """FARMHOUSE 2026 SIGNATURES (must include 2-3):
□ BLACK IRON element (bed frame, hardware, hooks, lamp) - ESSENTIAL!
□ Cast iron or bronze "forged" hardware detail
□ Limewashed or plastered wall texture effect
□ Railroad spike hooks or antler sculptural piece
□ Oversized apron-front stone sink (if kitchen visible)
□ Giant lantern pendant
□ Terracotta floor or accent (Hacienda fusion)
□ Indoor olive tree or trailing plants (organic chaos)
□ Vintage quilt or heritage textile

ART:
□ Antique or vintage piece with patina
□ OR landscape in weathered frame
□ OR architectural salvage as wall art

MUST INCLUDE: Black iron AND muddy/color-drenched element
FORBIDDEN: Stark white shiplap, gray/white palette, anything "new" or "trendy", chrome""",

                "rugs": "Vintage with FADED pattern in mushroom/terracotta/olive. OR terracotta tiles visible. Worn, storied - NOT new.",

                "lighting": """FARMHOUSE 2026 LIGHTING FIXTURES:
- Giant lantern pendant (SIGNATURE!)
- BLACK IRON fixtures
- Forged bronze or cast iron details
- Edison bulb warmth
- NO chrome, NO modern minimal""",

                "overall": "MUDDY PALETTE + BLACK IRON + PLASTERED/LIMEWASH + HACIENDA. Storied sanctuary.",

                "plants": """FARMHOUSE 2026 PLANTS (organic chaos):
□ Indoor olive tree (storied, ancient feel)
□ Trailing plants creating organic chaos
□ Herbs in terracotta (functional beauty)
□ NOT perfectly placed - "collected" feel
□ Galvanized or terracotta containers""",

                "damage_prevention": """⚠️ CRITICAL FOR FARMHOUSE:
- Show CHARACTER in FURNITURE (distressed, worn) - STYLE
- Do NOT invent damage on WALLS (cracks, holes, stains) - DEFECT
- Distressed FURNITURE is intentional design
- Damaged WALLS is misrepresentation
- Original wall condition must be preserved EXACTLY""",

                "what_makes_it_different": """FARMHOUSE 2026 vs OTHER STYLES:

vs AD: GROUNDED and heavy (not floating/airy), cast iron (not brass),
       muddy palette (not warm amber), rustic heritage (not editorial)

vs Modern: WARM soul (not cool void), imperfect (not pristine),
           collected chaos (not minimal), forged (not manufactured)

vs Coastal: EARTHY/grounded (not breezy/light), no blue or dopamine brights,
            landlocked heritage (not ocean heritage), heavier textures

vs Scandinavian: DARK and muddy (not bright/white), heavy (not light),
                 cast iron (not blonde wood), grounded (not ethereal)

vs Mid-Century: No retro shapes, no bold primary colors,
                rustic organic (not geometric), heritage (not vintage)

FARMHOUSE 2026 = MUDDY PALETTE + BLACK IRON + PLASTERED/LIMEWASH + HACIENDA""",

                "structural_rules": """⚠️ CRITICAL - STRUCTURAL PRESERVATION:
NEVER alter walls, doorways, openings, or architectural features.
NEVER invent damage, cracks, or imperfections that don't exist.
Distressed FURNITURE is style. Damaged WALLS is fraud.
If there is an opening to another room, it MUST remain.
Identify ALL openings before generating - preserve exactly."""
            },

            # ============================================================================
            # MID-CENTURY MODERN 2026: "Retro-Futurism - Atomic Optimism"
            # DARK WALNUT + BOLD RETRO COLORS + TAPERED LEGS + BRASS/SPUTNIK
            # ============================================================================
            "midcentury": {
                "description": "2026 Vision: Retro-Futurism - atomic-age optimism meets 21st-century luxury. Organic geometry meets invisible technology. High-energy emotional architecture.",
                "palette": "WARM WALNUT/cherry/burl + BOLD saturated colors (mustard, avocado, burnt orange, teal) + BRASS gold",
                "colors": "Wood: WARM WALNUT (not blonde!), CHERRY, burl wood accents. Saturated Earth: mustard yellow, burnt orange, avocado green, deep teal. Digital Pops: digital lavender, cobalt blue, paprika red. Metals: BRASS and GOLD with hyper-realistic reflections.",

                "signature_lighting": """MCM 2026 LIGHTING - "Aerodynamic Atmosphere":
- WARM, saturated, rich amber quality (2700-3000K)
- Sun-drenched transparency - light flooding through large windows
- "Indoor-outdoor connection" emphasized
- Sheer, invisible drapery maximizing light
- Hyper-realistic reflections on brass and metals
- Sputnik chandeliers like "exploding stars frozen in time"
- Lustrous, mood-lifting energy""",

                "signature_atmosphere": """MCM 2026 ATMOSPHERE - "Gravity-Defying Energy":
- Clean, uncluttered lines suggesting forward motion
- "Weightless" furniture with dramatic leg reveals
- Wide-plank floors visible - sense of infinite space
- Geometric tension - circles, triangles, kidney shapes vibrating
- Bold graphic compositions
- "Emotional architecture" - space evokes feeling
- Sleek, atmospheric, intentional""",

                "furniture": """MCM 2026 FURNITURE - "Sculptural Functionalism":

THE PHILOSOPHY: Furniture as functional art. Iconic anchors that feel "touchable."
Gravity-defying silhouettes on hairpin legs. Curated Minimalism.
Every piece is LEGENDARY.

⚠️ CRITICAL: Use DARK WALNUT with TAPERED LEGS
NOT light blonde wood - that's Scandinavian!

BED:
□ DARK WALNUT or CHERRY platform with TAPERED HAIRPIN LEGS (SIGNATURE!)
□ Dramatic leg reveal - frame lifted high showing floor beneath
□ Headboard: walnut panel, vertical slats, or burl wood
□ "Cocooning sleigh form" optional for drama
□ Lustrous wood grain rendered with tactile clarity

NIGHTSTANDS:
□ ICONIC MCM design - TAPERED LEGS essential!
□ Walnut or cherry with brass hardware
□ OR floating/cantilevered design
□ Burl wood accent pieces
□ "Brutalist heft" meets delicate legs

SEATING:
□ EAMES LOUNGE CHAIR AND OTTOMAN (the icon!)
□ OR molded plywood shell chair
□ OR Womb chair style in BOLD color
□ Tactile leather or rich upholstery
□ ONE legendary silhouette per room

RUG:
□ Bold geometric pattern in period colors
□ OR SHAG texture in mustard, avocado, or cream
□ OR abstract pattern with kidney/boomerang shapes
□ Statement underfoot - conversation starter""",

                "decor": """MCM 2026 SIGNATURES (must include 2-3):
□ TAPERED HAIRPIN LEGS on major furniture (ESSENTIAL!)
□ BOLD COLOR accent - mustard, orange, or avocado (ESSENTIAL!)
□ SPUTNIK chandelier or starburst lighting (SIGNATURE!)
□ EAMES or iconic silhouette piece
□ Burl wood or walnut surface
□ BRASS accent with lustrous reflection
□ Geometric art - starburst, kidney shape, atomic motif
□ SHAG texture element
□ Arc floor lamp

ART:
□ Abstract expressionist style
□ OR bold graphic print in period colors
□ OR vintage travel poster
□ Teak or walnut frame

MUST INCLUDE: Tapered legs + Bold color + Brass element
FORBIDDEN: Blonde wood (Scandinavian!), soft pastels, black iron, chunky knits""",

                "rugs": "Bold geometric pattern in period colors. OR SHAG in mustard, avocado, cream. Statement piece - conversation starter.",

                "lighting": """MCM 2026 LIGHTING FIXTURES:
- SPUTNIK chandelier (SIGNATURE!) - exploding star frozen in time
- Arc floor lamp in BRASS (Arco style)
- Nelson bubble lamp
- Sculptural ceramic table lamps
- BRASS essential everywhere""",

                "overall": "DARK WALNUT + BOLD RETRO COLORS + TAPERED LEGS + BRASS/SPUTNIK. Atomic optimism.",

                "plants": """MCM 2026 PLANTS (PERIOD PLANTERS essential):
□ Oversized indoor tree - sculptural, dramatic
□ Snake plant in BULLET PLANTER (period ceramic!) - SIGNATURE!
□ Or architectural plant mimicking vertical lines
□ Ceramic planter in PERIOD COLOR (mustard, olive, orange)
□ NOT woven baskets - that's Scandi/Coastal""",

                "what_makes_it_different": """⚠️ MID-CENTURY vs SCANDINAVIAN (CRITICAL DIFFERENCE):

MID-CENTURY 2026:
- DARK WALNUT with tapered hairpin legs
- BOLD SATURATED colors (mustard, orange, avocado)
- Sharp geometric shapes, atomic motifs
- Sleek, sculptural surfaces
- BRASS/SPUTNIK essential
- Warm amber light, saturated
- Grounded retro energy, gravity-defying legs

SCANDINAVIAN (different!):
- LIGHT BLONDE wood (ash, birch, raw oak)
- MUTED earth-shadows (stone gray, sage)
- Soft, "cloud-like" curves
- COZY textures (sheepskin, chunky knit)
- CANDLES essential
- Nordic diffused light
- Ethereal hygge

vs OTHER STYLES:
vs AD: RETRO bold colors (not earth tones), walnut/cherry (not oak),
       atomic shapes (not organic natural), Sputnik (not brass pendants)

vs Modern: WARM and saturated (not cool/ethereal),
           RETRO nostalgia (not future void), bold colors (not monochrome)

vs Coastal: No woven textures, no blue/dopamine, no natural fiber,
            geometric (not organic), grounded (not breezy)

vs Farmhouse: SLEEK (not rustic), manufactured icon (not forged artifact),
              bold primary (not muddy earth), levity (not weight)

MCM 2026 = DARK WALNUT + BOLD RETRO COLORS + TAPERED LEGS + BRASS/SPUTNIK""",

                "structural_rules": """⚠️ CRITICAL - STRUCTURAL PRESERVATION:
NEVER alter walls, doorways, openings, or architectural features.
NEVER invent damage, cracks, or imperfections that don't exist.
If there is an opening to another room, it MUST remain.
Identify ALL openings before generating - preserve exactly."""
            },
            "architecture_digest": {
                "description": "Editorial-quality transformation inspired by Adam Potts. NOT just warmer staging - requires dramatic lighting transformation + curated designer furniture + magazine-cover quality.",
                "palette": "warm woods (cedar, walnut, oak, teak), CREAM/IVORY whites only (never stark/cool white), brass/bronze ONLY (no chrome/silver), muted earth accents (terracotta, sage, rust, charcoal). NEVER cool/blue tones anywhere.",
                "colors": "warm woods (cedar, walnut, white oak, teak in golden-brown tones), warm whites (cream, ivory, warm plaster), natural stone (travertine, limestone, warm concrete), metals in brass/bronze only, accents in terracotta, sage green, rust, charcoal",
                "furniture": """LIVING: CURVED WHITE BOUCLÉ sofa (signature silhouette), organic coffee table (woven/wood slab/sculptural glass), cream swivel chairs OR cognac leather lounge chairs.
DINING: Solid light oak table with clean lines, Hans Wegner wishbone chairs in natural ash (6 chairs), large natural jute rug, ONE large abstract art (36x48 min, earth tones).
KITCHEN: Olive wood cutting board angled near stove, 6-8 lemons in shallow wooden bowl, dark ceramic vase with pink PROTEA - MINIMAL accessories only.
BEDROOM: Low platform bed with cream linen upholstered headboard, layered white/cream bedding with textured throw, sculptural/floating natural wood nightstands, ceramic lamps.""",
                "decor": """SIGNATURE ELEMENTS (include 1-2 per room, VARY across property - NOT every room needs an olive tree):
- Pink/coral PROTEA in sculptural dark ceramic vase
- Sculptural ceramic objects
- Art books (LIVING ROOM ONLY)
- Woven basket or natural texture element

PLANT VARIETY (do NOT use olive tree in every room):
- Living room: Olive tree OR fiddle leaf fig (choose one, not both)
- Dining room: Olive BRANCHES in vase (not full tree unless room is very large)
- Bedroom: Small plant on nightstand OR olive tree (not both) - tree only if room is large
- Kitchen: Small herb plant OR protea (small, not tree)
- Bathroom: Small architectural plant only (no large trees)
- Hallway: Small plant in terracotta OR nothing""",
                "rugs": "Large natural jute with subtle texture for dining/living, muted vintage Persian as alternative, natural fiber in warm browns and creams",
                "lighting": """DRAMATIC LIGHTING (CRITICAL - this separates 'nice staging' from 'editorial'):
1. Golden hour quality (1 hour before sunset feeling)
2. Visible warm light rays streaming through windows
3. Rich dimensional shadows (warm-toned brown/amber, NOT flat/gray)
4. Interior glow effect (space feels lit from within)
5. Color temp 2700K-3000K throughout - NO cool/blue tones
6. Windows should glow with soft golden light""",
                "overall": "Editorial, dramatic, warm, curated - magazine-cover quality. NOT just 'warmer' - truly transformed with dramatic lighting + designer furniture + cohesive styling.",
                "photo_treatment": """THREE TRANSFORMATION LAYERS:
1. LIGHTING: Golden hour rays, dimensional shadows, interior glow, 2700K warmth
2. COLOR GRADING: Push entire image warm/golden, whites=cream, shadows=warm brown/amber (never gray), wood=rich honey/amber
3. ATMOSPHERE: Magazine-cover drama, every image looks like 1 hour before sunset""",
                "exterior_requirements": """CRITICAL - Exteriors need MOST transformation:
- SKY: Golden hour gradient (blue at top → gold → pink/peach at horizon)
- WINDOWS: MUST show warm interior glow (amber light visible through every window)
- TREES: Catching golden side-light, warm highlights
- SIGNATURE: Olive tree in aged terracotta pot near entry
- OVERALL: 'Dwell magazine cover at sunset' quality""",

                "structural_rules": """⚠️ CRITICAL - STRUCTURAL PRESERVATION FOR AD STYLE:
NEVER alter walls, doorways, openings, or architectural features.
NEVER invent damage, cracks, holes, or imperfections that don't exist.

⚠️ ITEM REMOVAL - DAMAGE PREVENTION:
When removing items like TVs, wall art, or furniture:
- The wall/surface behind MUST appear CLEAN and UNDAMAGED
- Do NOT add mounting holes, screw marks, or discoloration
- Do NOT invent paint chips, cracks, or marks where items were
- The wall should look like it was always empty and pristine
- If removing a TV, that wall section becomes a clean, normal wall

If there is an opening to another room, it MUST remain.
Identify ALL openings before generating - preserve exactly."""
            }
        }

        style_guide = style_guides.get(style_preference, style_guides["modern"])

        # Build client special instructions section if comments provided
        comments_section = ""
        if comments and comments.strip():
            comments_section = f"""
=== CLIENT SPECIAL INSTRUCTIONS ===
The client has provided these specific instructions:
"{comments}"

Consider these instructions when staging. If the client mentions specific rooms or requests
(e.g., "stage small bedroom as nursery", "kitchen needs bar stools"), apply that guidance
to the relevant images. Incorporate their preferences while maintaining the selected style.
"""

        return f"""You are a professional virtual staging designer creating beautiful, realistic staging prompts for Gemini's image editor.

{STRUCTURAL_PRESERVATION_RULES}

=== CRITICAL: VIRTUAL STAGING ONLY - DO NOT REGENERATE THE ROOM ===

Treat this as a VIRTUAL STAGING task on top of the existing photograph, NOT a full room regeneration.
Use the original image as the strict base layer:
- Keep ALL walls, doors, windows, trim, flooring, ceiling height, and built-in elements EXACTLY where they are
- Do NOT move, resize, or remove ANY architectural features or built-ins
- Do NOT change paint colors, flooring materials, or window sizes
- Only adjust lighting/photo quality - NEVER alter the physical space itself
- The "bones" of the room must remain 100% unchanged from the original photo

=== CORE PRINCIPLES ===

1. PRESERVE STRUCTURE & ARCHITECTURE (NON-NEGOTIABLE)
   Keep walls, windows, doors, flooring, ceiling, and built-in features exactly as they are.
   Do not move walls, add windows, change flooring materials, or alter room dimensions.
   The room's shell must be identical to the original photo.

2. FURNITURE SCALE AND ROOM HONESTY (CRITICAL)
   When adding furniture or decor, keep all pieces at REALISTIC PHYSICAL SCALE for the room:
   - Beds, sofas, tables, and chairs must be sized so they would comfortably fit in the actual floor area visible
   - Do NOT use oversized or undersized furniture to make the room appear larger, wider, or deeper than it is
   - Do NOT widen the field of view or push the back wall farther away
   - The perceived dimensions of the room MUST match the original photo exactly
   - A small room should look like a nicely staged small room, not a fake large room

3. DO NOT USE STAGING TO HIDE DEFECTS (MLS COMPLIANCE - CRITICAL)
   NEVER use furniture, rugs, curtains, or decor to hide visible defects in the property:
   - Do NOT cover holes, cracks, stains, damaged trim, damaged flooring, or other issues
   - Do NOT strategically place couches, beds, rugs, plants, or artwork to obscure damage
   - If damage is visible in the original image, it MUST remain visible and unobstructed in the staged version
   - Do NOT fix, repaint, patch, blur, or cover any damage to walls, ceilings, floors, doors, windows, trim, or fixtures
   - Do NOT invent or add damage that doesn't exist in the original
   - This is LEGALLY REQUIRED for MLS honesty - violations can result in lawsuits

4. CAMERA ANGLE AND LEVELING (CRITICAL)
   You may adjust the camera ONLY in ways that make the shot look professionally photographed while staying honest:

   ALWAYS DO:
   - Level the image so horizontals are straight and vertical lines (walls, doors, windows, posts, cabinets) are truly vertical

   MAY DO:
   - Adjust the apparent camera height slightly (a bit higher or lower) if it improves composition
   - Adjust vertical angle (pitch) slightly for better framing

   MUST NOT DO:
   - Move the camera horizontally to a different wall or corner
   - Swing the view left or right (yaw) to reveal new walls, doors, windows, or areas not visible in the original
   - Widen the field of view to make the room appear larger
   - Pull the back wall farther away to fake depth
   - Invent or hallucinate any part of the room that wasn't photographed

5. STYLE CONSISTENCY FOR THIS PROPERTY
   Style: {style_preference.upper().replace('_', ' ')}
   {style_guide['description']}
   Colors: {style_guide['colors']}

   {style_guide.get('signature_lighting', '')}

   {style_guide.get('what_makes_it_different', '')}

   Apply this style consistently:
   - All staged furniture and decor must match this style's palette and aesthetic
   - Colors, materials, and forms should be cohesive throughout
   - Match the style to a believable price point for this property (attractive but not ultra-luxury unless the base photo supports it)
   - Do NOT mix multiple unrelated design styles in the same room

{universal_rules}

6. PROFESSIONAL PHOTO ENHANCEMENT
   Apply professional-level photographic improvements:
   - Correct exposure and increase contrast slightly for a crisp, well-defined look
   - Correct white balance to neutralize color casts
   - Reduce noise and haze so the image looks sharp like a DSLR photo
   - Enhance natural lighting - make rooms feel bright and welcoming
   - Result should be photorealistic - absolutely NO "AI art" vibe

7. REALISTIC FURNITURE PLACEMENT
   - Furniture must be properly scaled to the room dimensions (see principle #2)
   - Maintain realistic proportions and perspective
   - Leave appropriate walkways and spacing - don't overfill the room
   - Furniture should appear grounded with proper shadows and reflections
   - Circulation paths must be believable

8. TASTEFUL DECOR
   - Add appropriate decorative items: plants, art, rugs, throw pillows, lamps
   - Keep decor cohesive with the style preference
   - Don't over-stage - less is often more
   - Plants should be realistic and appropriate for indoor spaces

9. RUG ANCHORING (CRITICAL FOR COHESION)
   - Use a large area rug that extends under all seating, with at least the front legs of all sofas and chairs on the rug
   - The rug should make the furniture read as one cohesive conversation area, not isolated pieces
   - Choose a rug with subtle, sophisticated pattern or texture that complements wood tones and doesn't fight the architecture
   - Rug should be proportional to the seating area - err on the side of larger, not smaller

10. FOCAL POINTS (ONE PER ROOM)
   - Every room needs ONE clear focal point that photographs incredibly well
   - Living room: large art above fireplace + beautifully styled coffee table
   - Dining room: floral centerpiece or sculptural object + chandelier/pendant
   - Bedroom: headboard wall with art + styled nightstands
   - Keep the rest of the decor quieter so the focal point stands out
   - The focal point should draw the eye immediately when viewing the photo

11. SHELF & BOOKCASE STYLING (AVOID CLONE LOOK)
   - Style shelves with a CURATED but VARIED mix - not identical repeating objects
   - Books: different heights and spine colors within a soft, muted palette
   - Mix in sculptural objects, small plants, 1-2 framed photos or art pieces
   - Leave some negative space on shelves - don't overstuff
   - NEVER repeat the same object, vase, or book stack multiple times across shelves

12. MICRO-REALISM (SUBTLE LIFE)
   - Allow very subtle natural variation so the room feels lived-in rather than rendered
   - A pillow that isn't perfectly rigid, a throw with a slight realistic fold
   - Bed linens with natural drape, not frozen stiff
   - Keep it subtle - the room should still look professionally staged, just not sterile

=== FULL VIRTUAL STAGING – STRIP AND REFURNISH ===

For virtual staging, REMOVE ALL existing furniture and decor and REFURNISH the room from scratch in the selected style.

Treat the room as an EMPTY ARCHITECTURAL SHELL: keep only the walls, doors, windows, trim, flooring, built-ins, and any visible damage.

You MUST REMOVE: beds, sofas, chairs, tables, dressers, rugs, lamps, artwork, and personal decor – even if the room is currently furnished.

Then ADD a complete, realistic furniture layout appropriate for the room type and the selected style.

CRITICAL: Do NOT hide or cover holes, cracks, stains, or other damage with furniture, rugs, or decor — defects must remain visible.

=== OCCUPANCY DESCRIBES INPUT, NOT BEHAVIOR ===

You will detect whether the room is OCCUPIED (has existing furniture) or VACANT (empty/nearly empty). This describes the CURRENT STATE of the input image only.

REGARDLESS of occupancy status, the behavior is ALWAYS THE SAME:
1. STRIP THE ROOM: Remove all existing furniture and decor
2. RESTAGE THE ROOM: Add a complete furniture layout in the {style_preference.replace('_', ' ')} style

For this virtual staging pipeline, you may remove all existing furniture and decor whether the room is currently vacant or occupied. Occupancy only describes the current state – the goal is ALWAYS to show a fully virtually staged version.

=== STAGING REQUIREMENTS (FOR ALL ROOMS) ===

After stripping, add a complete {style_preference.replace('_', ' ')} furniture set:
- Main furniture pieces appropriate for room type (bed, sofa, dining table, desk, etc.)
- Accent pieces (nightstands, coffee table, side tables, accent chairs)
- Area rug appropriately sized and positioned
- Lighting (floor lamps, table lamps, enhanced overhead light)
- Plants and greenery
- Wall art
- Throw pillows, blankets, and soft accessories

Keep circulation paths believable - do NOT overfill the room.
All furniture must be properly scaled to fit the actual room dimensions.
Do NOT alter the existing finishes (walls, floors, windows, built-ins).
The goal is "move-in ready showcase" that helps buyers visualize living there.

=== STYLE GUIDE: {style_preference.upper().replace('_', ' ')} ===
{style_guide['description']}

Furniture: {style_guide['furniture']}
Decor: {style_guide['decor']}
Rugs: {style_guide['rugs']}
Lighting: {style_guide['lighting']}

=== 4-PARAGRAPH PROMPT SKELETON ===

Your staging_prompt MUST follow this exact structure. ALL rooms get fully staged (strip + refurnish):

PARAGRAPH 1 - Context & Goal:
"Using this uploaded [room type] photo, create a fully virtually staged real-estate listing image in a {style_preference.replace('_', ' ')} style. Keep the same architectural shell: walls, windows, doors, trim, ceiling height, flooring, and any built-in elements must stay exactly where they are. Do not move walls, add or remove windows, change flooring, or alter the size or position of doors or openings. Do not fix, repaint, patch, or cover any visible cracks, stains, holes, or other damage; these must remain visible."

PARAGRAPH 2 - Strip the Room:
"First, STRIP THE ROOM: remove all existing furniture and decor, including [list specific items you see - sofas, beds, chairs, tables, shelves, TV, rugs, freestanding lamps, personal items], so the room appears unfurnished while still showing the true architecture and any visible defects. Do not hide or cover damaged areas with furniture or rugs."

PARAGRAPH 3 - Refurnish in Selected Style:
"Then, RE-FURNISH the room completely in a {style_preference.replace('_', ' ')} style: [Add detailed, style-specific furniture descriptions with colors, materials, and placement appropriate for this room type. Be specific about: main furniture pieces, accent pieces, area rug description and placement, plants, wall art, and lighting. All items must match the {style_preference.replace('_', ' ')} palette and aesthetic.] All furniture and decor must be realistically scaled to the room size – do not use oversized or undersized furniture to make the room appear larger, wider, or deeper than it is. A small room should look like a nicely staged small room. CRITICAL: Do NOT place any furniture, rugs, or decor to cover or hide any visible damage, stains, cracks, or wear - all defects must remain fully visible and unobstructed."

PARAGRAPH 4 - Camera & Photo Enhancement:
"LEVEL THE PHOTO: straighten the image so horizontals are level and vertical lines (walls, windows, door frames) are truly vertical. You may adjust camera height and vertical angle slightly to improve composition, but do not move the camera horizontally or swing it left/right to reveal new walls or areas that are not visible in the original photo. Do not widen the field of view or make the room appear larger or deeper than it is.

Apply professional-level photo enhancement: correct exposure, increase contrast slightly, and improve overall clarity so details in the flooring, trim, and furniture are crisp and well defined. Correct white balance so whites and light surfaces look clean and true-to-life without becoming unnaturally bright. Reduce noise and haze so the image looks sharp and high-quality, similar to a DSLR real-estate photograph. The final result should be a photorealistic, fully staged {style_preference.replace('_', ' ')} [room type] that still honestly reflects the true architecture and condition of the property."

=== ROOM-TYPE STAGING PATTERNS (STRIP + REFURNISH) ===

For ALL room types: First strip existing furniture, then add style-appropriate staging.

KITCHEN:
- Preserve: cabinets, countertops, appliances, sink, flooring, windows
- Strip: Remove any existing items on counters, floor clutter, personal items
- Stage with: Bar stools (if island/peninsula), fruit bowl, small plant, coordinating hand towels, subtle countertop accessories
- Style touches: {style_guide['decor']}

BATHROOM:
- Preserve: vanity, sink, toilet, tub/shower, tile, mirrors, flooring
- Keep accessibility equipment (grab bars, shower chairs) if present
- Strip: Remove personal toiletries, clutter, mismatched towels
- Stage with: Neatly rolled towels in coordinating colors, small plant (like bamboo or pothos), tasteful soap dispenser, small tray with minimal items
- Style touches: Clean, spa-like aesthetic that complements the home's style

BEDROOM:
- Preserve: closets, windows, flooring, any built-in features
- Strip: Remove existing bed, nightstands, dressers, all furniture and decor
- Stage with: Bed with headboard, nightstands, lamps, dresser (if space allows), area rug, art above bed, plants
- Bed styling: Crisp white or neutral bedding with layered pillows and a folded throw at foot
- Style touches: {style_guide['furniture']} for bed and nightstands, {style_guide['decor']}

LIVING ROOM:
- Preserve: fireplace (if present), built-in shelving, windows, flooring
- Strip: Remove all existing sofas, chairs, tables, rugs, decor, personal items
- Stage with: Sofa, coffee table, accent chairs, area rug, floor lamp, table lamps, plants, wall art, throw pillows
- Layout: Create a conversation area, typically centered on fireplace or focal wall
- Style touches: {style_guide['furniture']}, {style_guide['rugs']}, {style_guide['lighting']}

DINING ROOM:
- Preserve: windows, flooring, any built-in features like wainscoting
- Strip: Remove existing dining furniture, decor, clutter
- Stage with: Dining table with chairs, area rug under table, pendant light or chandelier effect, simple centerpiece, sideboard if space allows
- Table styling: Simple place settings or just a centerpiece (not fully set)
- Style touches: {style_guide['furniture']}, {style_guide['decor']}

OFFICE/DEN:
- Preserve: windows, flooring, built-in shelving
- Strip: Remove existing desk, chair, and personal items
- Stage with: Desk, desk chair, bookshelf or shelving, task lamp, small plant, wall art
- Style touches: Professional but warm, styled bookshelves, {style_guide['decor']}

HALLWAY:
- Preserve: doors, flooring, trim, ceiling
- Strip: Remove clutter, personal items, existing decor
- Stage with: Console table (if space), mirror, small plant, runner rug, wall art
- Keep it simple - hallways should feel open and inviting

EXTERIOR / PORCH:
- Preserve: architecture, siding, windows, roof, landscaping, driveway
- Strip: Remove clutter, personal items from porch/deck
- Stage porch/deck with: Outdoor seating, potted plants, welcome mat, outdoor rug (if covered porch)
- Do NOT change landscaping, grass color, or add plants to yard

=== ARCHITECTURE DIGEST STYLE - SPECIAL ROOM GUIDANCE ===

IF the selected style is "architecture_digest", use these SPECIFIC staging patterns instead of the generic ones above.
This style requires THREE transformation layers: DRAMATIC LIGHTING + DESIGNER STAGING + WARM COLOR GRADING.

=============================================================================
⚠️ CRITICAL: STRUCTURAL PRESERVATION (HIGHEST PRIORITY) ⚠️
=============================================================================

YOU MUST PRESERVE ALL ARCHITECTURAL FEATURES EXACTLY AS THEY APPEAR IN THE
ORIGINAL IMAGE. This is a legal requirement for real estate photography.

NEVER ALTER, REMOVE, OR INVENT:
- Doorways and door openings (even if no door is visible)
- Archways and passages between rooms
- Windows and window placements
- Walls and wall positions
- Room openings to adjacent spaces
- Columns, pillars, or structural elements
- Built-in shelving, niches, or alcoves
- Ceiling heights or configurations
- Floor levels or transitions
- Fireplaces or fireplace openings
- Stairways or stair openings

SPECIFICALLY:
- If there is an opening to another room, it MUST remain an opening
- If you can see into a kitchen, hallway, or other space, that view MUST remain
- Do NOT fill in doorways with walls
- Do NOT extend walls where there are openings
- Do NOT remove or alter any architectural pass-throughs
- Do NOT add walls or structural elements that don't exist

WHY THIS MATTERS:
Altering the structure of a room misrepresents the property to potential buyers.
This violates real estate disclosure laws and MLS compliance requirements.
Our customers could face lawsuits if structural features are misrepresented.

BEFORE GENERATING: Carefully identify ALL openings, doorways, and passages
to adjacent spaces in the original image. These MUST appear in your output.

=============================================================================

*** ADDITIONAL: ARCHITECTURAL PRESERVATION (APPLIES TO ALL ARCHITECTURE DIGEST ROOMS) ***
The dramatic lighting transformation is PHOTO ENHANCEMENT ONLY - you are NOT changing the physical structure:
- Do NOT move, add, remove, or resize any windows
- Do NOT fill in, move, or alter any doorways or door openings
- Do NOT move, add, or remove any walls or change wall positions
- Do NOT change flooring materials, patterns, or boundaries
- Do NOT alter ceiling height, beams, or ceiling features
- Do NOT change room dimensions or make the space appear larger/smaller
- Do NOT add architectural features that don't exist (arches, columns, skylights, etc.)
- Do NOT remove or alter any built-in features (cabinets, fireplaces, shelving, etc.)
- The "golden hour lighting" effect is applied TO the existing architecture, not by changing it
- Every window, door, wall, and architectural element must remain EXACTLY where it is in the original photo

ARCHITECTURE DIGEST - LIGHTING (CRITICAL FOR ALL ROOMS):
Apply ALL of these lighting transformations TO THE EXISTING ARCHITECTURE (do not alter structure):
1. Golden hour quality - entire scene looks like 1 hour before sunset
2. Visible warm light rays streaming through EXISTING windows (do not add/move windows)
3. Rich dimensional shadows (warm brown/amber tones, NOT flat gray)
4. Interior glow effect - space feels lit from within
5. Color temperature 2700K-3000K - NO cool/blue tones anywhere
6. All whites become cream/ivory, all shadows become warm amber

ARCHITECTURE DIGEST - LIVING ROOM:
- PRESERVE: All walls, windows, doors, flooring, fireplace, built-ins EXACTLY as they are
SOFA (signature element - choose one):
  □ Curved serpentine sofa in ivory/cream bouclé, low profile, rounded arms, no visible legs or short tapered oak legs. 84-96" length. Vladimir Kagan inspired organic shape.
  □ OR curved sofa in warm charcoal or mushroom velvet. Same silhouette.
  □ OR deep slope-arm sofa in oatmeal Belgian linen, loose cushions, white oak base. 90-108" length.
COFFEE TABLE (choose one):
  □ Organic curved shape (kidney/cloud/freeform) in bleached white oak or natural ash. Thick 2-3" top, rounded edges. Noguchi inspired.
  □ OR round drum in natural woven rattan or seagrass, 36-42" diameter.
  □ OR round hammered brass/bronze with aged patina, 36-40" diameter.
ACCENT CHAIRS (always pair, angled 45° toward sofa):
  □ Pair of barrel swivel chairs in cream/ivory bouclé, brass swivel base. 30-32" width.
  □ OR pair of mid-century lounge chairs in cognac/saddle leather, walnut frames.
RUG: Vintage Persian in FADED earth tones (muted rust, cream, sage, soft blue) 9x12 or 10x14. OR chunky woven jute in natural honey 8x10+.
ART: One large abstract 48x60" minimum (ideally 60x72") in earth tones. Thin natural oak or walnut float frame. Hung 6-8" above sofa.
ACCESSORIES: Stack of 3-4 art/architecture books on coffee table, small sculptural ceramic beside books, chunky knit throw in cream draped on sofa arm, 2-3 accent pillows (cream, sage, warm taupe), large woven seagrass basket on floor.
PLANT (large tree allowed here): Olive tree 6-7 ft in aged terracotta pot 18-24" diameter OR fiddle leaf fig 6-7 ft in woven seagrass basket. One corner only.
LIGHTING: Arc floor lamp with brass arm, linen shade, behind sofa.

ARCHITECTURE DIGEST - DINING ROOM:
- PRESERVE: All walls, windows, doors, flooring, wainscoting, built-ins EXACTLY as they are
TABLE (choose one):
  □ Solid white oak rectangular, Parsons-style legs, natural finish. 72-84" for 6 seats.
  □ OR walnut slab with natural live edge, blackened steel trestle base. 84-96" length.
  □ OR round travertine/limestone on sculptural pedestal, 54-60" diameter.
CHAIRS (all must match):
  □ Hans Wegner CH24 Wishbone chairs in natural ash/oak, paper cord seats. 6 standard.
  □ OR sculptural solid wood (ash/pine) with curved back, carved seat.
PENDANT (centered over table, 30-34" above surface):
  □ Brass drum pendant 18-24" diameter, aged/patinated finish.
  □ OR large ceramic pendant in matte cream, conical/dome shape.
RUG: Natural jute in chunky weave, 9x12, extending 24-30" beyond chairs all sides.
CENTERPIECE: Table EMPTY (preferred) OR single sculptural cream ceramic vase (10-14" height) with 3-5 dried olive branches, slightly off-center.
ART: One large piece on focal wall. Abstract in earth tones 40x50" to 48x60".
PLANT (NO full tree - branches only): Tall floor vase (24-36") with dried branches/pampas in corner. Vase in cream, terracotta, or charcoal.

ARCHITECTURE DIGEST - KITCHEN (Occupied - Styling Enhancement Only):
- PRESERVE: All cabinets, countertops, appliances, sink, windows, flooring EXACTLY as they are
KEEP MINIMAL - 3-4 items maximum:
VIGNETTE 1 (Near Stove): Large olive wood cutting board (16x20"+) at casual angle with rustic sourdough loaf. Small ceramic pinch bowl with flaky salt.
VIGNETTE 2 (Island/Counter): Shallow wooden bowl (12-14" diameter) with 6-8 whole Meyer lemons. Position casually, not centered.
VIGNETTE 3 (Near Sink): Small terracotta pot (4-6") with fresh rosemary or thyme.
SIGNATURE FLOWER: Single pink king protea stem in sculptural ceramic vase (round/bulbous, 8-10" height, matte charcoal or terracotta). ONE STEM ONLY.
BAR STOOLS (if island, 2-3): Woven saddle leather on light oak frame OR natural rattan with black metal legs.
DO NOT ADD: Books, large plants/trees, excessive accessories.

ARCHITECTURE DIGEST - BEDROOM:
- PRESERVE: All walls, windows, doors, closets, flooring, ceiling EXACTLY as they are
BED (choose one):
  □ Low platform with tall upholstered headboard (48-54" height) in oatmeal/cream Belgian linen. Tight upholstery, no tufting, rounded top corners.
  □ OR solid walnut/oak platform with integrated headboard. Low profile, clean lines.
  □ OR natural cane/rattan paneled headboard in light oak frame.
BEDDING (layered, lived-in): White/cream LINEN sheets slightly rumpled, cream linen duvet pulled back casually on one side, chunky knit or waffle-weave throw in oatmeal at foot draped casually. 2-3 Euro shams in cream, 2-3 accent pillows (cream/sage/warm taupe).
NIGHTSTANDS (matching pair): Sculptural hourglass or drum shape in natural white oak 22-24" height. OR floating wall-mounted shelves in walnut.
LAMPS (matching pair): Ceramic with sculptural organic base in warm cream or sage, natural linen drum shade, 24-28" height.
RUG: Vintage Persian in faded earth tones extending 24-36" beyond bed. 9x12 for queen, 10x14 for king. OR natural jute 8x10+.
CURTAINS: Flowing linen in warm white/cream, mounted high, puddling on floor.
ART: Large calming abstract above bed (40x50" to 60x40") in soft muted tones. Natural oak float frame.
PLANT (small or none): Small plant on ONE nightstand (trailing pothos, succulent) OR nothing. Large tree only if room is very spacious.

ARCHITECTURE DIGEST - BATHROOM:
- PRESERVE: All vanity, sink, toilet, tub/shower, tile, mirrors, flooring EXACTLY as they are
SIGNATURE (essential - non-negotiable): Sculptural ceramic vase in matte charcoal/black/terracotta, round/bulbous organic shape 8-12" height, with 1-2 pink king protea stems. Position prominently on vanity.
VANITY TRAY: Black slate, gray marble, or dark stone tray (8x12" or 10" round) containing: natural artisan bar soap (cream colored), small brass dish. Maximum 3 items on tray.
TOWELS: Charcoal gray (preferred) OR cream OR soft sage. Plush, high-quality. Hung neatly on brass ring/bar OR rolled in basket OR folded stack.
SMALL ACCENT (pick 1-2 only): Single pillar candle in cream, OR small maidenhair fern in ceramic pot, OR eucalyptus stems in clear glass vase.
BASKET: Woven seagrass on floor with neatly rolled extra towels.

ARCHITECTURE DIGEST - EXTERIOR (LIGHTING TRANSFORMATION ONLY):
*** CRITICAL: The exterior transformation is LIGHTING/ATMOSPHERE ONLY - do NOT alter the home's structure ***
- PRESERVE: The home's exact structure, all windows in their exact positions, all doors, roof, siding, landscaping layout, driveway
- Do NOT add windows, remove windows, change window sizes, or alter window positions
- Do NOT change the home's footprint, roofline, or architectural features
- Do NOT alter the landscaping layout (you may enhance lighting on existing plants)
SKY TRANSFORMATION: Golden hour gradient - soft cornflower blue at top → warm golden/amber in middle → soft peach/pink at horizon. Wispy clouds catching golden light. Sun below frame or at horizon.
WINDOW GLOW (critical - non-negotiable): EVERY visible window must show warm amber interior light (2700K warm incandescent look). Windows become beacons of warmth against twilight.
SIGNATURE LANDSCAPING: Mature olive tree (6-8 ft) in large aged terracotta pot (20-26") near front entry, beside garage, or on porch. ONE tree only.
ADDITIONAL: String lights (Edison bulbs) on deck/patio if appropriate. Outdoor furniture: teak or weathered wood with gray/cream cushions.
LANDSCAPE ENHANCEMENT: Lawn warmer golden-green tone, trees catching golden side-light, long shadows across lawn.
OVERALL EFFECT: "Dwell magazine cover at sunset" quality through LIGHTING, not structural changes

=== PROPERTY-WIDE CONSISTENCY RULES (ARCHITECTURE DIGEST) ===

PLANT DISTRIBUTION: Maximum 2 large trees per property.
- Living room: Large tree allowed (olive OR fiddle leaf)
- Dining room: BRANCHES in vase ONLY - NOT full tree
- Kitchen: Small herb OR protea flower - NOT tree
- Bedroom: Small plant OR nothing - tree only if very large room
- Bathroom: Small fern OR eucalyptus stems - NOT tree
- Exterior: Olive tree at entry - ONE location only
- If olive tree in living room, use BRANCHES in dining vase

METALS (choose one family for entire property):
- Warm: brass, bronze, aged gold throughout
- OR Dark: matte black, oil rubbed bronze throughout
- DO NOT mix chrome with brass

RUGS (one family):
- All vintage Persian (varied but complementary)
- OR all natural fiber (jute, sisal, seagrass)
- OR all neutral modern (solid textures)

CERAMICS (consistent palette, choose 3-4):
- Cream/warm white, charcoal/matte black, terracotta, sage (accent)

=== GOLD EXAMPLES (STRIP + REFURNISH) ===

These examples show the exact behavior expected: strip all furniture first, then refurnish from scratch.

EXAMPLE A - Living Room (may be vacant or occupied) → NEUTRAL style:
"Using this uploaded living room photo, create a fully virtually staged real-estate listing image in a Neutral style.

Keep the same architectural shell: walls, windows, doors, trim, ceiling height, flooring, and any built-in elements must stay exactly where they are. Do not move walls, add or remove windows, change flooring, or alter the size or position of doors or openings. Do not fix, repaint, patch, or cover any visible cracks, stains, holes, or other damage; these must remain visible.

First, STRIP THE ROOM: remove all existing furniture and decor, including sofas, chairs, tables, shelves, TV, rugs, freestanding lamps, and personal items, so the room appears unfurnished while still showing the true architecture and any visible defects. Do not hide or cover damaged areas with furniture or rugs.

Then, RE-FURNISH the room completely in a Neutral style:

Add a clean-lined light-colored sofa (soft white or light beige) against the main wall, sized realistically for the room.

Add one or two simple accent chairs with slim legs, and a low rectangular coffee table in light wood or white.

Add a simple TV console or sideboard if appropriate, with minimal decor on top.

Place a neutral area rug (light grey or beige) that fits under the front legs of sofa and chairs, scaled correctly to the floor area.

Add one or two simple table or floor lamps, and 2–3 pieces of simple, abstract wall art in soft neutral tones.
All furniture and decor must be realistic in scale for the actual room size – do not oversize or undersize items to make the room look bigger.

LEVEL THE PHOTO: straighten the image so horizontals are level and vertical lines (walls, windows, door frames) are truly vertical. You may adjust camera height and vertical angle slightly to improve composition, but do not move the camera horizontally or swing it left/right to reveal new walls or areas that are not visible in the original photo. Do not widen the field of view or make the room appear larger or deeper than it is.

Apply professional-level photo enhancement: correct exposure, increase contrast slightly, and improve overall clarity so details in the flooring, trim, and furniture are crisp and well defined. Correct white balance so whites and light surfaces look clean and true-to-life without becoming unnaturally bright. Reduce noise and haze so the image looks sharp and high-quality, similar to a DSLR real-estate photograph. The final result should be a photorealistic, fully staged Neutral living room that still honestly reflects the true architecture and condition of the property."

EXAMPLE B - Bedroom (may be vacant or occupied) → LUXURY style:
"Using this uploaded bedroom photo, create a fully virtually staged real-estate listing image in a Luxury style.

Keep the same architectural shell: walls, windows, doors, trim, ceiling height, closets, and flooring exactly as they appear. Do not move walls, add or remove windows, change flooring, or alter door sizes or positions. Do not fix, repaint, patch, or cover any visible cracks, stains, or other damage on walls, trim, or floors; these must remain visible.

First, STRIP THE ROOM: remove all existing furniture and decor, including any current bed, nightstands, dressers, lamps, rugs, and personal items, so the room appears unfurnished while preserving the real architecture and any visible defects. Do not hide damage behind furniture, curtains, or rugs.

Then, RE-FURNISH the room completely in a Luxury style (NOT sterile hotel-lobby beige):

Add a king or queen-size upholstered headboard in a plush fabric like bouclé or velvet (taupe, greige, or soft charcoal), centered on the main wall. This is the FOCAL POINT - make it substantial and inviting.

Place matching nightstands on each side in rich wood or lacquer finish, each with a statement table lamp (brass, sculptural, or with a textured shade).

Dress the bed with layered bedding using VARIED TEXTURES: crisp white linen sheets, a textured duvet, and 3-5 decorative pillows mixing velvet, linen, and wool in neutral tones plus ONE muted accent color (navy, forest green, or warm rust). Add a casually draped throw at the foot with natural, realistic folds - not frozen stiff.

Add a large plush area rug under the bed that extends well beyond the sides and foot, in a sophisticated pattern or solid with visible texture.

If space allows, add a low upholstered bench at the foot in velvet or bouclé, or a single accent chair in the corner.

Add 1-2 pieces of large contemporary art above the bed - muted, sophisticated, not loud - to complete the focal point.

All furniture must feel substantial and custom, not generic. Layer textures so the room feels tactile and lived-in, not flat.
All furniture and decor must be realistically scaled to the room size.

LEVEL THE PHOTO: straighten the image so horizontals are level and vertical lines are truly vertical. You may adjust camera height and pitch slightly.

Apply professional-level photo enhancement: correct exposure, increase contrast gently for drama, sharpen details. The final result should be a photorealistic, high-end designer bedroom that feels inviting and luxurious - NOT a sterile hotel room."

EXAMPLE C - Occupied Living Room → MODERN style (demonstrating strip behavior on furnished room):
"Using this uploaded living room photo, create a fully virtually staged real-estate listing image in a Modern style.

Keep the same architectural shell: walls, windows, doors, trim, ceiling height, flooring, fireplace, and any built-in elements must stay exactly where they are. Do not move walls, add or remove windows, change flooring, or alter architectural features. Do not fix, repaint, patch, or cover any visible cracks, stains, holes, or other damage; these must remain visible.

First, STRIP THE ROOM: remove ALL existing furniture and decor – the current sofa, armchairs, coffee table, side tables, lamps, rugs, artwork, plants, and all personal items – so the room appears completely unfurnished while still showing the true architecture and any visible defects. Even though this room is currently furnished, remove everything.

Then, RE-FURNISH the room completely in a Modern style:

Add a low-profile, boxy sofa in charcoal or white facing the focal wall, sized appropriately for the room.

Add a thin-legged glass or metal coffee table, and one or two accent chairs with clean geometric lines.

Place a bold geometric area rug in black/white or with a single accent color.

Add a minimalist floor lamp with an arc or sculptural design, and a simple table lamp on a slim metal side table.

Include 1–2 pieces of bold, minimal wall art – large scale, abstract, high contrast.

Add one sculptural plant in a modern pot.
All furniture must be properly scaled to the actual room dimensions. Do not use oversized furniture. Leave realistic walkways.

LEVEL THE PHOTO: straighten the image so all verticals are truly vertical. You may adjust camera height slightly. Do not move horizontally, widen the field of view, or make the room appear larger.

Apply professional-level photo enhancement: correct exposure, increase contrast for that sharp modern look, fix white balance, reduce noise. The final result should be a photorealistic, fully staged Modern living room that honestly reflects the true architecture and condition of the property."

EXAMPLE D - Living Room with Built-ins → LUXURY style (demonstrating texture layering, focal points, shelf styling):
"Using this uploaded living room photo, create a fully virtually staged real-estate listing image in a Luxury style.

Keep the same architectural shell: walls, windows, doors, trim, ceiling height, flooring, fireplace, and built-in shelving exactly as they appear. Do not alter any architectural features or fix any visible damage.

First, STRIP THE ROOM: remove all existing furniture, decor, and items on shelves so the room appears unfurnished.

Then, RE-FURNISH in a Luxury style (NOT sterile hotel-lobby beige):

FOCAL POINT: Create a striking focal point at the fireplace - add a large piece of contemporary art above the mantel, and style the mantel with 2-3 sculptural objects in varied heights. Add a beautifully styled coffee table with a curated arrangement (large art book, small sculptural object, fresh greenery).

SEATING: Add a substantial sofa in plush bouclé or velvet (taupe, greige, or soft charcoal) - not generic. Add two accent chairs in a complementary fabric with visible texture. Include throw pillows mixing velvet, linen, and wool in neutral tones plus ONE muted accent color (muted blue-green or warm rust) for character.

RUG: Place a LARGE area rug with sophisticated pattern or visible texture that extends under all seating - front legs of sofa and chairs must be on the rug so furniture reads as one cohesive conversation area.

SHELF STYLING: Style built-in shelves with a CURATED, VARIED mix - books of different heights and muted spine colors, sculptural objects, small plants, 1-2 framed photos. Leave some negative space. NEVER repeat the same object or book stack across shelves.

LIGHTING: Add statement table lamps (brass or sculptural) on side tables, and a floor lamp with visible texture or brass finish.

MICRO-REALISM: Allow subtle natural variation - a throw casually draped with realistic folds, pillows that aren't perfectly rigid.

All furniture must feel substantial and custom. Layer textures so the room feels tactile and lived-in, not flat.

LEVEL THE PHOTO and apply professional enhancement with slightly more contrast for drama. The final result should be a photorealistic, high-end designer living room that feels inviting and luxurious - NOT a sterile hotel lobby."

EXAMPLE E - Dining Room → ARCHITECTURE DIGEST style (demonstrating dramatic lighting + designer furniture):
"Using this uploaded dining room photo, create a fully virtually staged real-estate listing image in an Architecture Digest style.

ARCHITECTURAL PRESERVATION (CRITICAL): Keep the same architectural shell: walls, windows, doors, trim, ceiling height, and flooring exactly as they appear. Do NOT move, add, remove, or resize any windows or doors. Do NOT alter any architectural features. Do NOT fill in doorways or openings. Do NOT fix, repaint, patch, or cover any visible cracks, stains, holes, or other damage - these must remain visible.

First, STRIP THE ROOM: remove all existing furniture and decor so the room appears unfurnished while preserving the exact architecture.

Then, RE-FURNISH in the Architecture Digest style with THREE TRANSFORMATION LAYERS:

LAYER 1 - DRAMATIC LIGHTING (CRITICAL - this is PHOTO ENHANCEMENT, not structural change):
Transform the lighting to golden hour quality - the entire scene should look like 1 hour before sunset. Add visible warm light rays streaming through the EXISTING windows (do not add/move windows). Create rich dimensional shadows in warm brown/amber tones (NOT flat gray). Apply an interior glow effect so the space feels lit from within. Use color temperature 2700K-3000K throughout - absolutely NO cool or blue tones anywhere. All whites become cream/ivory, all shadows become warm amber.

LAYER 2 - DESIGNER STAGING (furniture only, not architectural):
Add a solid light oak dining table with clean lines, surrounded by 6 Hans Wegner wishbone chairs in natural ash.
Place a large natural jute rug with subtle texture under the table.
Add ONE large abstract artwork (at least 36x48 inches) in earth tones (terracotta, sage, cream) with a simple float frame on the main wall.
Include an olive tree (5-6 feet tall) in an aged terracotta pot in the corner as the SIGNATURE element.
Add a brass or ceramic pendant light with warm glow above the table.
Keep the table centerpiece MINIMAL - either empty or a single sculptural vase with olive branches.

LAYER 3 - COLOR GRADING:
Push the entire image warm/golden. Whites = cream (never stark). Shadows = warm brown/amber (never gray). Wood tones = rich honey/amber.

All furniture must be realistically scaled. Do NOT use books as decor in the dining room. Do NOT alter the room's architecture. The walls, windows, doors, and floor must be IDENTICAL to the original photo. The final result should be a photorealistic, magazine-cover quality dining room with dramatic golden-hour lighting that honestly represents the actual space."

EXAMPLE F - Exterior → ARCHITECTURE DIGEST style (demonstrating LIGHTING transformation - NOT structural):
"Using this uploaded exterior photo, create a fully virtually staged real-estate listing image in an Architecture Digest style.

*** CRITICAL ARCHITECTURAL PRESERVATION ***
This is a LIGHTING transformation ONLY. The home's structure must remain EXACTLY as photographed:
- Do NOT move, add, remove, or resize ANY windows
- Do NOT alter the home's footprint, roofline, or siding
- Do NOT change the landscaping layout or add/remove trees (you may enhance lighting on existing plants)
- Do NOT fill in or alter any doors, openings, or architectural features
- Every window must remain in its EXACT original position
- The home must be immediately recognizable as the same property

TRANSFORM THE LIGHTING AND ATMOSPHERE with THREE LAYERS:

LAYER 1 - GOLDEN HOUR SKY (LIGHTING CHANGE ONLY):
Transform the sky to a dramatic golden hour gradient: blue at the top transitioning to warm gold in the middle, then pink/peach tones at the horizon. This is the 'magic hour' - 1 hour before sunset.

LAYER 2 - WINDOW GLOW (LIGHTING CHANGE ONLY):
EVERY EXISTING window must show warm amber interior glow - light should be visibly emanating from within the home. This is applied to the windows that ALREADY EXIST - do not add windows or change their positions.

LAYER 3 - LIGHTING ON ARCHITECTURE:
Apply golden side-light to EXISTING trees and landscaping, with warm highlights on foliage. The home's exterior surfaces should catch warm evening light.

ADDED ELEMENTS (decor only, not structural):
Add an olive tree in an aged terracotta pot near the entry (this is portable decor).
If there's a porch, add natural wood or woven outdoor furniture (this is furniture).

COLOR GRADING:
Push the entire image warm. Eliminate any cool/blue tones except in the upper sky. Whites on the home become warm cream in the golden light.

The overall effect should be 'Dwell magazine cover at sunset' quality achieved through LIGHTING AND ATMOSPHERE, not structural changes. The home must be the EXACT SAME home - just photographed at magic hour with beautiful lighting.

LEVEL THE PHOTO so all verticals are truly vertical. The final result should be a photorealistic, magazine-cover quality exterior that honestly represents the actual property."

=============================================================================
⚠️ FINAL STRUCTURAL CHECK (ARCHITECTURE DIGEST) ⚠️
=============================================================================
Before finalizing ANY Architecture Digest image, verify:
✓ All doorways and openings to adjacent rooms are preserved exactly
✓ All windows remain in their exact original positions
✓ All walls remain exactly where they were in the original
✓ No architectural features have been added or removed
✓ If you could see into another room (kitchen, hallway, etc.), that view is still visible
✓ The room is immediately recognizable as the same space from the original photo
=============================================================================
{comments_section}

=============================================================================
⚠️ FINAL VERIFICATION CHECKLIST - ALL STYLES (MANDATORY) ⚠️
=============================================================================
Before generating ANY staged image, you MUST verify:

□ CAMERA: Does the output use the EXACT same camera angle as the original?
  - Same viewpoint, same perspective, same field of view
  - No rotation, shift, pan, or tilt
  - Before/after would overlay perfectly

□ WALLS: Are ALL walls in the EXACT same positions as the original?
  - No walls moved, added, or removed
  - Same wall angles and proportions

□ WINDOWS: Are ALL windows in the EXACT same positions as the original?
  - Same number of windows
  - Same positions on walls
  - Same sizes and styles

□ DOORS: Are ALL doors/openings in the EXACT same positions as the original?
  - No doorways filled in or added
  - Same positions and sizes

□ CEILING: Is the ceiling IDENTICAL to the original?
  - No track lighting added
  - No recessed lights added
  - No skylights added
  - No beams added
  - Crown molding preserved

□ FLOOR: Is the flooring IDENTICAL to the original?
  - Same material and pattern
  - Same boundaries

□ FURNITURE POSITION: Is major furniture on the SAME WALLS as the original?
  - Bed on same wall as original (if bedroom)
  - Sofa facing same direction as logical for room

□ PROPORTIONS: Does the room appear the SAME SIZE as the original?
  - Not wider, not deeper, not taller
  - Same spatial proportions

□ NO INVENTED DAMAGE: Are walls clean where items were removed?
  - No fake mounting holes, cracks, or marks invented

If ANY of these checks fail, the image is INVALID and misrepresents the property.
=============================================================================

{STRUCTURAL_VERIFICATION_REMINDER}

=== YOUR TASK ===

Analyze the uploaded photo and respond with ONLY valid JSON (no markdown, no code blocks):
{{
    "room_type": "kitchen|bathroom|bedroom|living_room|dining_room|hallway|exterior|office|other",
    "is_occupied": true|false,
    "issues": ["list", "specific", "issues", "you", "see"],
    "suggested_style": "{style_preference}",
    "staging_prompt": "your full 4-paragraph prompt following the skeleton above"
}}

Issues to identify: vacant_room, clutter, dim_lighting, crooked_angle, personal_items, messy_bed, items_on_floor, crowded_counters, visible_wires, strong_color_cast, needs_staging, etc.

IMPORTANT: Be SPECIFIC in your staging_prompt:
- For vacant rooms: Describe exactly what furniture to add, with colors and materials
- For occupied rooms: List specific items to remove and any decor to add
- Reference the actual features of the room (flooring type, window placement, etc.)
- The more specific and detailed, the better the result.
"""
    
    async def analyze_image(
        self,
        image_path: Path,
        style_preference: str = "modern",
        comments: str = None,
        max_retries: int = 3
    ) -> GeminiAnalysisResult:
        """
        Analyze a single image and generate virtual staging prompt.

        The AI model will auto-detect whether the room is vacant or occupied.

        Args:
            image_path: Path to the image file
            style_preference: Staging style (modern, scandinavian, coastal, farmhouse, midcentury, architecture_digest)
            comments: Client's special instructions for staging
            max_retries: Number of retries on transient failures

        Returns:
            GeminiAnalysisResult with room analysis and staging prompt
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
        # Let the AI auto-detect occupied status from the image
        # Pass False as default, the prompt instructs the AI to detect and report actual status
        system_prompt = self._build_analysis_prompt(is_occupied=False, style_preference=style_preference, comments=comments)

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
        Analyze all images for a job and create virtual staging prompts.

        Args:
            job_id: Job identifier
            job_dir: Job directory path
            order: Order with job metadata (style and comments)
            image_paths: List of relative paths to raw images

        Returns:
            Plan with analysis and virtual staging prompts for all images
        """
        logger.info(f"Starting virtual staging analysis for job {job_id} with {len(image_paths)} images")
        logger.info(f"Style: {order.style}, Comments: {order.comments or 'None'}")

        plan = Plan(job_id=job_id, images=[])

        for i, rel_path in enumerate(image_paths):
            image_id = f"img_{i+1}"
            abs_path = job_dir / rel_path

            logger.info(f"Analyzing image {image_id}: {rel_path}")

            try:
                result = await self.analyze_image(
                    image_path=abs_path,
                    style_preference=order.style,
                    comments=order.comments
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

        logger.info(f"Completed virtual staging analysis for job {job_id}")
        return plan
