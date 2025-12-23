# Architecture Digest Style - Improved Prompt Framework v3
## Addressing Gaps Identified in Current Outputs

Based on analysis of actual Stage Vision outputs vs. Adam Potts portfolio, this document provides enhanced prompts to close the gap between "nice virtual staging" and "Architectural Digest cover."

---

## KEY INSIGHT: The 3 Transformation Layers

Every image needs transformation in THREE layers, not just staging:

```
Layer 1: LIGHTING (most important - this makes it "editorial")
Layer 2: STAGING (furniture and decor)
Layer 3: COLOR GRADING (cohesive warmth across all images)
```

Current outputs are doing Layer 2 (staging) well but Layers 1 and 3 need significant improvement.

---

## IMPROVED GEMINI ANALYSIS PROMPT

Replace the existing Architecture Digest prompt with this enhanced version:

```python
ARCHITECTURE_DIGEST_STYLE_GUIDE_V3 = """
## ARCHITECTURE DIGEST STYLE - EDITORIAL TRANSFORMATION

Transform this real estate photo into an editorial-quality image suitable for Architectural 
Digest magazine cover. This requires THREE transformation layers applied together:

=============================================================================
LAYER 1: LIGHTING TRANSFORMATION (MOST CRITICAL)
=============================================================================

This is what separates "nice staging" from "magazine cover." Apply ALL of these:

1. **GOLDEN HOUR QUALITY**
   - Transform lighting to feel like 1 hour before sunset
   - Warm color temperature (2700K-3000K) throughout entire image
   - NOT just a color filter - actual directional warm light

2. **VISIBLE LIGHT RAYS**
   - Add subtle golden light rays streaming through windows
   - Light should visibly "pour" into the space
   - Creates the signature editorial drama

3. **RICH SHADOWS**
   - Create dimensional shadows - NOT flat lighting
   - Shadows should be warm-toned, not gray or cool
   - Side lighting effect emphasizing texture and depth

4. **WINDOW TREATMENT**
   - Windows should glow with soft golden/white light
   - Balance exposure so windows aren't blown out but are luminous
   - If exterior visible, it should have golden hour sky

5. **INTERIOR GLOW EFFECT**
   - The entire interior should feel "lit from within"
   - Warm and inviting, like the space is glowing
   - Particularly important for exterior shots - windows should emit warm glow

=============================================================================
LAYER 2: STAGING (Furniture & Decor)
=============================================================================

**FURNITURE SPECIFICATIONS (Be Precise)**

DINING ROOM:
- Table: Solid light oak or walnut with organic/live edge OR clean Parsons style
- Chairs: Hans Wegner wishbone chairs in natural ash OR wire-frame dining chairs
- NOT: Dark wood, NOT: Upholstered dining chairs, NOT: Farmhouse style
- Rug: Natural fiber (jute/sisal) with subtle pattern OR muted vintage Persian
- Centerpiece: EMPTY or single sculptural ceramic vase with olive branches/dried grasses
- Art: One LARGE abstract piece (earth tones: terracotta, sage, cream, rust) in simple float frame

LIVING ROOM:
- Sofa: CURVED silhouette in WHITE BOUCLÉ or cream linen - this is signature
- Coffee table: Organic shape - woven, carved wood slab, or sculptural glass
- Accent chairs: Cream swivel chairs OR cognac leather lounge chairs
- NOT: Sectionals, NOT: Dark sofas, NOT: Traditional silhouettes

KITCHEN (Occupied - Style Enhancement):
- Remove: Generic decor, cluttered counters, dated accessories
- Add: Wood cutting board (olive or walnut), ceramic bowl with citrus (lemons)
- Add: Small herb plant OR single stem in ceramic bud vase by sink
- If counter space: Artisanal ceramic vessel with pink PROTEA flowers
- Runner: Muted vintage-style runner in earth tones (optional, only if space allows)
- NOT: Coffee table books (wrong room), NOT: Too many accessories

BEDROOM:
- Bed: Low platform style with upholstered headboard in oatmeal/cream linen
- Bedding: LAYERED - white/cream base, add textured throw, linen pillows
- Nightstands: Sculptural or floating, natural wood or stone
- Lighting: Ceramic or sculptural table lamps (not traditional)
- Plant: Single fiddle leaf or olive tree in ceramic pot

BATHROOM:
- Signature: Sculptural dark ceramic vase with PINK PROTEA (this is essential)
- Accessories: Stone tray, single soap, brush - minimal
- Towels: Plush, neatly folded or rolled, in charcoal or cream
- Plant: Small architectural plant (string of pearls, small succulent)

**SIGNATURE ELEMENTS CHECKLIST**
Include 2-3 per interior image for consistency:
□ Pink/coral PROTEA flowers in sculptural ceramic vase
□ Olive tree OR fiddle leaf fig in terracotta/ceramic pot  
□ Ceramic sculptural objects
□ Art books (living room coffee table ONLY)
□ Woven basket or natural texture element
□ Brass/bronze accent (candle holder, tray, hardware)

=============================================================================
LAYER 3: COLOR GRADING (Consistency)
=============================================================================

Apply this color treatment to EVERY image for cohesive editorial feel:

1. **OVERALL WARMTH**
   - Push entire image toward warm/golden tones
   - Whites should be cream, never stark/cool
   - No blue or cool color casts anywhere

2. **WOOD TONES**
   - Enhance natural wood to look rich and warm
   - Honey, amber, and golden tones
   - Even existing cabinets should feel warmer

3. **SHADOW COLOR**
   - Shadows should have warm brown/amber tone
   - Never gray, never blue, never cool
   - Rich and dimensional

4. **HIGHLIGHT TREATMENT**
   - Highlights should be soft and warm
   - Creamy whites, not pure white
   - Gentle roll-off, not harsh

=============================================================================
EXTERIOR SPECIFIC TRANSFORMATION
=============================================================================

Exteriors require the MOST dramatic transformation:

1. **SKY REPLACEMENT/ENHANCEMENT**
   - Transform to GOLDEN HOUR sky
   - Soft gradient: warm blue → golden → soft pink/peach at horizon
   - NOT harsh blue, NOT midday harsh light
   - Soft clouds catching golden light are ideal

2. **INTERIOR GLOW**
   - ALL windows should show warm interior light
   - This is critical - creates the "magazine" effect
   - Soft warm glow emanating from within

3. **LANDSCAPE ENHANCEMENT**
   - Enhance existing trees to feel golden-lit
   - If lawn is brown/patchy, improve color subtly (but don't fake green lawn)
   - Add: Olive tree in large terracotta pot at entry
   - Add: Architectural plantings if appropriate (ornamental grasses)

4. **REMOVE DISTRACTIONS**
   - Power lines (minimize or remove if possible)
   - Harsh shadows
   - Visible cars/clutter

=============================================================================
EXAMPLE COMPLETE PROMPTS
=============================================================================

**DINING ROOM EXAMPLE:**
"Transform lighting to golden hour quality with visible warm light streaming through 
windows and rich dimensional shadows. Remove existing furniture and staging. Add solid 
light oak dining table with six Hans Wegner wishbone chairs in natural ash. Large natural 
fiber rug extending under table. Single large abstract painting (earth tones: terracotta, 
sage, cream) in simple oak float frame on main wall. Tall olive tree in terracotta pot 
in corner. Empty table or single cream ceramic vase with dried olive branches. Apply warm 
color grading throughout - cream whites, warm shadows, golden tones."

**KITCHEN EXAMPLE (Occupied):**
"Transform lighting to warm golden hour quality - enhance natural light through windows 
to feel like late afternoon sun streaming in. Keep existing cabinetry but enhance wood 
tones to feel richer and warmer. Remove generic accessories. Add: olive wood cutting board 
near stove, cream ceramic bowl with whole lemons on counter, single stem pink protea in 
sculptural dark ceramic vase by sink. If space allows, add muted earth-tone vintage runner. 
Apply warm color grading - no cool tones, cream whites, warm amber shadows."

**EXTERIOR EXAMPLE:**
"Transform to golden hour/dusk lighting. Replace or enhance sky to soft golden hour 
gradient (warm blue to golden to soft peach at horizon). Add warm interior glow visible 
through all windows - soft amber light emanating from within. Enhance tree foliage to 
catch golden light. Add mature olive tree in large terracotta pot near front entry. 
Improve lawn color slightly (warmer, healthier tone, but realistic). Minimize or remove 
power lines. Overall warm, editorial quality - this should look like a Dwell magazine cover."

=============================================================================
CONSISTENCY REQUIREMENTS
=============================================================================

For a multi-image property shoot, ALL images must share:

1. Same color temperature (warm golden throughout)
2. Same "time of day" feeling (all golden hour, not mixed)
3. Same accessory vocabulary (if protea in kitchen, similar in bathroom)
4. Same furniture family (all natural wood, no style mixing)
5. Same lighting quality (all dramatic editorial, not flat)

This creates the "editorial photoshoot" feeling vs. "random staged photos."

=============================================================================
DO NOT
=============================================================================

- Use cool or blue lighting ANYWHERE
- Add generic/catalog furniture
- Over-stage with too many accessories  
- Mix furniture styles (no farmhouse + modern)
- Leave lighting flat/even
- Forget the signature elements (protea, olive tree, ceramics)
- Apply inconsistent color grading between images
- Leave exteriors as midday harsh light
"""
```

---

## IMPROVED FALLBACK PROMPT

```python
ARCHITECTURE_DIGEST_FALLBACK_V3 = """
Transform into Architectural Digest editorial quality:

LIGHTING (CRITICAL):
- Golden hour quality - warm light streaming through windows
- Rich dimensional shadows (not flat)
- Interior glow effect - space feels lit from within
- Color temp: 2700K warm, no cool/blue anywhere

STAGING:
- Dining: Light oak table, wishbone chairs, olive tree, abstract art
- Kitchen: Cutting board, lemons in ceramic bowl, protea in vase
- Living: Curved bouclé sofa, organic coffee table, indoor tree
- Exterior: Golden hour sky, warm window glow, olive tree at entry

SIGNATURE ELEMENTS (add 2-3):
- Pink protea flowers in dark ceramic vase
- Olive tree in terracotta pot
- Ceramic sculptural objects
- Art books (living room only)

COLOR GRADING:
- Everything warm/golden
- Cream whites, never stark
- Warm brown shadows, never gray
- Rich amber wood tones

EXTERIOR CRITICAL:
- Sky must be golden hour (golden/pink/soft blue gradient)
- Windows MUST show warm interior glow
- Add olive tree near entry
"""
```

---

## SPECIFIC FIXES FOR OBSERVED ISSUES

### Issue: Dining Room Staging Quality

**Current:** Good furniture but feels "catalog"
**Fix:** More specific furniture + better accessory placement

```
BEFORE PROMPT: "Add dining table with wishbone chairs and olive tree"

AFTER PROMPT: "Add solid light oak rectangular dining table (clean lines, not rustic) 
with six Hans Wegner wishbone chairs in natural ash with woven paper cord seats. 
Large natural fiber rug (jute with subtle geometric pattern) extending 24" beyond 
chairs on all sides. One large-scale abstract painting (36"x48" minimum, earth tones: 
terracotta, sage, rust, cream) in thin natural oak float frame on focal wall. 
Tall olive tree (5-6 ft) in aged terracotta pot in corner nearest window. 
Table is EMPTY or has single large sculptural cream ceramic vase with 3 dried 
olive branches. No other accessories."
```

### Issue: Kitchen Feels Staged Not Curated

**Current:** Added good elements but coffee table books feel wrong
**Fix:** Kitchen-appropriate accessories only

```
BEFORE PROMPT: "Add protea flowers, cutting board, citrus, books"

AFTER PROMPT: "Style kitchen counters minimally: olive wood cutting board 
angled near stove with artisan bread loaf, shallow cream ceramic bowl with 
6-8 whole lemons on island/peninsula, small terracotta pot with fresh herbs 
(rosemary or thyme) on windowsill, sculptural dark ceramic vase with single 
stem pink protea near sink. Remove any generic plants or decor. If floor 
visible and space allows, add narrow vintage-style runner in muted 
earth tones (faded rust, cream, sage). No books - wrong room."
```

### Issue: Exterior Minimally Transformed

**Current:** Added olive tree but lighting unchanged
**Fix:** Dramatic lighting transformation required

```
BEFORE PROMPT: "Add olive tree, warm lighting"

AFTER PROMPT: "DRAMATIC LIGHTING TRANSFORMATION: Convert harsh daylight to 
golden hour/dusk. Sky should be soft gradient from warm blue at top through 
golden to soft peach/pink at horizon - the 'magic hour' sky. All windows 
must show warm amber interior glow emanating from within - this is essential 
for editorial quality. Trees and landscaping should be lit with golden 
side-light, creating warm highlights and dimensional shadows. Add mature 
olive tree (6-8 ft) in large aged terracotta pot positioned near front entry. 
Lawn should appear healthier with warmer green/golden tones (not fake, just 
enhanced). Minimize power lines if visible. Overall effect should be 'Dwell 
magazine cover at sunset.'"
```

---

## CONSISTENCY FRAMEWORK

To ensure all images from a property feel like one cohesive editorial shoot:

### Pre-Processing Checklist
Before generating any images for a property:
1. Define the specific furniture pieces to use across all rooms
2. Define the 2-3 signature elements that will appear throughout
3. Define the exact lighting quality (all golden hour)
4. Define the color grade (warm, cream whites, amber shadows)

### Consistency Token
Add this to EVERY prompt for a property:

```
CONSISTENCY REQUIREMENTS:
- Color temperature: 2700K warm golden throughout
- Time of day: Golden hour (1 hour before sunset)
- Signature elements: Pink protea, olive tree, cream ceramics
- Wood tones: Natural oak, walnut - warm honey finish
- Metal accents: Brass/bronze only, no chrome/silver
- White point: Cream/ivory, never stark cool white
- Shadow tone: Warm brown/amber, never gray/blue
```

---

## TESTING CHECKLIST

After implementation, verify EVERY output against this checklist:

### Lighting (Most Important)
- [ ] Light feels like "golden hour" not midday
- [ ] Visible light rays or glow through windows
- [ ] Shadows are dimensional, warm-toned, not flat
- [ ] Interior has "lit from within" quality
- [ ] No cool/blue tones anywhere

### Staging
- [ ] Furniture matches specific style (bouclé sofas, wishbone chairs, etc.)
- [ ] 2-3 signature elements present (protea, olive tree, ceramics)
- [ ] Accessories are minimal but intentional
- [ ] Nothing feels "catalog" or generic

### Color Grading
- [ ] Overall warm/golden tone
- [ ] Whites are cream, not stark
- [ ] Wood tones are rich and warm
- [ ] Shadows have amber tone, not gray

### Consistency (Multi-Image)
- [ ] All images feel like same "photoshoot"
- [ ] Same lighting quality across all
- [ ] Same color temperature throughout
- [ ] Same accessory vocabulary

### The "Magazine Cover" Test
- [ ] Would this genuinely look at home on Architectural Digest cover?
- [ ] Does it have editorial drama and quality?
- [ ] Does it feel curated, not staged?
