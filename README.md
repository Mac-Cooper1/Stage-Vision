# Stage Vision - AI Virtual Staging for Real Estate

A professional virtual staging backend for real estate photos. Uses AI to transform vacant and occupied properties into beautifully staged, move-in ready spaces while maintaining MLS compliance.

## Core Philosophy: "Stage It Beautifully, Keep It Honest"

This system transforms empty rooms into stunning, professionally staged spaces while maintaining structural honesty about the property's actual condition.

### What We Do
- **Stage vacant rooms** with complete furniture sets, rugs, plants, art, and decor
- **Declutter occupied rooms** and enhance with coordinating decor
- **Apply style preferences** (modern neutral, coastal, farmhouse, etc.)
- **Enhance photo quality** with professional lighting, color correction, and leveling
- Create photorealistic results that look like professional staging photography

### What We Don't Do (Structural Honesty)
- Hide damage, cracks, stains, or wear on walls, floors, or fixtures
- Change room dimensions or make spaces appear larger
- Invent parts of rooms that weren't photographed
- Alter architecture, flooring, or built-in features
- Add fake landscaping or change grass color for exteriors

### Why This Matters
Buyers expect virtual staging - it helps them visualize potential. But MLS compliance requires honesty about the property's condition. Our approach gives sellers the marketing advantage of staging while maintaining legal compliance.

---

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Airtable      │────▶│   FastAPI       │────▶│   Gemini Pro    │
│   (Webhook)     │     │   Backend       │     │   (Analysis)    │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                               │
                               ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Email SMTP    │◀────│   Job Manager   │────▶│   Nano Banana   │
│   (Delivery)    │     │   (State)       │     │   (Image Gen)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Processing Flow

1. **Webhook Received** - Airtable automation triggers on form submission
2. **Job Created** - Unique job ID generated, folder structure created
3. **Images Downloaded** - Photos fetched from Airtable attachment URLs
4. **Analysis (Gemini Pro)** - Each photo analyzed for room type, occupancy status, and a detailed staging prompt is generated based on style preference
5. **Image Generation (Nano Banana)** - Gemini's image model applies the virtual staging
6. **Packaging** - Staged photos zipped for delivery
7. **Email Delivery** - Results sent to customer with "Virtually Staged" labels
8. **Airtable Update** - Status updated (Done/ERROR) for tracking

---

## Style Preferences

Stage Vision supports multiple design styles to match different property aesthetics:

| Style | Description |
|-------|-------------|
| **Modern Neutral** | Clean lines, warm grays, soft whites, natural wood tones |
| **Modern Minimalist** | Ultra-clean, stark whites, black accents, selective furniture |
| **Coastal** | Light blues, sandy neutrals, white-washed woods, natural textures |
| **Farmhouse** | Rustic wood, white shiplap aesthetic, cozy textiles |
| **Traditional** | Rich woods, elegant fabrics, timeless silhouettes |
| **Scandinavian** | Bright whites, light woods, functional design, hygge comfort |
| **Mid-Century** | Iconic shapes, warm woods, bold accent colors |

---

## File Structure

```
stage-vision/
├── main.py              # FastAPI app, endpoints, background task orchestration
├── config.py            # Pydantic settings, environment variable management
├── models.py            # All Pydantic schemas (Order, Plan, ImagePlan, StylePreference, etc.)
├── job_manager.py       # Job folder creation, state persistence, image downloads
├── gemini_client.py     # Gemini Pro analysis + staging prompt generation (THE BRAIN)
├── nano_client.py       # Gemini image generation with retry logic
├── stager_planner.py    # Orchestrates the analysis phase
├── stager_runner.py     # Orchestrates the image generation phase
├── stager_delivery.py   # Packaging, labeling, email delivery
├── airtable_client.py   # Airtable API for status updates
├── image_utils.py       # PIL utilities, "Virtually Staged" label overlay
├── utils/
│   ├── slugify.py       # URL-safe job ID generation
│   └── time_utils.py    # Timestamp formatting
├── requirements.txt
├── .env.example
└── README.md
```

---

## Detailed Component Breakdown

### `gemini_client.py` - The Brain

This is the most critical file. It contains the `_build_analysis_prompt()` method that generates detailed staging prompts based on:

- **Room type** (bedroom, living room, kitchen, etc.)
- **Occupancy status** (vacant vs occupied)
- **Style preference** (modern neutral, coastal, farmhouse, etc.)

**Key Design Decisions:**

#### 1. Style-Aware Staging Prompts
Each style has a complete guide including:
- Color palette
- Furniture styles
- Decor items
- Rug patterns
- Lighting fixtures

The prompt system generates specific, detailed instructions like "Add a light gray sectional sofa with clean lines facing the fireplace" rather than generic "add furniture."

#### 2. Room-Type Patterns
Each room type has specific staging guidance:
- **Bedrooms**: Bed with headboard, nightstands, lamps, area rug, art above bed
- **Living Rooms**: Sofa, coffee table, accent chairs, area rug, plants, art
- **Dining Rooms**: Table with chairs, rug underneath, centerpiece, lighting
- **Kitchens**: Bar stools (if island), coordinating accessories, minimal decor
- **Bathrooms**: Rolled towels, small plants, spa-like accessories

#### 3. Camera Angle Rules
We discovered that Gemini's image model would often "improve" photos by changing the camera angle dramatically. Our rules:

**ALLOWED:**
- Level the photo (fix tilt, make verticals vertical)
- Adjust camera height and pitch (up/down viewing angle)
- Minor framing adjustments

**FORBIDDEN:**
- Horizontal camera movement (sliding to different wall/corner)
- Yaw rotation (swinging view left/right to reveal new surfaces)
- Widening field of view (making room look bigger)
- Inventing unseen parts of the room

#### 4. Structural Honesty (Bidirectional)
- Don't hide damage that exists
- Don't invent damage that doesn't exist
- Furniture can be placed in front of damage but can't magically repair it

### `nano_client.py` - Image Generation

Handles the actual image transformation using Gemini's image generation model.

**Key Features:**

#### Dynamic Aspect Ratio Matching
The `choose_gemini_image_config()` function analyzes input image dimensions and selects the closest supported aspect ratio. This prevents the model from cropping or stretching images.

Supported ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9

#### Retry Logic with Smart Fallback
The image generation sometimes fails. Our strategy:

1. **6 retry attempts** with linear backoff
2. **First attempt**: Uses the full detailed prompt
3. **Subsequent attempts**: Uses simplified fallback prompt that:
   - Detects if room is vacant (needs staging) or occupied (needs declutter)
   - Extracts style preference from original prompt
   - Provides room-appropriate furniture suggestions

---

## Staging Guidelines

### For Vacant Rooms

The system adds complete furniture sets appropriate to each room:

| Room Type | Typical Staging |
|-----------|----------------|
| Bedroom | Bed with headboard, nightstands, lamps, area rug, art, plant |
| Living Room | Sofa, coffee table, accent chairs, area rug, floor lamp, plants, art, throw pillows |
| Dining Room | Table with chairs, area rug, centerpiece, pendant lighting effect |
| Office | Desk, chair, bookshelf, task lamp, plant, art |
| Kitchen | Bar stools (if island), fruit bowl, herb plant, coordinating accessories |
| Bathroom | Rolled towels, small plant, tasteful accessories |

### For Occupied Rooms

The system:
1. Removes visible clutter and personal items
2. Keeps existing furniture in place
3. May add coordinating decor (pillows, plants, throws)
4. Straightens beds and tidies soft furnishings
5. Enhances photo quality

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Google API key with Gemini access (both vision and image generation)
- SMTP credentials for email delivery
- (Optional) Airtable API key for status updates

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd stage-vision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables (.env)

```bash
# =============================================================================
# Stage Vision - Environment Configuration
# =============================================================================

# Google API (Required)
# Get your key at: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# Gemini Models (Optional - defaults shown)
GEMINI_VISION_MODEL=gemini-2.5-pro-preview-06-05
GEMINI_IMAGE_MODEL=gemini-2.5-flash-image

# Airtable Integration (Optional - enables status updates)
AIRTABLE_API_KEY=your_airtable_pat_here
AIRTABLE_BASE_ID=appXXXXXXXXXXXXXX
AIRTABLE_TABLE_NAME=Orders

# Email Delivery (Required for sending results)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here
EMAIL_FROM=Stage Vision <your_email@gmail.com>

# Processing Settings (Optional - defaults shown)
BASE_JOBS_DIR=./stager_jobs
MAX_RETRIES=6
REQUEST_TIMEOUT=120
```

### Running the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --port 8000

# Or run directly
python main.py
```

The API will be available at `http://localhost:8000`

---

## API Endpoints

### POST `/api/stager/airtable/webhook`
Receives Airtable webhook payload and queues the staging job.

**Request:**
```json
{
  "record_id": "recXXXXXXXX",
  "fields": {
    "Name": "Jane Seller",
    "Email": "jane@example.com",
    "Address": "123 Main St, Boston, MA 02116",
    "Occupied": "No",
    "Style": "coastal",
    "Photos": [
      {"url": "https://...", "filename": "photo1.jpg"}
    ]
  }
}
```

**Response:**
```json
{
  "job_id": "123-main-st-boston-ma-14f9c8",
  "status": "pending",
  "message": "Job queued. Processing 5 images..."
}
```

### GET `/api/stager/jobs/{job_id}`
Get status of a specific job.

### GET `/api/stager/jobs`
List all jobs with status.

### POST `/api/stager/jobs/{job_id}/retry`
Retry a failed job from a specific stage ("plan", "stage", or "deliver").

### GET `/health`
Health check endpoint with job statistics.

---

## Prompt Engineering Details

The system uses a two-model approach:

### Model 1: Gemini Pro (Analysis)
Analyzes each photo and generates a detailed staging prompt. The analysis prompt includes:

- 9 Core Principles (structure preservation, realistic staging, style adherence, etc.)
- Style-specific furniture and decor guides for all 7 supported styles
- Room-type specific staging patterns
- 3 Gold Examples showing ideal prompt structure for different scenarios
- Explicit ALLOWED/FORBIDDEN lists for camera adjustments

### Model 2: Gemini Image (Generation)
Takes the staging prompt + original image and generates the staged version.

Key settings:
- `responseModalities: ["TEXT", "IMAGE"]`
- `aspectRatio` - Dynamically matched to input image
- `imageSize` - Auto-selected (1K, 2K, or 4K based on input)

---

## Troubleshooting

### "No image data in response" / `finishReason: OTHER`
This is a mystery failure from Google's API. The system automatically retries with a simplified staging-aware prompt. If it persists, the image quality may be too degraded.

Common triggers:
- Very dark/underexposed images
- Heavy noise/grain
- Extremely low resolution
- Severe blur

### Furniture looks unrealistic or wrong scale
The system relies on the AI's understanding of room dimensions. If furniture appears off:
1. Check the generated prompt in `plan.json`
2. Ensure the original photo clearly shows room proportions
3. Consider if a different camera angle might help

### Staging doesn't match style preference
Verify the style preference is being passed correctly through the pipeline. Check logs for `Style: {style_preference}` during analysis.

---

## Design Decisions Log

1. **Why virtual staging instead of just cleanup?**
   Virtual staging dramatically increases buyer interest and perceived value. Empty rooms are hard to visualize; staged rooms feel like homes.

2. **Why maintain structural honesty?**
   MLS compliance and legal liability. Hiding defects can result in lawsuits. Our approach stages beautifully while showing the real condition.

3. **Why two models instead of one?**
   Separation of concerns. Analysis needs reasoning (Pro), generation needs image capability (Flash Image).

4. **Why 7 style options?**
   Covers the most popular interior design styles in real estate. Each has distinct furniture and decor that appeals to different buyer demographics.

5. **Why style-specific furniture descriptions?**
   Generic "add furniture" prompts produce inconsistent results. Specific descriptions like "white slipcovered sofa" for coastal style produce cohesive, believable staging.

6. **Why dynamic aspect ratio?**
   Fixed ratios cause the model to crop or pad, leading to hallucinated content at edges.

7. **Why 6 retries with smart fallback?**
   Empirical testing showed ~15-20% of images fail initially. The smart fallback detects vacant vs occupied and applies appropriate simplified staging.

---

## License

MIT License - Feel free to use and modify for your own projects.

---

Built with FastAPI, Google Gemini AI, and Pillow.
