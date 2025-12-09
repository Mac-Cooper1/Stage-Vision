# Stage Vision - Virtual Staging for Real Estate

A conservative virtual staging backend for FSBO (For Sale By Owner) real estate photos. Uses AI to professionally enhance listing photos while maintaining MLS compliance and honesty.

## Core Philosophy: "Clean Up, Don't Lie"

This system is fundamentally different from typical virtual staging tools. We are a **"clean up this photo" tool**, NOT a **"lie about what this house is" tool**.

### What We Do
- Remove clutter, trash, and loose items
- Correct exposure, white balance, and color casts
- Level tilted photos and optimize camera angles
- Make photos look like they were taken by a professional real estate photographer

### What We Don't Do
- Add furniture to empty rooms (for vacant properties)
- Hide damage, cracks, stains, or wear
- Make rooms appear larger than they are
- Add fake plants, decor, or staging items
- Invent parts of the room that weren't photographed

### Why This Matters
MLS (Multiple Listing Service) has strict rules about misleading photos. Buyers have legal recourse if photos misrepresent a property. Our conservative approach protects sellers from liability while still dramatically improving photo quality.

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
4. **Analysis (Gemini Pro)** - Each photo analyzed for room type, issues, and a conservative cleanup prompt is generated
5. **Image Generation (Nano Banana)** - Gemini's image model applies the cleanup
6. **Packaging** - Staged photos zipped for delivery
7. **Email Delivery** - Results sent to customer with "Virtually Staged" labels
8. **Airtable Update** - Status updated (Done/ERROR) for tracking

---

## File Structure

```
stage-vision/
├── main.py              # FastAPI app, endpoints, background task orchestration
├── config.py            # Pydantic settings, environment variable management
├── models.py            # All Pydantic schemas (Order, Plan, ImagePlan, etc.)
├── job_manager.py       # Job folder creation, state persistence, image downloads
├── gemini_client.py     # Gemini Pro analysis + prompt generation (THE BRAIN)
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

This is the most critical file. It contains the `_build_analysis_prompt()` method that defines our entire philosophy through a carefully crafted system prompt.

**Key Design Decisions:**

#### 1. Conservative Prompt Generation
The prompt instructs Gemini to generate cleanup instructions that follow our philosophy. It uses a 4-paragraph structure:
- Paragraph 1: Context ("Using this uploaded [room type] photo...")
- Paragraph 2: What must stay (walls, windows, flooring, fixtures)
- Paragraph 3: What can change (loose clutter only)
- Paragraph 4: Camera adjustments and photo enhancement

#### 2. Camera Angle Rules (Critical)
We discovered that Gemini's image model would often "improve" photos by changing the camera angle dramatically, which could reveal hallucinated parts of the room. Our rules:

**ALLOWED:**
- Level the photo (fix tilt, make verticals vertical)
- Adjust camera height and pitch (up/down viewing angle)
- Minor framing adjustments

**FORBIDDEN:**
- Horizontal camera movement (sliding to different wall/corner)
- Yaw rotation (swinging view left/right to reveal new surfaces)
- Widening field of view (making room look bigger)
- Inventing unseen parts of the room

#### 3. Damage Honesty (Bidirectional)
- Don't hide damage that exists
- Don't invent damage that doesn't exist

This bidirectional rule prevents the AI from both improving AND degrading the property's apparent condition.

#### 4. No New Plants or Decor
AI image generators love to add plants and decorations. We explicitly forbid this because:
- It's staging that didn't exist
- It could misrepresent the property
- Vacant properties should look vacant

### `nano_client.py` - Image Generation

Handles the actual image transformation using Gemini's image generation model (`gemini-2.5-flash-image`, internally called "Nano Banana").

**Key Features:**

#### Dynamic Aspect Ratio Matching
The `choose_gemini_image_config()` function analyzes input image dimensions and selects the closest supported aspect ratio. This prevents the model from cropping or stretching images, which would cause hallucinations.

Supported ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9

#### Retry Logic with Fallback Prompts
The image generation sometimes fails with `finishReason: OTHER` (a mystery failure from Google's API). Our strategy:

1. **6 retry attempts** with linear backoff (1s, 2s, 3s, 4s, 5s between attempts)
2. **First attempt**: Uses the full detailed prompt from Gemini analysis
3. **Subsequent attempts**: Uses a simplified fallback prompt that focuses on core cleanup without extensive constraints

The fallback prompt is built by `_build_fallback_prompt()` and is much simpler:
```
Clean up and enhance this {room_type} photo for a real estate listing.
Keep the same layout, walls, flooring, and all major fixtures exactly as they are.
Remove any visible clutter, trash, or loose items to make the space look tidy.
Apply professional photo enhancement: correct exposure, fix white balance, reduce haze...
Level the photo so vertical lines are truly vertical.
Do not add furniture, plants, or decor. Do not hide any damage. Keep it honest and realistic.
```

### `airtable_client.py` - Status Tracking

Updates the Airtable record as the job progresses:
- `mark_in_progress()` - When processing starts
- `mark_done()` - When all photos processed successfully
- `mark_error()` - When failures occur (writes error message to ERROR field)

Note: The field name is `ERROR` (all caps) in Airtable.

### `config.py` - Configuration

Uses `pydantic-settings` for type-safe environment variable management with sensible defaults.

Key settings:
- `MAX_RETRIES`: 6 (increased from default 3 for flaky API)
- `REQUEST_TIMEOUT`: 120 seconds (image generation is slow)

---

## Job Folder Structure

Each job creates a folder under `stager_jobs/`:

```
stager_jobs/
└── 123-main-st-boston-ma-14f9c8/
    ├── order.json          # Job metadata, client info, status
    ├── plan.json           # Per-image analysis, prompts, results
    ├── raw/                # Original photos from Airtable
    │   ├── photo_1.jpg
    │   └── photo_2.jpg
    ├── staged/             # Generated staged images
    │   ├── img_1_staged_final.jpg
    │   └── img_2_staged_final.jpg
    └── final/              # Delivery packages
        └── staged_photos.zip
```

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
    "Occupied": "Yes",
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
Analyzes each photo and generates a detailed, conservative cleanup prompt. The analysis prompt (`_build_analysis_prompt()`) is ~3000 tokens and includes:

- 8 Core Principles (anchoring, structure lock, camera rules, damage honesty, etc.)
- Room-type specific patterns (kitchen, bathroom, bedroom, exterior, etc.)
- 3 Gold Examples showing ideal prompt structure
- Explicit ALLOWED/FORBIDDEN lists for camera adjustments

### Model 2: Gemini Image (Generation)
Takes the cleanup prompt + original image and generates the enhanced version.

Key settings:
- `responseModalities: ["TEXT", "IMAGE"]` - Model can explain and show
- `aspectRatio` - Dynamically matched to input image
- `imageSize` - Auto-selected (1K, 2K, or 4K based on input)

---

## Troubleshooting

### "No image data in response" / `finishReason: OTHER`
This is a mystery failure from Google's API. The system automatically retries with a simplified prompt. If it persists across all 6 attempts, the image quality may be too degraded for the model to process.

Common triggers:
- Very dark/underexposed images
- Heavy noise/grain
- Extremely low resolution
- Severe blur

### Airtable 422 Error: Unknown field name
Ensure field names match exactly (case-sensitive). The status field is `Status`, the error field is `ERROR`.

### Images look too different from original
The camera angle rules may not be strong enough for certain images. Check the generated prompt in `plan.json` to see what instructions were given.

### Empty email delivery
Check `stager_delivery.py` logs. SMTP credentials may be incorrect or the zip file failed to generate.

---

## Design Decisions Log

1. **Why conservative cleanup instead of full virtual staging?**
   MLS compliance and legal liability. Full staging can misrepresent properties.

2. **Why two models instead of one?**
   Separation of concerns. Analysis needs reasoning (Pro), generation needs image capability (Flash Image).

3. **Why dynamic aspect ratio?**
   Fixed ratios cause the model to crop or pad, leading to hallucinated content at edges.

4. **Why 6 retries?**
   Empirical testing showed ~15-20% of images fail on first attempt but succeed on retry with simplified prompt.

5. **Why linear backoff instead of exponential?**
   User preference for faster retries. Total wait is 15s (1+2+3+4+5) instead of 62s exponential.

6. **Why no furniture in vacant rooms?**
   Adding furniture that doesn't exist is deceptive. Vacant should look vacant.

7. **Why explicit "don't invent damage" rule?**
   AI models can sometimes add artifacts that look like damage. This ensures the property isn't made to look worse than reality.

---

## License

MIT License - Feel free to use and modify for your own projects.

---

Built with FastAPI, Google Gemini AI, and Pillow.
