"""
Pydantic models for Stager Agent.
Defines schemas for webhook payloads, job state, and internal data structures.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class JobStatus(str, Enum):
    """Status values for job processing."""
    PENDING = "pending"
    INGESTED = "ingested"
    PLANNING = "planning"
    PLANNED = "planned"
    STAGING = "staging"
    STAGED = "staged"
    PACKAGING = "packaging"
    DELIVERED = "delivered"
    FAILED = "failed"


class ImageStatus(str, Enum):
    """Status values for individual image processing."""
    PENDING = "pending"
    PLANNED = "planned"
    STAGING = "staging"
    STAGED = "staged"
    NEEDS_REGEN = "needs_regen"
    FAILED = "failed"


class StylePreference(str, Enum):
    """Available style preferences for staging."""
    MODERN_NEUTRAL = "modern_neutral"
    MODERN_MINIMALIST = "modern_minimalist"
    COASTAL = "coastal"
    FARMHOUSE = "farmhouse"
    TRADITIONAL = "traditional"
    SCANDINAVIAN = "scandinavian"
    MID_CENTURY = "mid_century"


# ============================================================================
# Airtable Webhook Models
# ============================================================================

class AirtablePhoto(BaseModel):
    """Single photo attachment from Airtable."""
    url: str
    filename: str
    
    class Config:
        extra = "ignore"


class AirtableFields(BaseModel):
    """Fields from Airtable record."""
    Name: str = Field(..., description="Client name")
    Email: EmailStr = Field(..., description="Client email")
    Address: str = Field(..., description="Property address")
    Occupied: Optional[str] = Field(default="No", description="Is property occupied (Yes/No)")
    Photos: list[AirtablePhoto] = Field(..., description="Uploaded photos")

    class Config:
        extra = "ignore"


class AirtableWebhookPayload(BaseModel):
    """Webhook payload from Airtable automation."""
    record_id: str = Field(..., description="Airtable record ID")
    fields: AirtableFields
    
    class Config:
        extra = "ignore"


# ============================================================================
# Client Info
# ============================================================================

class ClientInfo(BaseModel):
    """Client information extracted from webhook."""
    name: str
    email: EmailStr


# ============================================================================
# Order Schema (order.json)
# ============================================================================

def _format_datetime(v: datetime) -> str:
    """Format datetime as ISO8601 with Z suffix, handling both naive and aware datetimes."""
    if v is None:
        return None
    # Remove timezone info to avoid +00:00Z issue, then add Z
    if v.tzinfo is not None:
        v = v.replace(tzinfo=None)
    return v.isoformat() + "Z"


class Order(BaseModel):
    """
    Order represents a single staging job.
    Stored as order.json in each job folder.
    """
    job_id: str = Field(..., description="Unique job identifier")
    airtable_record_id: str = Field(..., description="Original Airtable record ID")
    client: ClientInfo
    address: str = Field(..., description="Property address")
    source: str = Field(default="fsbo_airtable")
    occupied: bool = Field(default=False, description="Is property occupied (affects cleanup vs empty room)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: JobStatus = Field(default=JobStatus.PENDING)
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

    class Config:
        json_encoders = {
            datetime: _format_datetime
        }


# ============================================================================
# Plan Schema (plan.json)
# ============================================================================

class ImagePlan(BaseModel):
    """Plan for a single image in the staging job."""
    id: str = Field(..., description="Image identifier (e.g., img_1)")
    source_path: str = Field(..., description="Path to source image in raw/")
    room_type: Optional[str] = Field(default=None, description="Detected room type")
    is_occupied: bool = Field(default=False, description="Whether room has existing furniture")
    issues: list[str] = Field(default_factory=list, description="Detected issues (clutter, lighting, etc.)")
    nano_prompt: Optional[str] = Field(default=None, description="Conservative cleanup prompt for Nano Banana")
    status: ImageStatus = Field(default=ImageStatus.PENDING)
    output_path: Optional[str] = Field(default=None, description="Path to staged output")
    error_message: Optional[str] = Field(default=None)

    class Config:
        json_encoders = {
            datetime: _format_datetime
        }


class Plan(BaseModel):
    """
    Plan for the entire staging job.
    Stored as plan.json in each job folder.
    """
    job_id: str
    images: list[ImagePlan] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: _format_datetime
        }


# ============================================================================
# Gemini Response Models
# ============================================================================

class GeminiAnalysisResult(BaseModel):
    """Expected JSON response from Gemini analysis."""
    room_type: str = Field(..., description="Type of room (living_room, bedroom, etc.)")
    is_occupied: bool = Field(..., description="Whether room has existing furniture")
    issues: list[str] = Field(default_factory=list, description="Issues to address")
    suggested_style: str = Field(default="modern_neutral")
    staging_prompt: str = Field(..., description="Detailed prompt for staging")


# ============================================================================
# API Response Models
# ============================================================================

class JobResponse(BaseModel):
    """Response for job creation/status endpoints."""
    job_id: str
    status: JobStatus
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = "ok"
    total_jobs: int = 0
    last_job_updated: Optional[str] = None
