"""
Configuration module for Stager Agent.
Uses pydantic-settings for environment variable management.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Base directories
    BASE_JOBS_DIR: str = Field(default="./stager_jobs", description="Base directory for job folders")
    
    # Google API Configuration
    GOOGLE_API_KEY: str = Field(..., description="Google API key for Gemini")
    GEMINI_API_BASE_URL: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Base URL for Gemini API"
    )
    
    # Model IDs
    GEMINI_VISION_MODEL: str = Field(
        default="gemini-2.5-pro-preview-06-05",
        description="Model for image analysis and planning"
    )
    GEMINI_IMAGE_MODEL: str = Field(
        default="gemini-2.5-flash-image",
        description="Model for image generation/editing (Nano Banana)"
    )
    
    # Airtable Configuration
    AIRTABLE_API_KEY: str = Field(default="", description="Airtable Personal Access Token")
    AIRTABLE_BASE_ID: str = Field(default="", description="Airtable Base ID (starts with 'app')")
    AIRTABLE_TABLE_NAME: str = Field(default="Orders", description="Airtable table name")

    # SMTP Email Configuration
    SMTP_HOST: str = Field(default="smtp.gmail.com", description="SMTP server host")
    SMTP_PORT: int = Field(default=587, description="SMTP server port")
    SMTP_USERNAME: str = Field(default="", description="SMTP username")
    SMTP_PASSWORD: str = Field(default="", description="SMTP password")
    EMAIL_FROM: str = Field(default="Stager Agent <no-reply@stageragent.com>", description="From email address")
    
    # Processing settings
    MAX_RETRIES: int = Field(default=6, description="Max retries for API calls")
    REQUEST_TIMEOUT: int = Field(default=120, description="Request timeout in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
