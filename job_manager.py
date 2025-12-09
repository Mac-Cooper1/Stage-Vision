"""
Job Manager for Stager Agent.
Handles job folder creation, state management, and file operations.
"""

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
import httpx

from config import get_settings
from models import (
    Order, Plan, ImagePlan, ClientInfo,
    AirtableWebhookPayload, JobStatus, ImageStatus
)
from utils import generate_job_id, utc_now

logger = logging.getLogger(__name__)


class JobManager:
    """
    Manages job folders, state files, and image downloads.
    
    Each job gets a folder structure:
        stager_jobs/{job_id}/
            order.json      - Job metadata
            plan.json       - Per-image planning data
            raw/            - Original photos
            staged/         - Staged output images
            final/          - Zipped deliverables
            logs/           - Debug logs
            .done.lock      - Completion marker
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize JobManager.
        
        Args:
            base_dir: Base directory for jobs. Uses config default if not provided.
        """
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.BASE_JOBS_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JobManager initialized with base_dir: {self.base_dir}")
    
    def _get_job_dir(self, job_id: str) -> Path:
        """Get the directory path for a job."""
        return self.base_dir / job_id
    
    def job_exists(self, job_id: str) -> bool:
        """Check if a job folder exists."""
        return self._get_job_dir(job_id).exists()
    
    def is_job_complete(self, job_id: str) -> bool:
        """Check if job has .done.lock file."""
        return (self._get_job_dir(job_id) / ".done.lock").exists()
    
    def mark_job_complete(self, job_id: str) -> None:
        """Create .done.lock file to mark job as complete."""
        lock_path = self._get_job_dir(job_id) / ".done.lock"
        lock_path.touch()
        logger.info(f"Marked job {job_id} as complete")
    
    def create_job_from_webhook(self, payload: AirtableWebhookPayload) -> str:
        """
        Create a new job from Airtable webhook payload.
        
        Args:
            payload: Validated Airtable webhook payload
            
        Returns:
            Created job_id
            
        Raises:
            ValueError: If job creation fails
        """
        # Generate job ID
        short_uuid = uuid.uuid4().hex[:6]
        job_id = generate_job_id(payload.fields.Address, short_uuid)
        
        job_dir = self._get_job_dir(job_id)
        
        # Create directory structure
        (job_dir / "raw").mkdir(parents=True, exist_ok=True)
        (job_dir / "staged").mkdir(parents=True, exist_ok=True)
        (job_dir / "final").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Parse occupied status (Yes/No from Airtable dropdown)
        occupied = payload.fields.Occupied and payload.fields.Occupied.lower() in ("yes", "true", "1")

        # Create order
        order = Order(
            job_id=job_id,
            airtable_record_id=payload.record_id,
            client=ClientInfo(
                name=payload.fields.Name,
                email=payload.fields.Email
            ),
            address=payload.fields.Address,
            occupied=occupied,
            status=JobStatus.PENDING
        )
        
        # Save order.json
        self.save_order(order)
        
        logger.info(f"Created job {job_id} for {payload.fields.Address}")
        return job_id
    
    async def download_images(self, job_id: str, photos: list) -> list[str]:
        """
        Download images from Airtable URLs to raw/ folder.
        
        Args:
            job_id: Job identifier
            photos: List of AirtablePhoto objects
            
        Returns:
            List of local file paths
        """
        job_dir = self._get_job_dir(job_id)
        raw_dir = job_dir / "raw"
        downloaded = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i, photo in enumerate(photos):
                try:
                    response = await client.get(photo.url)
                    response.raise_for_status()
                    
                    # Use original filename or generate one
                    filename = photo.filename or f"photo_{i+1}.jpg"
                    # Sanitize filename
                    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                    
                    file_path = raw_dir / filename
                    file_path.write_bytes(response.content)
                    
                    downloaded.append(str(file_path.relative_to(job_dir)))
                    logger.info(f"Downloaded {filename} for job {job_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to download {photo.url}: {e}")
                    raise
        
        return downloaded
    
    def save_order(self, order: Order) -> None:
        """Save order to order.json."""
        order.updated_at = utc_now()
        job_dir = self._get_job_dir(order.job_id)
        order_path = job_dir / "order.json"
        
        with open(order_path, "w") as f:
            json.dump(order.model_dump(mode="json"), f, indent=2, default=str)
        
        logger.debug(f"Saved order.json for {order.job_id}")
    
    def load_order(self, job_id: str) -> Order:
        """Load order from order.json."""
        job_dir = self._get_job_dir(job_id)
        order_path = job_dir / "order.json"
        
        with open(order_path, "r") as f:
            data = json.load(f)
        
        return Order(**data)
    
    def save_plan(self, plan: Plan) -> None:
        """Save plan to plan.json."""
        plan.updated_at = utc_now()
        job_dir = self._get_job_dir(plan.job_id)
        plan_path = job_dir / "plan.json"
        
        with open(plan_path, "w") as f:
            json.dump(plan.model_dump(mode="json"), f, indent=2, default=str)
        
        logger.debug(f"Saved plan.json for {plan.job_id}")
    
    def load_plan(self, job_id: str) -> Plan:
        """Load plan from plan.json."""
        job_dir = self._get_job_dir(job_id)
        plan_path = job_dir / "plan.json"
        
        with open(plan_path, "r") as f:
            data = json.load(f)
        
        return Plan(**data)
    
    def plan_exists(self, job_id: str) -> bool:
        """Check if plan.json exists for a job."""
        job_dir = self._get_job_dir(job_id)
        return (job_dir / "plan.json").exists()
    
    def update_order_status(self, job_id: str, status: JobStatus, error_message: Optional[str] = None) -> None:
        """Update order status."""
        order = self.load_order(job_id)
        order.status = status
        if error_message:
            order.error_message = error_message
        self.save_order(order)
    
    def get_raw_image_paths(self, job_id: str) -> list[str]:
        """Get list of raw image paths for a job."""
        job_dir = self._get_job_dir(job_id)
        raw_dir = job_dir / "raw"
        
        if not raw_dir.exists():
            return []
        
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        paths = []
        
        for f in sorted(raw_dir.iterdir()):
            if f.suffix.lower() in image_extensions:
                paths.append(str(f.relative_to(job_dir)))
        
        return paths
    
    def get_absolute_path(self, job_id: str, relative_path: str) -> Path:
        """Convert relative path to absolute path within job directory."""
        return self._get_job_dir(job_id) / relative_path
    
    def list_jobs(self) -> list[dict]:
        """List all jobs with basic info."""
        jobs = []
        
        for job_dir in self.base_dir.iterdir():
            if not job_dir.is_dir():
                continue
            
            order_path = job_dir / "order.json"
            if order_path.exists():
                try:
                    with open(order_path) as f:
                        order_data = json.load(f)
                    jobs.append({
                        "job_id": order_data.get("job_id"),
                        "status": order_data.get("status"),
                        "created_at": order_data.get("created_at"),
                        "updated_at": order_data.get("updated_at"),
                    })
                except Exception as e:
                    logger.error(f"Failed to read order.json in {job_dir}: {e}")
        
        return sorted(jobs, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    def cleanup_job(self, job_id: str) -> None:
        """Remove job folder entirely."""
        job_dir = self._get_job_dir(job_id)
        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info(f"Cleaned up job {job_id}")
