"""
Stager Agent - Main FastAPI Application

Virtual staging backend for real estate photos.
Receives photos via Airtable webhook, processes with Gemini AI,
and delivers staged images to clients.
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from config import get_settings
from models import (
    AirtableWebhookPayload,
    JobResponse,
    HealthResponse,
    JobStatus
)
from job_manager import JobManager
from stager_planner import StagerPlanner
from stager_runner import StagerRunner
from stager_delivery import StagerDelivery
from airtable_client import AirtableClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Stager Agent...")
    
    # Initialize job directory
    settings = get_settings()
    Path(settings.BASE_JOBS_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Jobs directory: {settings.BASE_JOBS_DIR}")
    
    yield
    
    logger.info("Shutting down Stager Agent...")


# Create FastAPI app
app = FastAPI(
    title="Stager Agent",
    description="Virtual staging backend for real estate photos",
    version="1.0.0",
    lifespan=lifespan
)


# Initialize shared components
job_manager = JobManager()
stager_planner = StagerPlanner(job_manager=job_manager)
stager_runner = StagerRunner(job_manager=job_manager)
stager_delivery = StagerDelivery(job_manager=job_manager)
airtable_client = AirtableClient()


# ============================================================================
# Exception handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# ============================================================================
# Health check endpoint
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns basic service status and job statistics.
    """
    jobs = job_manager.list_jobs()
    
    last_updated = None
    if jobs:
        last_updated = jobs[0].get("updated_at")
    
    return HealthResponse(
        status="ok",
        total_jobs=len(jobs),
        last_job_updated=last_updated
    )


# ============================================================================
# Main webhook endpoint
# ============================================================================

async def process_staging_job(job_id: str, record_id: str, photos: list):
    """
    Background task to process a staging job.

    This runs asynchronously after the webhook returns.
    """
    try:
        # Update Airtable: In progress
        await airtable_client.mark_in_progress(record_id)

        # Step 2: Download images
        logger.info(f"[{job_id}] Step 2: Downloading images...")
        downloaded = await job_manager.download_images(job_id, photos)
        logger.info(f"[{job_id}] Downloaded {len(downloaded)} images")

        # Update status
        job_manager.update_order_status(job_id, JobStatus.INGESTED)

        # Step 3: Plan (analyze with Gemini)
        logger.info(f"[{job_id}] Step 3: Analyzing images with Gemini...")
        plan = await stager_planner.create_plan_for_job(job_id)
        logger.info(f"[{job_id}] Created plan with {len(plan.images)} images")

        # Step 4: Stage (generate with Nano Banana)
        logger.info(f"[{job_id}] Step 4: Generating staged images...")
        plan = await stager_runner.run_staging_for_job(job_id)

        # Count staged images
        staged_count = sum(1 for img in plan.images if img.status == "staged")
        total_count = len(plan.images)
        logger.info(f"[{job_id}] Staged {staged_count}/{total_count} images")

        # Check if any images failed
        if staged_count == 0:
            raise ValueError(f"All {total_count} images failed to stage")

        # Step 5: Package and deliver
        logger.info(f"[{job_id}] Step 5: Packaging and delivering...")
        stager_delivery.package_and_send(job_id)

        # Update Airtable based on success
        if staged_count < total_count:
            # Partial success - some images failed
            failed_count = total_count - staged_count
            await airtable_client.update_status(
                record_id,
                "ERROR",
                {"ERROR": f"Only {staged_count}/{total_count} photos processed. {failed_count} failed."}
            )
            logger.warning(f"[{job_id}] Partial success: {staged_count}/{total_count} images")
        else:
            # Full success
            await airtable_client.mark_done(record_id)
            logger.info(f"[{job_id}] Job completed successfully!")

    except Exception as e:
        logger.exception(f"[{job_id}] Job processing failed: {e}")
        job_manager.update_order_status(
            job_id,
            JobStatus.FAILED,
            error_message=str(e)
        )
        # Update Airtable: ERROR
        await airtable_client.mark_error(record_id, str(e))


@app.post(
    "/api/stager/airtable/webhook",
    response_model=JobResponse,
    tags=["Staging"]
)
async def airtable_webhook(
    payload: AirtableWebhookPayload,
    background_tasks: BackgroundTasks
):
    """
    Receive Airtable webhook and queue staging job.

    This endpoint returns immediately after creating the job,
    then processes in the background:
    1. Creates a job from the webhook payload (sync)
    2. Downloads images from Airtable (background)
    3. Analyzes images with Gemini (background)
    4. Generates staged images with Nano Banana (background)
    5. Packages and emails results to client (background)

    Check job status at GET /api/stager/jobs/{job_id}
    """
    logger.info(f"Received webhook for Airtable record: {payload.record_id}")

    # Validate required fields
    if not payload.fields.Photos:
        raise HTTPException(
            status_code=400,
            detail="No photos provided in webhook payload"
        )

    if not payload.fields.Email:
        raise HTTPException(
            status_code=400,
            detail="No email provided in webhook payload"
        )

    try:
        # Step 1: Create job (synchronous - fast)
        logger.info("Step 1: Creating job...")
        job_id = job_manager.create_job_from_webhook(payload)

        # Queue the rest of the processing in the background
        background_tasks.add_task(
            process_staging_job,
            job_id,
            payload.record_id,
            payload.fields.Photos
        )

        logger.info(f"Job {job_id} queued for background processing")

        # Return immediately
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Job queued. Processing {len(payload.fields.Photos)} images. Check status at /api/stager/jobs/{job_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create job: {str(e)}"
        )


# ============================================================================
# Job status endpoint
# ============================================================================

@app.get("/api/stager/jobs/{job_id}", response_model=JobResponse, tags=["Staging"])
async def get_job_status(job_id: str):
    """
    Get status of a staging job.
    
    Args:
        job_id: Job identifier
    """
    if not job_manager.job_exists(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    order = job_manager.load_order(job_id)
    
    return JobResponse(
        job_id=job_id,
        status=order.status,
        message=order.error_message
    )


# ============================================================================
# List jobs endpoint
# ============================================================================

@app.get("/api/stager/jobs", tags=["Staging"])
async def list_jobs(limit: int = 20):
    """
    List recent staging jobs.
    
    Args:
        limit: Maximum number of jobs to return
    """
    jobs = job_manager.list_jobs()
    return {"jobs": jobs[:limit], "total": len(jobs)}


# ============================================================================
# Retry staging endpoint
# ============================================================================

@app.post("/api/stager/jobs/{job_id}/retry", response_model=JobResponse, tags=["Staging"])
async def retry_job(job_id: str, stage: Optional[str] = None):
    """
    Retry a failed or incomplete job.
    
    Args:
        job_id: Job identifier
        stage: Specific stage to retry ("plan", "stage", "deliver")
    """
    if not job_manager.job_exists(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    order = job_manager.load_order(job_id)
    
    try:
        if stage == "plan" or order.status == JobStatus.INGESTED:
            plan = await stager_planner.create_plan_for_job(job_id)
            return JobResponse(job_id=job_id, status=JobStatus.PLANNED, message="Planning complete")
        
        elif stage == "stage" or order.status == JobStatus.PLANNED:
            plan = await stager_runner.run_staging_for_job(job_id)
            return JobResponse(job_id=job_id, status=JobStatus.STAGED, message="Staging complete")
        
        elif stage == "deliver" or order.status == JobStatus.STAGED:
            stager_delivery.package_and_send(job_id)
            return JobResponse(job_id=job_id, status=JobStatus.DELIVERED, message="Delivery complete")
        
        else:
            return JobResponse(
                job_id=job_id,
                status=order.status,
                message=f"Job is in status {order.status.value}, no retry needed"
            )
            
    except Exception as e:
        logger.exception(f"Retry failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Retry failed: {str(e)}")


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
