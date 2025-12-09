"""
Stager Planner - Coordinates image analysis and plan creation.
Acts as the orchestration layer between JobManager and GeminiPlannerClient.
"""

import logging
from pathlib import Path
from typing import Optional

from config import get_settings
from models import Order, Plan, JobStatus
from job_manager import JobManager
from gemini_client import GeminiPlannerClient

logger = logging.getLogger(__name__)


class StagerPlanner:
    """
    Orchestrates the planning phase of virtual staging.
    
    Responsibilities:
    - Load job data from JobManager
    - Invoke GeminiPlannerClient for image analysis
    - Save the resulting plan
    - Update job status
    """
    
    def __init__(
        self,
        job_manager: Optional[JobManager] = None,
        gemini_client: Optional[GeminiPlannerClient] = None
    ):
        """
        Initialize StagerPlanner.
        
        Args:
            job_manager: JobManager instance. Creates new if not provided.
            gemini_client: GeminiPlannerClient instance. Creates new if not provided.
        """
        self.job_manager = job_manager or JobManager()
        self.gemini_client = gemini_client or GeminiPlannerClient()
        
        logger.info("StagerPlanner initialized")
    
    async def create_plan_for_job(self, job_id: str) -> Plan:
        """
        Create a staging plan for a job by analyzing all raw images.
        
        This is the main entry point for the planning phase.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Created Plan object
            
        Raises:
            ValueError: If job doesn't exist or is already complete
            Exception: If planning fails
        """
        # Check if job is already complete
        if self.job_manager.is_job_complete(job_id):
            logger.info(f"Job {job_id} is already complete, skipping planning")
            return self.job_manager.load_plan(job_id)
        
        # Check if plan already exists
        if self.job_manager.plan_exists(job_id):
            logger.info(f"Plan already exists for job {job_id}")
            plan = self.job_manager.load_plan(job_id)
            # Check if all images are planned
            all_planned = all(img.nano_prompt is not None for img in plan.images)
            if all_planned:
                logger.info("All images already planned")
                return plan
            logger.info("Some images need planning, continuing...")
        
        # Load order
        order = self.job_manager.load_order(job_id)
        
        # Update status to planning
        self.job_manager.update_order_status(job_id, JobStatus.PLANNING)
        
        # Get job directory
        settings = get_settings()
        job_dir = Path(settings.BASE_JOBS_DIR) / job_id
        
        # Get raw image paths
        image_paths = self.job_manager.get_raw_image_paths(job_id)
        
        if not image_paths:
            raise ValueError(f"No raw images found for job {job_id}")
        
        logger.info(f"Planning {len(image_paths)} images for job {job_id}")
        
        try:
            # Run analysis
            plan = await self.gemini_client.analyze_and_plan_images(
                job_id=job_id,
                job_dir=job_dir,
                order=order,
                image_paths=image_paths
            )
            
            # Save plan
            self.job_manager.save_plan(plan)
            
            # Update status
            self.job_manager.update_order_status(job_id, JobStatus.PLANNED)
            
            logger.info(f"Successfully created plan for job {job_id}")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create plan for job {job_id}: {e}")
            self.job_manager.update_order_status(
                job_id, 
                JobStatus.FAILED, 
                error_message=f"Planning failed: {str(e)}"
            )
            raise
    
    async def replan_failed_images(self, job_id: str) -> Plan:
        """
        Re-analyze images that failed during initial planning.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Updated Plan object
        """
        plan = self.job_manager.load_plan(job_id)
        order = self.job_manager.load_order(job_id)
        
        settings = get_settings()
        job_dir = Path(settings.BASE_JOBS_DIR) / job_id
        
        # Find failed images
        failed_images = [img for img in plan.images if img.status == "failed" and img.nano_prompt is None]
        
        if not failed_images:
            logger.info(f"No failed images to replan for job {job_id}")
            return plan
        
        logger.info(f"Replanning {len(failed_images)} failed images for job {job_id}")
        
        for img in failed_images:
            try:
                abs_path = job_dir / img.source_path
                result = await self.gemini_client.analyze_image(
                    image_path=abs_path,
                    style=order.style,
                    is_occupied=order.occupied
                )
                
                img.room_type = result.room_type
                img.is_occupied = result.is_occupied
                img.issues = result.issues
                img.nano_prompt = result.staging_prompt
                img.status = "planned"
                img.error_message = None
                
            except Exception as e:
                logger.error(f"Failed to replan {img.id}: {e}")
                img.error_message = str(e)
        
        self.job_manager.save_plan(plan)
        return plan
