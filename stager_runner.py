"""
Stager Runner - Executes the virtual staging process.
Calls Nano Banana to generate staged images and applies "Virtually Staged" labels.
"""

import logging
from pathlib import Path
from typing import Optional

from config import get_settings
from models import Plan, ImagePlan, JobStatus, ImageStatus
from job_manager import JobManager
from nano_client import NanoBananaClient
from image_utils import (
    load_image_from_bytes,
    save_image,
    overlay_virtually_staged_label
)

logger = logging.getLogger(__name__)


class StagerRunner:
    """
    Executes the staging phase of the pipeline.
    
    Responsibilities:
    - Load plan from JobManager
    - Call NanoBananaClient for each image
    - Apply "Virtually Staged" overlay
    - Save staged images
    - Update plan with results
    """
    
    def __init__(
        self,
        job_manager: Optional[JobManager] = None,
        nano_client: Optional[NanoBananaClient] = None
    ):
        """
        Initialize StagerRunner.
        
        Args:
            job_manager: JobManager instance. Creates new if not provided.
            nano_client: NanoBananaClient instance. Creates new if not provided.
        """
        self.job_manager = job_manager or JobManager()
        self.nano_client = nano_client or NanoBananaClient()
        
        logger.info("StagerRunner initialized")
    
    async def run_staging_for_job(self, job_id: str) -> Plan:
        """
        Run staging for all planned images in a job.
        
        This is the main entry point for the staging phase.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Updated Plan object
            
        Raises:
            ValueError: If job doesn't exist or has no plan
            Exception: If staging fails
        """
        # Check if job is already complete
        if self.job_manager.is_job_complete(job_id):
            logger.info(f"Job {job_id} is already complete, skipping staging")
            return self.job_manager.load_plan(job_id)
        
        # Load plan
        if not self.job_manager.plan_exists(job_id):
            raise ValueError(f"No plan found for job {job_id}")
        
        plan = self.job_manager.load_plan(job_id)
        
        # Check if all images are already staged
        pending_images = [
            img for img in plan.images 
            if img.status in (ImageStatus.PLANNED, ImageStatus.NEEDS_REGEN)
        ]
        
        if not pending_images:
            logger.info(f"All images already staged for job {job_id}")
            return plan
        
        # Update job status
        self.job_manager.update_order_status(job_id, JobStatus.STAGING)
        
        settings = get_settings()
        job_dir = Path(settings.BASE_JOBS_DIR) / job_id
        staged_dir = job_dir / "staged"
        staged_dir.mkdir(exist_ok=True)
        
        logger.info(f"Staging {len(pending_images)} images for job {job_id}")
        
        success_count = 0
        fail_count = 0
        
        for img in pending_images:
            try:
                await self._stage_single_image(job_id, job_dir, img)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to stage {img.id}: {e}")
                img.status = ImageStatus.FAILED
                img.error_message = str(e)
                fail_count += 1
            
            # Save plan after each image (for resumability)
            self.job_manager.save_plan(plan)
        
        logger.info(f"Staging complete for job {job_id}: {success_count} success, {fail_count} failed")
        
        # Update job status
        if fail_count == 0:
            self.job_manager.update_order_status(job_id, JobStatus.STAGED)
        elif success_count > 0:
            # Partial success - still mark as staged but with a note
            self.job_manager.update_order_status(
                job_id, 
                JobStatus.STAGED,
                error_message=f"{fail_count} images failed staging"
            )
        else:
            # Total failure
            self.job_manager.update_order_status(
                job_id,
                JobStatus.FAILED,
                error_message="All images failed staging"
            )
        
        return plan
    
    async def _stage_single_image(
        self,
        job_id: str,
        job_dir: Path,
        image_plan: ImagePlan
    ) -> None:
        """
        Stage a single image.
        
        Args:
            job_id: Job identifier
            job_dir: Job directory path
            image_plan: ImagePlan object to update
        """
        logger.info(f"Staging image {image_plan.id} for job {job_id}")
        
        if not image_plan.nano_prompt:
            raise ValueError(f"No staging prompt for image {image_plan.id}")
        
        # Get source image path
        source_path = job_dir / image_plan.source_path
        if not source_path.exists():
            raise ValueError(f"Source image not found: {source_path}")
        
        # Mark as staging
        image_plan.status = ImageStatus.STAGING
        
        # Call Nano Banana
        staged_bytes = await self.nano_client.stage_image(
            base_image_path=source_path,
            prompt_text=image_plan.nano_prompt
        )
        
        # Load staged image
        staged_img = load_image_from_bytes(staged_bytes)
        
        # Save raw staged image (without label, for reference)
        base_output_path = job_dir / "staged" / f"{image_plan.id}_staged_base.jpg"
        save_image(staged_img, base_output_path)
        logger.debug(f"Saved base staged image to {base_output_path}")
        
        # Apply "Virtually Staged" overlay
        labeled_img = overlay_virtually_staged_label(staged_img)
        
        # Save final labeled image
        final_output_path = job_dir / "staged" / f"{image_plan.id}_staged_final.jpg"
        save_image(labeled_img, final_output_path)
        logger.info(f"Saved final staged image to {final_output_path}")
        
        # Update image plan
        image_plan.status = ImageStatus.STAGED
        image_plan.output_path = f"staged/{image_plan.id}_staged_final.jpg"
        image_plan.error_message = None
    
    async def restage_image(self, job_id: str, image_id: str) -> ImagePlan:
        """
        Re-stage a specific image.
        
        Useful for regenerating a single image that didn't turn out well.
        
        Args:
            job_id: Job identifier
            image_id: Image identifier (e.g., "img_1")
            
        Returns:
            Updated ImagePlan
        """
        plan = self.job_manager.load_plan(job_id)
        
        # Find the image
        image_plan = None
        for img in plan.images:
            if img.id == image_id:
                image_plan = img
                break
        
        if not image_plan:
            raise ValueError(f"Image {image_id} not found in plan")
        
        settings = get_settings()
        job_dir = Path(settings.BASE_JOBS_DIR) / job_id
        
        # Mark for regen and restage
        image_plan.status = ImageStatus.NEEDS_REGEN
        
        await self._stage_single_image(job_id, job_dir, image_plan)
        
        self.job_manager.save_plan(plan)
        
        return image_plan


async def run_staging_for_job(job_id: str) -> Plan:
    """
    Convenience function to run staging for a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Updated Plan object
    """
    runner = StagerRunner()
    return await runner.run_staging_for_job(job_id)
