"""
Stager Delivery - Handles packaging and email delivery of staged photos.
"""

import logging
import smtplib
import zipfile
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Optional

from config import get_settings
from models import Order, Plan, JobStatus
from job_manager import JobManager

logger = logging.getLogger(__name__)


class StagerDelivery:
    """
    Handles the delivery phase: packaging and emailing staged photos.
    
    Responsibilities:
    - Zip staged images
    - Send email to client
    - Mark job as complete
    - Update Airtable (stub)
    """
    
    def __init__(self, job_manager: Optional[JobManager] = None):
        """
        Initialize StagerDelivery.
        
        Args:
            job_manager: JobManager instance. Creates new if not provided.
        """
        self.job_manager = job_manager or JobManager()
        self.settings = get_settings()
        
        logger.info("StagerDelivery initialized")
    
    def package_staged_images(self, job_id: str) -> Path:
        """
        Create a zip file of all staged images.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Path to the created zip file
        """
        settings = get_settings()
        job_dir = Path(settings.BASE_JOBS_DIR) / job_id
        staged_dir = job_dir / "staged"
        final_dir = job_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        zip_path = final_dir / "staged_photos.zip"
        
        # Find all final staged images
        staged_files = list(staged_dir.glob("*_staged_final.jpg"))
        
        if not staged_files:
            raise ValueError(f"No staged images found for job {job_id}")
        
        logger.info(f"Creating zip with {len(staged_files)} images")
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_path in staged_files:
                # Use a clean filename in the zip
                arc_name = img_path.name.replace("_staged_final", "_staged")
                zf.write(img_path, arc_name)
        
        logger.info(f"Created zip file: {zip_path}")
        return zip_path
    
    def send_email(
        self,
        job_id: str,
        order: Order,
        zip_path: Path,
        attach_zip: bool = True
    ) -> None:
        """
        Send delivery email to client.
        
        Args:
            job_id: Job identifier
            order: Order with client info
            zip_path: Path to the zip file
            attach_zip: Whether to attach the zip file
        """
        settings = self.settings
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Your virtually staged photos for {order.address} are ready!"
        msg["From"] = settings.EMAIL_FROM
        msg["To"] = order.client.email
        
        # Plain text version
        text_body = f"""Hi {order.client.name},

Great news! Your professionally cleaned-up photos for {order.address} are ready.

{'Please find them attached to this email.' if attach_zip else 'Please download them from the link below.'}

Each photo has been professionally enhanced - decluttered, lighting improved, and straightened for a clean, professional real estate presentation.

If you have any questions or would like any adjustments, please don't hesitate to reply to this email.

Best regards,
44 Frames
www.44frames.com
"""
        
        # HTML version
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #4A90A4; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background: #f9f9f9; }}
        .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Your Staged Photos Are Ready! 🏠</h1>
        </div>
        <div class="content">
            <p>Hi {order.client.name},</p>
            
            <p>Great news! Your virtually staged photos for <strong>{order.address}</strong> are ready.</p>
            
            <p>{'Please find them attached to this email.' if attach_zip else 'Please download them using the link provided.'}</p>
            
            <p>Each photo has been professionally enhanced - <strong>decluttered, lighting improved, and straightened</strong>
            for a clean, professional real estate presentation.</p>
            
            <p>If you have any questions or would like any adjustments, please don't hesitate to reply to this email.</p>
            
            <p>Best regards,<br>The Stager Agent Team</p>
        </div>
        <div class="footer">
            <p>This email was sent by Stager Agent. All images are AI-generated virtual staging.</p>
        </div>
    </div>
</body>
</html>
"""
        
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        # Attach zip file if requested and not too large (< 25MB)
        if attach_zip and zip_path.stat().st_size < 25 * 1024 * 1024:
            with open(zip_path, "rb") as f:
                part = MIMEBase("application", "zip")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename=staged_photos_{job_id}.zip"
                )
                msg.attach(part)
            logger.info("Attached zip file to email")
        elif attach_zip:
            logger.warning(f"Zip file too large to attach: {zip_path.stat().st_size} bytes")
        
        # Send email
        try:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.starttls()
                if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
                    server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {order.client.email}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise
    
    def package_and_send(self, job_id: str) -> None:
        """
        Package staged images and send to client.
        
        This is the main entry point for the delivery phase.
        
        Args:
            job_id: Job identifier
        """
        # Check if already complete
        if self.job_manager.is_job_complete(job_id):
            logger.info(f"Job {job_id} already delivered, skipping")
            return
        
        logger.info(f"Starting delivery for job {job_id}")
        
        # Update status
        self.job_manager.update_order_status(job_id, JobStatus.PACKAGING)
        
        # Load order
        order = self.job_manager.load_order(job_id)
        
        try:
            # Package images
            zip_path = self.package_staged_images(job_id)
            
            # Send email
            self.send_email(job_id, order, zip_path)
            
            # Mark as complete
            self.job_manager.mark_job_complete(job_id)
            self.job_manager.update_order_status(job_id, JobStatus.DELIVERED)
            
            # Update Airtable (stub)
            self.update_airtable_status(order.airtable_record_id, "Delivered", str(zip_path))
            
            logger.info(f"Delivery complete for job {job_id}")
            
        except Exception as e:
            logger.error(f"Delivery failed for job {job_id}: {e}")
            self.job_manager.update_order_status(
                job_id,
                JobStatus.FAILED,
                error_message=f"Delivery failed: {str(e)}"
            )
            raise
    
    def update_airtable_status(
        self,
        record_id: str,
        status: str,
        download_url: Optional[str] = None
    ) -> None:
        """
        Update Airtable record with job status.
        
        TODO: Implement actual Airtable API call.
        This is a stub for future implementation.
        
        Args:
            record_id: Airtable record ID
            status: New status string
            download_url: Optional URL to staged photos
        """
        # TODO: Implement Airtable API integration
        # Example implementation:
        # 
        # import httpx
        # 
        # async with httpx.AsyncClient() as client:
        #     response = await client.patch(
        #         f"https://api.airtable.com/v0/{base_id}/{table_name}/{record_id}",
        #         headers={
        #             "Authorization": f"Bearer {airtable_api_key}",
        #             "Content-Type": "application/json"
        #         },
        #         json={
        #             "fields": {
        #                 "Status": status,
        #                 "Download URL": download_url,
        #                 "Completed At": datetime.utcnow().isoformat()
        #             }
        #         }
        #     )
        
        logger.info(f"[STUB] Would update Airtable record {record_id} to status: {status}")


def package_and_send(job_id: str) -> None:
    """
    Convenience function to package and send for a job.
    
    Args:
        job_id: Job identifier
    """
    delivery = StagerDelivery()
    delivery.package_and_send(job_id)
