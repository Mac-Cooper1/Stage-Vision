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


# Style display names for email template
STYLE_DISPLAY_NAMES = {
    "architecture_digest": "Architecture Digest",
    "modern": "Modern",
    "coastal": "Coastal",
    "farmhouse": "Farmhouse",
    "midcentury": "Mid-Century Modern",
    "scandinavian": "Scandinavian",
}


def get_style_display_name(internal_style: str) -> str:
    """Convert internal style code to friendly display name."""
    return STYLE_DISPLAY_NAMES.get(internal_style, internal_style.replace("_", " ").title())


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
        photo_count: int,
        attach_zip: bool = True
    ) -> None:
        """
        Send delivery email to client.

        Args:
            job_id: Job identifier
            order: Order with client info
            zip_path: Path to the zip file
            photo_count: Number of photos staged
            attach_zip: Whether to attach the zip file
        """
        settings = self.settings

        # Get friendly style display name
        style_name = get_style_display_name(order.style)

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Your Stage Vision Photos Are Ready! | {order.address}"
        msg["From"] = settings.EMAIL_FROM
        msg["To"] = order.client.email

        # Plain text version
        text_body = f"""Hi {order.client.name},

Great news! Your virtually staged photos for {order.address} are ready.

Style Applied: {style_name}

Your {photo_count} photo(s) have been professionally transformed with virtual staging - empty rooms are now beautifully furnished, and occupied spaces have been decluttered and refreshed. Each image includes realistic furniture, decor, and lighting enhancements designed to help buyers envision the full potential of the property.

{'Your staged photos are attached to this email.' if attach_zip else 'Please download them from the link provided.'}

Important Notes:
- Each photo is labeled "Virtually Staged" for MLS compliance
- Original architectural features and room dimensions are preserved
- These images are optimized for MLS listings and marketing materials

Questions? Reach out to mcooper@44frames.com if you need any assistance.

Thank you for choosing Stage Vision!

Best regards,
The Stage Vision Team
A 44 Frames Service
www.44frames.com
"""

        # HTML version
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.7; color: #333; margin: 0; padding: 0; background-color: #f4f4f4; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; }}
        .header {{ background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%); color: white; padding: 30px 20px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 24px; font-weight: 600; }}
        .header p {{ margin: 8px 0 0 0; opacity: 0.9; font-size: 14px; }}
        .content {{ padding: 30px; }}
        .style-badge {{ background: #E8F4F8; border-left: 4px solid #2C3E50; padding: 12px 16px; margin: 20px 0; }}
        .style-badge strong {{ color: #2C3E50; }}
        .notes {{ background: #FFF9E6; border: 1px solid #F0E6CC; border-radius: 6px; padding: 16px; margin: 20px 0; }}
        .notes h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #8B7355; }}
        .notes ul {{ margin: 0; padding-left: 20px; }}
        .notes li {{ margin: 4px 0; font-size: 13px; color: #666; }}
        .footer {{ padding: 20px; text-align: center; background: #f9f9f9; border-top: 1px solid #eee; }}
        .footer p {{ margin: 4px 0; font-size: 12px; color: #888; }}
        .footer a {{ color: #2C3E50; text-decoration: none; }}
        .signature {{ margin-top: 25px; padding-top: 20px; border-top: 1px solid #eee; }}
        .signature strong {{ color: #2C3E50; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Your Stage Vision Photos Are Ready!</h1>
            <p>{order.address}</p>
        </div>
        <div class="content">
            <p>Hi {order.client.name},</p>

            <p>Great news! Your virtually staged photos are ready.</p>

            <div class="style-badge">
                <strong>Style Applied:</strong> {style_name}
            </div>

            <p>Your <strong>{photo_count} photo(s)</strong> have been professionally transformed with virtual staging &mdash;
            empty rooms are now beautifully furnished, and occupied spaces have been decluttered and refreshed.
            Each image includes realistic furniture, decor, and lighting enhancements designed to help buyers
            envision the full potential of the property.</p>

            <p>{'📎 <strong>Your staged photos are attached to this email.</strong>' if attach_zip else 'Please download them using the link provided.'}</p>

            <div class="notes">
                <h3>Important Notes:</h3>
                <ul>
                    <li>Each photo is labeled "Virtually Staged" for MLS compliance</li>
                    <li>Original architectural features and room dimensions are preserved</li>
                    <li>These images are optimized for MLS listings and marketing materials</li>
                </ul>
            </div>

            <p>Questions? Reach out to <a href="mailto:mcooper@44frames.com">mcooper@44frames.com</a> if you need any assistance.</p>

            <p>Thank you for choosing Stage Vision!</p>

            <div class="signature">
                <p>Best regards,<br>
                <strong>The Stage Vision Team</strong><br>
                <span style="color: #888; font-size: 13px;">A 44 Frames Service</span></p>
            </div>
        </div>
        <div class="footer">
            <p><a href="https://www.44frames.com">www.44frames.com</a></p>
            <p>All images are AI-generated virtual staging for marketing purposes.</p>
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

        # Load order and plan
        order = self.job_manager.load_order(job_id)

        # Get photo count from plan (count successfully staged images)
        photo_count = 0
        if self.job_manager.plan_exists(job_id):
            plan = self.job_manager.load_plan(job_id)
            photo_count = sum(1 for img in plan.images if img.status == "staged")

        # Fallback: count files in staged directory if plan unavailable
        if photo_count == 0:
            settings = get_settings()
            staged_dir = Path(settings.BASE_JOBS_DIR) / job_id / "staged"
            if staged_dir.exists():
                photo_count = len(list(staged_dir.glob("*_staged_final.jpg")))

        try:
            # Package images
            zip_path = self.package_staged_images(job_id)

            # Send email with photo count
            self.send_email(job_id, order, zip_path, photo_count)

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
