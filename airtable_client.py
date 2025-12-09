"""
Airtable Client for updating order status.

Updates the Status column in Airtable to track job progress:
- "To do" → "In progress" (when processing starts)
- "In progress" → "Done" (when email is sent)
"""

import logging
from typing import Optional

import httpx

from config import get_settings

logger = logging.getLogger(__name__)


class AirtableClient:
    """Client for Airtable API to update order status."""

    def __init__(self):
        """Initialize Airtable client with config settings."""
        settings = get_settings()
        self.api_key = settings.AIRTABLE_API_KEY
        self.base_id = settings.AIRTABLE_BASE_ID
        self.table_name = settings.AIRTABLE_TABLE_NAME
        self.timeout = settings.REQUEST_TIMEOUT

        # Check if Airtable is configured
        self.enabled = bool(self.api_key and self.base_id)
        if self.enabled:
            logger.info(f"AirtableClient initialized for base {self.base_id}")
        else:
            logger.warning("AirtableClient disabled - missing AIRTABLE_API_KEY or AIRTABLE_BASE_ID")

    async def update_status(
        self,
        record_id: str,
        status: str,
        additional_fields: Optional[dict] = None
    ) -> bool:
        """
        Update the Status field for an Airtable record.

        Args:
            record_id: Airtable record ID (starts with 'rec')
            status: New status value ("To do", "In progress", "Done")
            additional_fields: Optional dict of additional fields to update

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.info(f"[STUB] Would update Airtable record {record_id} to status: {status}")
            return True

        url = f"https://api.airtable.com/v0/{self.base_id}/{self.table_name}/{record_id}"

        fields = {"Status": status}
        if additional_fields:
            fields.update(additional_fields)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.patch(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"fields": fields}
                )

                if response.status_code == 200:
                    logger.info(f"Updated Airtable record {record_id} to status: {status}")
                    return True
                else:
                    logger.error(f"Airtable update failed: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Failed to update Airtable record {record_id}: {e}")
            return False

    async def mark_in_progress(self, record_id: str) -> bool:
        """Mark a record as 'In progress'."""
        return await self.update_status(record_id, "In progress")

    async def mark_done(self, record_id: str) -> bool:
        """Mark a record as 'Done'."""
        return await self.update_status(record_id, "Done")

    async def mark_error(self, record_id: str, error_message: Optional[str] = None) -> bool:
        """Mark a record as 'ERROR' with optional error message."""
        additional = {"ERROR": error_message} if error_message else None
        return await self.update_status(record_id, "ERROR", additional)
