"""
Time utilities for consistent timestamp handling.
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """
    Get current UTC time as a timezone-aware datetime.
    
    Returns:
        Current UTC datetime
    """
    return datetime.now(timezone.utc)


def format_iso8601(dt: datetime) -> str:
    """
    Format datetime as ISO8601 string with Z suffix.
    
    Args:
        dt: Datetime to format
        
    Returns:
        ISO8601 formatted string
        
    Examples:
        >>> format_iso8601(datetime(2025, 12, 4, 18, 30, 0))
        '2025-12-04T18:30:00Z'
    """
    # Remove timezone info and add Z suffix
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt.isoformat() + "Z"


def parse_iso8601(s: str) -> datetime:
    """
    Parse ISO8601 string to datetime.
    
    Args:
        s: ISO8601 formatted string
        
    Returns:
        Parsed datetime
    """
    # Handle Z suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)
