"""
Slugify utility for creating URL-safe identifiers.
"""

import re
import unicodedata


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to a URL-safe slug.
    
    Args:
        text: Input text to slugify
        max_length: Maximum length of the resulting slug
        
    Returns:
        URL-safe slug string
        
    Examples:
        >>> slugify("123 Main St, Boston, MA 02116")
        '123-main-st-boston-ma-02116'
        >>> slugify("Hello World!")
        'hello-world'
    """
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and special characters with hyphens
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    
    # Remove leading/trailing hyphens
    text = text.strip("-")
    
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length].rstrip("-")
    
    return text


def generate_job_id(address: str, uuid_suffix: str) -> str:
    """
    Generate a job ID from address and UUID.
    
    Args:
        address: Property address
        uuid_suffix: Short UUID suffix (6-8 chars)
        
    Returns:
        Job ID in format: slugified-address-uuid
        
    Examples:
        >>> generate_job_id("123 Main St, Boston, MA", "a1b2c3")
        '123-main-st-boston-ma-a1b2c3'
    """
    address_slug = slugify(address, max_length=40)
    return f"{address_slug}-{uuid_suffix}"
