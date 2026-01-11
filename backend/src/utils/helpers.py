import hashlib
import uuid
from typing import Any, Dict, List
import logging
from datetime import datetime


def generate_checksum(content: str) -> str:
    """
    Generate SHA-256 checksum for content to detect changes
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def generate_uuid() -> str:
    """
    Generate a UUID string
    """
    return str(uuid.uuid4())


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with the specified name and level
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def format_response(data: Any, message: str = "Success", status_code: int = 200) -> Dict:
    """
    Format a standardized response
    """
    return {
        "status_code": status_code,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }


def paginate_list(items: List[Any], page: int = 1, size: int = 10) -> Dict:
    """
    Paginate a list of items
    """
    start = (page - 1) * size
    end = start + size
    paginated_items = items[start:end]

    return {
        "items": paginated_items,
        "total": len(items),
        "page": page,
        "size": size,
        "pages": (len(items) + size - 1) // size
    }


def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary with a default fallback
    """
    return dictionary.get(key, default)


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries, with dict2 values taking precedence
    """
    merged = dict1.copy()
    merged.update(dict2)
    return merged