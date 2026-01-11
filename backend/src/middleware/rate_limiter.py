import time
from typing import Dict
from collections import defaultdict
from fastapi import Request, HTTPException, status
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter
    In production, use Redis or similar for distributed rate limiting
    """
    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window  # in seconds
        self.requests_log: Dict[str, list] = defaultdict(list)

    def check_rate_limit(self, key: str) -> bool:
        """
        Check if the key has exceeded the rate limit
        """
        now = time.time()
        # Remove old requests outside the window
        self.requests_log[key] = [
            req_time for req_time in self.requests_log[key]
            if now - req_time < self.window
        ]

        # Check if limit exceeded
        if len(self.requests_log[key]) >= self.requests:
            return False

        # Add current request
        self.requests_log[key].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(requests=100, window=60)  # 100 requests per minute


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware
    """
    # Use client IP as the rate limit key
    client_ip = request.client.host
    key = f"ip:{client_ip}"

    if not rate_limiter.check_rate_limit(key):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

    response = await call_next(request)
    return response