import functools
import time
from typing import Any, Callable, Dict
from threading import Lock


class SimpleCache:
    """
    A simple in-memory cache with TTL (Time To Live)
    In production, consider using Redis or Memcached
    """
    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._lock = Lock()

    def get(self, key: str) -> Any:
        """
        Get a value from the cache
        Returns None if the key doesn't exist or has expired
        """
        with self._lock:
            if key in self._cache:
                value, expiry_time = self._cache[key]
                if time.time() < expiry_time:
                    return value
                else:
                    # Remove expired entry
                    del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """
        Set a value in the cache with TTL in seconds
        Default TTL is 5 minutes (300 seconds)
        """
        with self._lock:
            expiry_time = time.time() + ttl
            self._cache[key] = (value, expiry_time)

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache
        Returns True if the key existed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False

    def clear(self) -> None:
        """
        Clear all entries from the cache
        """
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache
        Returns the number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry_time) in self._cache.items()
                if current_time >= expiry_time
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)


# Global cache instance
cache = SimpleCache()


def cached(ttl: int = 300):
    """
    Decorator to cache function results
    ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from function name and arguments
            cache_key = f"{func.__module__}.{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator


# Example usage:
# @cached(ttl=60)  # Cache for 60 seconds
# def expensive_operation(param):
#     # Simulate expensive operation
#     time.sleep(2)
#     return f"Result for {param}"