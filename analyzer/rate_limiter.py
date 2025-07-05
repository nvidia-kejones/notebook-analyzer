#!/usr/bin/env python3
"""
Rate Limiter for API Endpoints

Implements sliding window rate limiting to protect against abuse
and ensure fair usage of the notebook analyzer API endpoints.
"""

import time
import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib
import os


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10  # Allow short bursts
    cleanup_interval: int = 300  # Cleanup old entries every 5 minutes


@dataclass
class RateLimitStatus:
    """Status of rate limiting for a client."""
    allowed: bool
    remaining_requests: int
    reset_time: int
    retry_after: Optional[int] = None


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with configurable limits.
    
    Uses a sliding window approach to track request rates
    and provides burst protection while being memory efficient.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize the rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config or RateLimitConfig()
        
        # Storage for request timestamps per client
        self._client_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self._client_last_cleanup: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Last global cleanup time
        self._last_global_cleanup = time.time()
    
    def _get_client_id(self, ip_address: str, user_agent: Optional[str] = None) -> str:
        """
        Generate a client ID for rate limiting.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent (optional)
            
        Returns:
            Unique client identifier
        """
        # Use IP address as primary identifier
        client_key = ip_address
        
        # Optionally include user agent for more granular tracking
        if user_agent and len(user_agent) > 10:  # Only if meaningful
            # Hash user agent to avoid storing long strings
            ua_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
            client_key = f"{ip_address}:{ua_hash}"
        
        return client_key
    
    def _cleanup_old_requests(self, client_id: str, current_time: float):
        """
        Clean up old request timestamps for a client.
        
        Args:
            client_id: Client identifier
            current_time: Current timestamp
        """
        if client_id not in self._client_requests:
            return
        
        requests = self._client_requests[client_id]
        
        # Remove requests older than 1 hour (our longest window)
        cutoff_time = current_time - 3600
        
        while requests and requests[0] < cutoff_time:
            requests.popleft()
        
        # Update last cleanup time
        self._client_last_cleanup[client_id] = current_time
    
    def _global_cleanup(self, current_time: float):
        """
        Perform global cleanup of inactive clients.
        
        Args:
            current_time: Current timestamp
        """
        if current_time - self._last_global_cleanup < self.config.cleanup_interval:
            return
        
        # Find clients to clean up (no activity in last 2 hours)
        inactive_cutoff = current_time - 7200
        clients_to_remove = []
        
        for client_id, requests in self._client_requests.items():
            if not requests or requests[-1] < inactive_cutoff:
                clients_to_remove.append(client_id)
        
        # Remove inactive clients
        for client_id in clients_to_remove:
            del self._client_requests[client_id]
            self._client_last_cleanup.pop(client_id, None)
        
        self._last_global_cleanup = current_time
    
    def _count_requests_in_window(self, requests: deque, window_seconds: int, 
                                 current_time: float) -> int:
        """
        Count requests within a time window.
        
        Args:
            requests: Deque of request timestamps
            window_seconds: Window size in seconds
            current_time: Current timestamp
            
        Returns:
            Number of requests in the window
        """
        cutoff_time = current_time - window_seconds
        
        # Count requests after cutoff time
        count = 0
        for timestamp in reversed(requests):
            if timestamp >= cutoff_time:
                count += 1
            else:
                break  # Requests are ordered, so we can stop here
        
        return count
    
    def check_rate_limit(self, ip_address: str, user_agent: Optional[str] = None) -> RateLimitStatus:
        """
        Check if a request is allowed under rate limits.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent (optional)
            
        Returns:
            RateLimitStatus indicating if request is allowed
        """
        current_time = time.time()
        client_id = self._get_client_id(ip_address, user_agent)
        
        with self._lock:
            # Perform cleanup
            self._cleanup_old_requests(client_id, current_time)
            self._global_cleanup(current_time)
            
            requests = self._client_requests[client_id]
            
            # Check minute limit
            minute_requests = self._count_requests_in_window(requests, 60, current_time)
            if minute_requests >= self.config.requests_per_minute:
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=int(current_time + 60),
                    retry_after=60
                )
            
            # Check hour limit
            hour_requests = self._count_requests_in_window(requests, 3600, current_time)
            if hour_requests >= self.config.requests_per_hour:
                # Calculate when the oldest request in the hour will expire
                oldest_in_hour = None
                for timestamp in requests:
                    if timestamp >= current_time - 3600:
                        oldest_in_hour = timestamp
                        break
                
                retry_after = int((oldest_in_hour or current_time) + 3600 - current_time)
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=int(current_time + retry_after),
                    retry_after=retry_after
                )
            
            # Check burst limit (requests in last 10 seconds)
            burst_requests = self._count_requests_in_window(requests, 10, current_time)
            if burst_requests >= self.config.burst_limit:
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=int(current_time + 10),
                    retry_after=10
                )
            
            # Request is allowed - record it
            requests.append(current_time)
            
            # Calculate remaining requests (use the most restrictive limit)
            remaining_minute = max(0, self.config.requests_per_minute - minute_requests - 1)
            remaining_hour = max(0, self.config.requests_per_hour - hour_requests - 1)
            remaining_burst = max(0, self.config.burst_limit - burst_requests - 1)
            
            remaining_requests = min(remaining_minute, remaining_hour, remaining_burst)
            
            return RateLimitStatus(
                allowed=True,
                remaining_requests=remaining_requests,
                reset_time=int(current_time + 60)  # Next minute reset
            )
    
    def get_client_stats(self, ip_address: str, user_agent: str = None) -> Dict:
        """
        Get statistics for a specific client.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent (optional)
            
        Returns:
            Dictionary with client statistics
        """
        current_time = time.time()
        client_id = self._get_client_id(ip_address, user_agent)
        
        with self._lock:
            self._cleanup_old_requests(client_id, current_time)
            requests = self._client_requests[client_id]
            
            minute_requests = self._count_requests_in_window(requests, 60, current_time)
            hour_requests = self._count_requests_in_window(requests, 3600, current_time)
            burst_requests = self._count_requests_in_window(requests, 10, current_time)
            
            return {
                'client_id': client_id,
                'requests_last_minute': minute_requests,
                'requests_last_hour': hour_requests,
                'requests_last_10_seconds': burst_requests,
                'total_requests': len(requests),
                'first_request': requests[0] if requests else None,
                'last_request': requests[-1] if requests else None
            }
    
    def get_global_stats(self) -> Dict:
        """
        Get global rate limiter statistics.
        
        Returns:
            Dictionary with global statistics
        """
        current_time = time.time()
        
        with self._lock:
            self._global_cleanup(current_time)
            
            total_clients = len(self._client_requests)
            total_requests = sum(len(requests) for requests in self._client_requests.values())
            
            # Count active clients (requests in last hour)
            active_clients = 0
            recent_requests = 0
            cutoff_time = current_time - 3600
            
            for requests in self._client_requests.values():
                if requests and requests[-1] >= cutoff_time:
                    active_clients += 1
                    # Count recent requests for this client
                    for timestamp in reversed(requests):
                        if timestamp >= cutoff_time:
                            recent_requests += 1
                        else:
                            break
            
            return {
                'total_clients': total_clients,
                'active_clients_last_hour': active_clients,
                'total_requests_all_time': total_requests,
                'requests_last_hour': recent_requests,
                'config': {
                    'requests_per_minute': self.config.requests_per_minute,
                    'requests_per_hour': self.config.requests_per_hour,
                    'burst_limit': self.config.burst_limit
                }
            }


# Global rate limiter instance
_rate_limiter = None

def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        # Load configuration from environment variables
        config = RateLimitConfig(
            requests_per_minute=int(os.environ.get('RATE_LIMIT_PER_MINUTE', '60')),
            requests_per_hour=int(os.environ.get('RATE_LIMIT_PER_HOUR', '1000')),
            burst_limit=int(os.environ.get('RATE_LIMIT_BURST', '10')),
            cleanup_interval=int(os.environ.get('RATE_LIMIT_CLEANUP_INTERVAL', '300'))
        )
        
        _rate_limiter = SlidingWindowRateLimiter(config)
    
    return _rate_limiter


def check_rate_limit(ip_address: str, user_agent: str = None) -> RateLimitStatus:
    """
    Convenience function to check rate limits.
    
    Args:
        ip_address: Client IP address
        user_agent: Client user agent (optional)
        
    Returns:
        RateLimitStatus indicating if request is allowed
    """
    limiter = get_rate_limiter()
    return limiter.check_rate_limit(ip_address, user_agent)


def get_rate_limit_headers(status: RateLimitStatus) -> Dict[str, str]:
    """
    Generate HTTP headers for rate limiting.
    
    Args:
        status: Rate limit status
        
    Returns:
        Dictionary of HTTP headers
    """
    headers = {
        'X-RateLimit-Remaining': str(status.remaining_requests),
        'X-RateLimit-Reset': str(status.reset_time),
    }
    
    if status.retry_after:
        headers['Retry-After'] = str(status.retry_after)
    
    return headers 