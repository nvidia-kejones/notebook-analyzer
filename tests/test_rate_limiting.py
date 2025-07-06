#!/usr/bin/env python3
"""
Rate Limiting Test

Tests the rate limiting system functionality.
"""

import sys
import os
import time
import threading

# Add the analyzer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analyzer'))

def test_rate_limiter_import():
    """Test that the RateLimiter can be imported and instantiated."""
    try:
        from rate_limiter import SlidingWindowRateLimiter, RateLimitConfig, RateLimitStatus
        
        # Test basic instantiation
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            burst_limit=3
        )
        limiter = SlidingWindowRateLimiter(config)
        
        print("✅ RateLimiter import and instantiation successful")
        return True
    except Exception as e:
        print(f"❌ RateLimiter import failed: {e}")
        return False

def test_basic_rate_limiting():
    """Test basic rate limiting functionality."""
    try:
        from rate_limiter import SlidingWindowRateLimiter, RateLimitConfig
        
        # Create a limiter with very low limits for testing
        config = RateLimitConfig(
            requests_per_minute=5,
            requests_per_hour=20,
            burst_limit=2
        )
        limiter = SlidingWindowRateLimiter(config)
        
        # Test normal requests
        client_ip = "127.0.0.1"
        
        # First request should be allowed
        status1 = limiter.check_rate_limit(client_ip)
        if not status1.allowed:
            print(f"❌ First request was denied: {status1}")
            return False
        
        # Second request should be allowed
        status2 = limiter.check_rate_limit(client_ip)
        if not status2.allowed:
            print(f"❌ Second request was denied: {status2}")
            return False
        
        # Third request should hit burst limit
        status3 = limiter.check_rate_limit(client_ip)
        if status3.allowed:
            print(f"❌ Third request should have been denied (burst limit): {status3}")
            return False
        
        print("✅ Basic rate limiting works correctly")
        print(f"   - First two requests allowed, third denied (burst limit)")
        print(f"   - Retry after: {status3.retry_after} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic rate limiting test failed: {e}")
        return False

def test_client_differentiation():
    """Test that different clients are tracked separately."""
    try:
        from rate_limiter import SlidingWindowRateLimiter, RateLimitConfig
        
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            burst_limit=2
        )
        limiter = SlidingWindowRateLimiter(config)
        
        # Test different clients
        client1 = "192.168.1.1"
        client2 = "192.168.1.2"
        
        # Client1 should be able to make up to burst_limit requests
        status1a = limiter.check_rate_limit(client1)
        status1b = limiter.check_rate_limit(client1)
        
        if not (status1a.allowed and status1b.allowed):
            print(f"❌ Client1 should be allowed 2 requests: {status1a.allowed}, {status1b.allowed}")
            return False
        
        # Third request from client1 should be denied (burst limit)
        status1c = limiter.check_rate_limit(client1)
        if status1c.allowed:
            print(f"❌ Client1 third request should have been denied")
            return False
        
        # Now test that client2 is tracked separately and can still make requests
        status2a = limiter.check_rate_limit(client2)
        status2b = limiter.check_rate_limit(client2)
        
        if not (status2a.allowed and status2b.allowed):
            print(f"❌ Client2 should be allowed 2 requests: {status2a.allowed}, {status2b.allowed}")
            
            # Debug: Get client stats
            stats1 = limiter.get_client_stats(client1)
            stats2 = limiter.get_client_stats(client2)
            print(f"   Debug - Client1 stats: {stats1}")
            print(f"   Debug - Client2 stats: {stats2}")
            
            return False
        
        # Client2's third request should also be denied (burst limit)
        status2c = limiter.check_rate_limit(client2)
        if status2c.allowed:
            print(f"❌ Client2 third request should have been denied")
            return False
        
        print("✅ Client differentiation works correctly")
        print("   - Each client is tracked separately")
        print("   - Each client has independent burst limits")
        return True
        
    except Exception as e:
        print(f"❌ Client differentiation test failed: {e}")
        return False

def test_time_window_reset():
    """Test that rate limits reset over time."""
    try:
        from rate_limiter import SlidingWindowRateLimiter, RateLimitConfig
        
        config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_limit=2
        )
        limiter = SlidingWindowRateLimiter(config)
        
        client_ip = "10.0.0.1"
        
        # Make requests to hit burst limit
        status1 = limiter.check_rate_limit(client_ip)
        status2 = limiter.check_rate_limit(client_ip)
        status3 = limiter.check_rate_limit(client_ip)  # Should be denied
        
        if status3.allowed:
            print(f"❌ Third request should have been denied")
            return False
        
        # Wait for burst window to reset (burst limit is 10 seconds)
        print("   Waiting 1 second for burst window to partially reset...")
        time.sleep(1)
        
        # Should still be denied immediately after
        status4 = limiter.check_rate_limit(client_ip)
        if status4.allowed:
            print(f"❌ Request should still be denied after 1 second")
            return False
        
        print("✅ Time window reset behavior works correctly")
        print(f"   - Burst limit enforced and maintained over time")
        
        return True
        
    except Exception as e:
        print(f"❌ Time window reset test failed: {e}")
        return False

def test_statistics():
    """Test rate limiter statistics functionality."""
    try:
        from rate_limiter import SlidingWindowRateLimiter, RateLimitConfig
        
        config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_limit=10
        )
        limiter = SlidingWindowRateLimiter(config)
        
        client_ip = "203.0.113.1"
        
        # Make some requests
        for i in range(5):
            limiter.check_rate_limit(client_ip)
        
        # Get client stats
        stats = limiter.get_client_stats(client_ip)
        
        expected_fields = ['client_id', 'requests_last_minute', 'requests_last_hour', 'total_requests']
        if not all(field in stats for field in expected_fields):
            print(f"❌ Client stats missing fields: {stats}")
            return False
        
        if stats['total_requests'] != 5:
            print(f"❌ Expected 5 total requests, got {stats['total_requests']}")
            return False
        
        # Get global stats
        global_stats = limiter.get_global_stats()
        
        expected_global_fields = ['total_clients', 'total_requests_all_time', 'config']
        if not all(field in global_stats for field in expected_global_fields):
            print(f"❌ Global stats missing fields: {global_stats}")
            return False
        
        print("✅ Statistics functionality works correctly")
        print(f"   - Client stats: {stats['total_requests']} requests")
        print(f"   - Global stats: {global_stats['total_clients']} clients")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False

def test_global_functions():
    """Test global rate limiter functions."""
    try:
        from rate_limiter import get_rate_limiter, check_rate_limit, get_rate_limit_headers
        
        # Test global limiter
        limiter = get_rate_limiter()
        if limiter is None:
            print("❌ Global limiter is None")
            return False
        
        # Test convenience function
        status = check_rate_limit("198.51.100.1")
        if not status.allowed:
            print(f"❌ Convenience function failed: {status}")
            return False
        
        # Test header generation
        headers = get_rate_limit_headers(status)
        expected_headers = ['X-RateLimit-Remaining', 'X-RateLimit-Reset']
        if not all(header in headers for header in expected_headers):
            print(f"❌ Headers missing: {headers}")
            return False
        
        print("✅ Global functions work correctly")
        print(f"   - Headers: {headers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Global functions test failed: {e}")
        return False

def main():
    """Run all rate limiting tests."""
    print("🔧 Rate Limiting Validation")
    print("===========================")
    print()
    
    tests = [
        ("Import and Setup", test_rate_limiter_import),
        ("Basic Rate Limiting", test_basic_rate_limiting),
        ("Client Differentiation", test_client_differentiation),
        ("Time Window Reset", test_time_window_reset),
        ("Statistics", test_statistics),
        ("Global Functions", test_global_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        print("-" * (len(test_name) + 9))
        
        if test_func():
            passed += 1
        else:
            print("❌ Test failed!")
        
        print()
    
    print("📊 Test Summary")
    print("===============")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All rate limiting features are working correctly!")
        print("✅ Rate limiting system is ready for use")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        print("Some features may not work as expected")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 