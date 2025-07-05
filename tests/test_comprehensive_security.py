#!/usr/bin/env python3
"""
Comprehensive Security Test Suite

Tests all Phase 2 security features including:
- Process isolation and resource monitoring
- Filesystem restrictions and secure file operations
- Rate limiting with sliding window algorithm
- Request validation and security pattern detection
- Security event logging and audit trails
- Security context managers and resource cleanup
- Secure error handling and information disclosure prevention
"""

import unittest
import os
import sys
import tempfile
import shutil
import time
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import security modules
try:
    from analyzer.security_sandbox import SecuritySandbox
    from analyzer.security_logger import SecurityLogger, SecurityEventType, SecurityEventSeverity
    from analyzer.rate_limiter import SlidingWindowRateLimiter, RateLimitConfig
    from analyzer.request_validator import RequestValidator, ValidationResult
    from analyzer.security_context_managers import SecurityContextManager, SecurityContext
    from analyzer.secure_error_handler import SecureErrorHandler, ErrorSeverity, ErrorCategory
    from analyzer.filesystem_restrictions import FilesystemRestrictions
    SECURITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some security modules not available: {e}")
    SECURITY_MODULES_AVAILABLE = False


class TestComprehensiveSecurity(unittest.TestCase):
    """Comprehensive security test suite."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []
        
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temp files
        for test_file in self.test_files:
            try:
                if os.path.exists(test_file):
                    os.unlink(test_file)
            except Exception:
                pass
        
        # Clean up temp directory
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    @unittest.skipUnless(SECURITY_MODULES_AVAILABLE, "Security modules not available")
    def test_process_isolation_integration(self):
        """Test process isolation with resource monitoring."""
        print("\n🔒 Testing Process Isolation Integration...")
        
        # Create security sandbox
        sandbox = SecuritySandbox(max_memory_mb=128, max_time_seconds=5)
        
        # Test process isolation context
        with sandbox.isolated_process_context() as context:
            # Create a simple test process
            import subprocess
            process = subprocess.Popen(['echo', 'test'], stdout=subprocess.PIPE)
            context.add_process(process)
            
            # Monitor process resources
            resources = sandbox.monitor_process_resources(process)
            
            # Verify resource monitoring
            self.assertIsInstance(resources, dict)
            self.assertIn('memory_mb', resources)
            self.assertIn('cpu_percent', resources)
            
            # Process should complete successfully
            process.wait()
            self.assertEqual(process.returncode, 0)
        
        print("✅ Process isolation integration working")
    
    @unittest.skipUnless(SECURITY_MODULES_AVAILABLE, "Security modules not available")
    def test_filesystem_restrictions_integration(self):
        """Test filesystem restrictions with secure operations."""
        print("\n🔒 Testing Filesystem Restrictions Integration...")
        
        # Create filesystem restrictions
        restrictions = FilesystemRestrictions()
        
        # Test secure file operations
        test_content = "Test content for security validation"
        
        # Create secure temp file
        with restrictions.secure_filesystem_context() as fs_context:
            temp_file = fs_context.create_secure_temp_file(test_content, ".txt")
            self.test_files.append(temp_file)
            
            # Verify file was created securely
            self.assertTrue(os.path.exists(temp_file))
            
            # Test secure file reading
            success, content, error = restrictions.read_file_safely(temp_file)
            self.assertTrue(success)
            self.assertEqual(content, test_content)
            self.assertEqual(error, "")
            
            # Test path validation
            is_allowed, reason = restrictions._is_path_allowed(temp_file, 'read')
            self.assertTrue(is_allowed)
            
            # Test blocked path access
            is_allowed, reason = restrictions._is_path_allowed('/etc/passwd', 'read')
            self.assertFalse(is_allowed)
            self.assertIn('blocked', reason.lower())
        
        print("✅ Filesystem restrictions integration working")
    
    @unittest.skipUnless(SECURITY_MODULES_AVAILABLE, "Security modules not available")
    def test_rate_limiting_integration(self):
        """Test rate limiting with multiple clients."""
        print("\n🔒 Testing Rate Limiting Integration...")
        
        # Create rate limiter with strict limits
        config = RateLimitConfig(
            max_requests_per_minute=5,
            max_requests_per_hour=20,
            burst_limit=3
        )
        rate_limiter = SlidingWindowRateLimiter(config)
        
        # Test normal usage
        client_ip = "192.168.1.100"
        user_agent = "test-client"
        
        # Should allow initial requests
        for i in range(3):
            status = rate_limiter.check_rate_limit(client_ip, user_agent)
            self.assertTrue(status.allowed)
            self.assertGreater(status.remaining_requests, 0)
        
        # Should hit burst limit
        for i in range(3):
            status = rate_limiter.check_rate_limit(client_ip, user_agent)
            if not status.allowed:
                self.assertGreater(status.retry_after, 0)
                break
        
        # Test different client (should be allowed)
        different_client = "192.168.1.101"
        status = rate_limiter.check_rate_limit(different_client, user_agent)
        self.assertTrue(status.allowed)
        
        # Test statistics
        stats = rate_limiter.get_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_requests', stats)
        self.assertIn('blocked_requests', stats)
        
        print("✅ Rate limiting integration working")
    
    @unittest.skipUnless(SECURITY_MODULES_AVAILABLE, "Security modules not available")
    def test_request_validation_integration(self):
        """Test request validation with security patterns."""
        print("\n🔒 Testing Request Validation Integration...")
        
        # Create request validator
        validator = RequestValidator()
        
        # Test valid request
        valid_request = {
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'json': {'file': 'test.ipynb', 'content': 'valid content'}
        }
        
        result = validator.validate_request(valid_request, "192.168.1.100")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.risk_level, "low")
        
        # Test malicious request with SQL injection
        malicious_request = {
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'json': {'query': "'; DROP TABLE users; --"}
        }
        
        result = validator.validate_request(malicious_request, "192.168.1.100")
        self.assertFalse(result.is_valid)
        self.assertIn("sql", result.error_code.lower())
        
        # Test XSS attempt
        xss_request = {
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'json': {'content': '<script>alert("xss")</script>'}
        }
        
        result = validator.validate_request(xss_request, "192.168.1.100")
        self.assertFalse(result.is_valid)
        self.assertIn("xss", result.error_code.lower())
        
        # Test file upload validation
        file_request = {
            'method': 'POST',
            'files': {'file': {'filename': 'test.ipynb', 'size': 1024}}
        }
        
        result = validator.validate_request(file_request, "192.168.1.100")
        self.assertTrue(result.is_valid)
        
        print("✅ Request validation integration working")
    
    @unittest.skipUnless(SECURITY_MODULES_AVAILABLE, "Security modules not available")
    def test_security_logging_integration(self):
        """Test security logging with event types."""
        print("\n🔒 Testing Security Logging Integration...")
        
        # Create security logger
        log_file = os.path.join(self.temp_dir, "security.log")
        logger = SecurityLogger(log_to_file=True, log_file_path=log_file)
        
        # Test different event types
        logger.log_file_upload("test.ipynb", "192.168.1.100", success=True)
        logger.log_security_violation("sql_injection", {"query": "malicious"}, "192.168.1.100")
        logger.log_rate_limit_hit("192.168.1.100", retry_after=60)
        
        # Verify log file was created
        self.assertTrue(os.path.exists(log_file))
        
        # Read and verify log content
        with open(log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("file_upload", log_content)
            self.assertIn("security_violation", log_content)
            self.assertIn("rate_limit_hit", log_content)
        
        # Test event analysis
        events = logger.get_recent_events(limit=10)
        self.assertGreater(len(events), 0)
        
        print("✅ Security logging integration working")
    
    @unittest.skipUnless(SECURITY_MODULES_AVAILABLE, "Security modules not available")
    def test_security_context_managers_integration(self):
        """Test security context managers with full lifecycle."""
        print("\n🔒 Testing Security Context Managers Integration...")
        
        # Create security context
        context = SecurityContext(
            client_ip="192.168.1.100",
            user_agent="test-client",
            operation_type="test_operation",
            risk_level="medium"
        )
        
        temp_files_created = []
        
        # Test full context lifecycle
        with SecurityContextManager(context) as manager:
            # Test rate limiting
            rate_limit_ok = manager.check_rate_limit()
            self.assertTrue(rate_limit_ok)
            
            # Test temp file creation
            temp_file = manager.create_temp_file("test content", ".txt")
            temp_files_created.append(temp_file)
            self.assertTrue(os.path.exists(temp_file))
            
            # Test process tracking
            import subprocess
            process = subprocess.Popen(['echo', 'test'], stdout=subprocess.PIPE)
            manager.add_process(process)
            process.wait()
            
            # Test custom cleanup handler
            cleanup_called = False
            def custom_cleanup():
                nonlocal cleanup_called
                cleanup_called = True
            
            manager.add_cleanup_handler(custom_cleanup)
        
        # Verify cleanup happened
        for temp_file in temp_files_created:
            self.assertFalse(os.path.exists(temp_file))
        
        self.assertTrue(cleanup_called)
        
        print("✅ Security context managers integration working")
    
    @unittest.skipUnless(SECURITY_MODULES_AVAILABLE, "Security modules not available")
    def test_secure_error_handling_integration(self):
        """Test secure error handling with information disclosure prevention."""
        print("\n🔒 Testing Secure Error Handling Integration...")
        
        # Create secure error handler
        handler = SecureErrorHandler(debug_mode=False)
        
        # Test file not found error
        try:
            with open("/nonexistent/file.txt", 'r') as f:
                pass
        except FileNotFoundError as e:
            response = handler.handle_error(e, {"operation": "file_read"})
            
            # Verify secure response
            self.assertFalse("nonexistent" in response.public_message)
            self.assertEqual(response.error_code, "FILE_NOT_FOUND")
            self.assertEqual(response.severity, ErrorSeverity.LOW)
            self.assertEqual(response.category, ErrorCategory.FILE_OPERATION)
            self.assertIsNotNone(response.error_id)
        
        # Test validation error
        validation_response = handler.handle_validation_error(
            "email", "invalid-email", "Invalid email format"
        )
        
        self.assertEqual(validation_response.error_code, "VALIDATION_ERROR")
        self.assertIn("email", validation_response.public_message)
        
        # Test rate limit error
        rate_limit_response = handler.handle_rate_limit_error("192.168.1.100", 60)
        
        self.assertEqual(rate_limit_response.error_code, "RATE_LIMIT_EXCEEDED")
        self.assertEqual(rate_limit_response.http_status, 429)
        
        # Test HTTP response creation
        try:
            raise ValueError("Test error")
        except ValueError as e:
            response_dict, status_code = handler.create_http_response(e)
            
            self.assertIn("error", response_dict)
            self.assertIn("error_code", response_dict)
            self.assertIn("error_id", response_dict)
            self.assertEqual(status_code, 400)
        
        print("✅ Secure error handling integration working")
    
    @unittest.skipUnless(SECURITY_MODULES_AVAILABLE, "Security modules not available")
    def test_end_to_end_security_workflow(self):
        """Test complete end-to-end security workflow."""
        print("\n🔒 Testing End-to-End Security Workflow...")
        
        # Simulate a complete request lifecycle with security
        client_ip = "192.168.1.100"
        user_agent = "test-client"
        
        # Step 1: Create security context
        context = SecurityContext(
            client_ip=client_ip,
            user_agent=user_agent,
            operation_type="notebook_analysis",
            risk_level="medium"
        )
        
        with SecurityContextManager(context) as manager:
            # Step 2: Check rate limiting
            rate_limit_ok = manager.check_rate_limit()
            self.assertTrue(rate_limit_ok)
            
            # Step 3: Validate request
            if manager.validator:
                request_data = {
                    'method': 'POST',
                    'headers': {'Content-Type': 'multipart/form-data'},
                    'files': {'file': {'filename': 'test.ipynb', 'size': 1024}}
                }
                
                result = manager.validator.validate_request(request_data, client_ip)
                self.assertTrue(result.is_valid)
            
            # Step 4: Create secure temp file
            notebook_content = json.dumps({
                "cells": [{"cell_type": "code", "source": ["print('Hello, World!')"]}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 4
            })
            
            temp_file = manager.create_temp_file(notebook_content, ".ipynb")
            self.assertTrue(os.path.exists(temp_file))
            
            # Step 5: Validate file content with sandbox
            if manager.sandbox:
                is_safe, error_msg, sanitized = manager.sandbox.validate_notebook_structure(notebook_content)
                self.assertTrue(is_safe)
                self.assertEqual(error_msg, "")
            
            # Step 6: Process with resource monitoring
            import subprocess
            process = subprocess.Popen(['echo', 'processing'], stdout=subprocess.PIPE)
            manager.add_process(process)
            
            if manager.sandbox:
                resources = manager.sandbox.monitor_process_resources(process)
                self.assertIsInstance(resources, dict)
            
            process.wait()
            
            # Step 7: Log successful operation
            if manager.logger:
                manager.logger.log_file_upload("test.ipynb", client_ip, success=True)
        
        # Step 8: Verify cleanup
        self.assertFalse(os.path.exists(temp_file))
        
        print("✅ End-to-end security workflow working")
    
    def test_security_modules_availability(self):
        """Test that security modules are properly available."""
        print("\n🔒 Testing Security Modules Availability...")
        
        modules_to_test = [
            'analyzer.security_sandbox',
            'analyzer.security_logger',
            'analyzer.rate_limiter',
            'analyzer.request_validator',
            'analyzer.security_context_managers',
            'analyzer.secure_error_handler',
            'analyzer.filesystem_restrictions'
        ]
        
        available_modules = []
        missing_modules = []
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
                available_modules.append(module_name)
            except ImportError:
                missing_modules.append(module_name)
        
        print(f"✅ Available modules: {len(available_modules)}")
        for module in available_modules:
            print(f"   • {module}")
        
        if missing_modules:
            print(f"⚠️  Missing modules: {len(missing_modules)}")
            for module in missing_modules:
                print(f"   • {module}")
        
        # At least core modules should be available
        self.assertGreater(len(available_modules), 0)
        
        print("✅ Security modules availability check complete")
    
    def test_security_configuration_validation(self):
        """Test security configuration validation."""
        print("\n🔒 Testing Security Configuration Validation...")
        
        # Test environment variables
        security_env_vars = [
            'SECURITY_LOG_CONSOLE',
            'SECURITY_LOG_PATH',
            'REQUEST_VALIDATION_RATE_LIMIT',
            'REQUEST_VALIDATION_WINDOW'
        ]
        
        config_status = {}
        for var in security_env_vars:
            config_status[var] = os.environ.get(var, 'not_set')
        
        print("Security configuration status:")
        for var, value in config_status.items():
            print(f"   • {var}: {value}")
        
        # Test default configurations
        if SECURITY_MODULES_AVAILABLE:
            # Test rate limiter config
            config = RateLimitConfig()
            self.assertGreater(config.max_requests_per_minute, 0)
            self.assertGreater(config.max_requests_per_hour, 0)
            
            # Test request validator config
            validator = RequestValidator()
            self.assertIsNotNone(validator.limits)
            
            # Test security sandbox config
            sandbox = SecuritySandbox()
            self.assertGreater(sandbox.max_memory_mb, 0)
            self.assertGreater(sandbox.max_time_seconds, 0)
        
        print("✅ Security configuration validation complete")


def run_comprehensive_security_tests():
    """Run all comprehensive security tests."""
    print("🔒 Running Comprehensive Security Test Suite")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComprehensiveSecurity)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🔒 Comprehensive Security Test Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == '__main__':
    success = run_comprehensive_security_tests()
    sys.exit(0 if success else 1) 