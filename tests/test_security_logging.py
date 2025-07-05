#!/usr/bin/env python3
"""
Security Logging Test

Tests the security event logging system functionality.
"""

import sys
import os
import tempfile
import json
import time

# Add the analyzer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analyzer'))

def test_security_logger_import():
    """Test that the SecurityLogger can be imported and instantiated."""
    try:
        from security_logger import SecurityLogger, SecurityEventType, SecurityEventSeverity
        
        # Test basic instantiation with a temporary log file
        with tempfile.NamedTemporaryFile(delete=False) as temp_log:
            temp_log_path = temp_log.name
        
        logger = SecurityLogger(
            log_to_file=True,
            log_to_console=False,
            log_file_path=temp_log_path
        )
        
        print("✅ SecurityLogger import and instantiation successful")
        
        # Cleanup
        try:
            os.unlink(temp_log_path)
        except OSError:
            pass
        
        return True
    except Exception as e:
        print(f"❌ SecurityLogger import failed: {e}")
        return False

def test_basic_logging():
    """Test basic security event logging."""
    try:
        from security_logger import SecurityLogger, SecurityEventType, SecurityEventSeverity
        
        # Create a temporary log file
        with tempfile.NamedTemporaryFile(delete=False) as temp_log:
            temp_log_path = temp_log.name
        
        logger = SecurityLogger(
            log_to_file=True,
            log_to_console=False,
            log_file_path=temp_log_path
        )
        
        # Test file upload logging
        logger.log_file_upload(
            file_name="test.ipynb",
            file_size=1024,
            source_ip="127.0.0.1"
        )
        
        # Test security violation logging
        logger.log_security_violation(
            violation_type="malicious_code",
            details={"pattern": "subprocess.call"},
            file_name="bad.ipynb",
            source_ip="192.168.1.100"
        )
        
        # Test file validation logging
        logger.log_file_validation(
            file_name="safe.ipynb",
            is_safe=True,
            source_ip="10.0.0.1"
        )
        
        # Verify log file was created and contains events
        if os.path.exists(temp_log_path):
            with open(temp_log_path, 'r') as f:
                log_content = f.read()
                
            # Check that events were logged
            if "file_upload" in log_content and "security_violation" in log_content:
                print("✅ Basic security logging successful")
                
                # Parse and validate JSON structure
                lines = log_content.strip().split('\n')
                for line in lines:
                    try:
                        event_data = json.loads(line)
                        required_fields = ['event_type', 'severity', 'message', 'timestamp']
                        if all(field in event_data for field in required_fields):
                            continue
                        else:
                            print(f"❌ Log entry missing required fields: {line}")
                            return False
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON in log: {line}")
                        return False
                
                print("✅ All log entries have valid JSON structure")
                success = True
            else:
                print(f"❌ Expected events not found in log: {log_content}")
                success = False
        else:
            print("❌ Log file was not created")
            success = False
        
        # Cleanup
        try:
            os.unlink(temp_log_path)
        except OSError:
            pass
        
        return success
        
    except Exception as e:
        print(f"❌ Basic logging test failed: {e}")
        return False

def test_log_analysis():
    """Test log analysis and summary functionality."""
    try:
        from security_logger import SecurityLogger, SecurityEventType, SecurityEventSeverity
        
        # Create a temporary log file
        with tempfile.NamedTemporaryFile(delete=False) as temp_log:
            temp_log_path = temp_log.name
        
        logger = SecurityLogger(
            log_to_file=True,
            log_to_console=False,
            log_file_path=temp_log_path
        )
        
        # Log several events
        logger.log_file_upload("test1.ipynb", 1024, "127.0.0.1")
        logger.log_file_upload("test2.ipynb", 2048, "127.0.0.1")
        logger.log_security_violation("malicious_code", {"pattern": "eval("}, "bad.ipynb", "192.168.1.100")
        logger.log_file_validation("safe.ipynb", True, source_ip="10.0.0.1")
        
        # Test getting recent events
        recent_events = logger.get_recent_events(hours=1)
        
        if len(recent_events) >= 4:
            print(f"✅ Retrieved {len(recent_events)} recent events")
            
            # Test security summary
            summary = logger.get_security_summary(hours=1)
            
            expected_fields = ['total_events', 'event_types', 'severity_counts', 'violations', 'file_uploads']
            if all(field in summary for field in expected_fields):
                print(f"✅ Security summary generated: {summary['total_events']} events, {summary['violations']} violations")
                success = True
            else:
                print(f"❌ Security summary missing fields: {summary}")
                success = False
        else:
            print(f"❌ Expected at least 4 events, got {len(recent_events)}")
            success = False
        
        # Cleanup
        try:
            os.unlink(temp_log_path)
        except OSError:
            pass
        
        return success
        
    except Exception as e:
        print(f"❌ Log analysis test failed: {e}")
        return False

def test_global_logger():
    """Test the global logger functionality."""
    try:
        from security_logger import get_security_logger, log_security_event, SecurityEventType, SecurityEventSeverity
        
        # Test global logger
        logger = get_security_logger()
        
        if logger is not None:
            print("✅ Global security logger instance created")
            
            # Test convenience function
            log_security_event(
                SecurityEventType.FILE_UPLOAD,
                "Test convenience logging",
                severity=SecurityEventSeverity.INFO,
                source_ip="127.0.0.1"
            )
            
            print("✅ Convenience logging function works")
            return True
        else:
            print("❌ Global logger is None")
            return False
            
    except Exception as e:
        print(f"❌ Global logger test failed: {e}")
        return False

def main():
    """Run all security logging tests."""
    print("🔧 Security Logging Validation")
    print("===============================")
    print()
    
    tests = [
        ("Logger Import and Setup", test_security_logger_import),
        ("Basic Event Logging", test_basic_logging),
        ("Log Analysis and Summary", test_log_analysis),
        ("Global Logger", test_global_logger),
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
        print("🎉 All security logging features are working correctly!")
        print("✅ Security event logging is ready for use")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        print("Some features may not work as expected")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 