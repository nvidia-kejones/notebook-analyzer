#!/usr/bin/env python3
"""
Simplified Security Context Managers

Provides essential context managers for secure operations with automatic cleanup.
Focuses on functionality over complex type annotations.
"""

import os
import time
import tempfile
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import shutil
import subprocess

# Import our security modules with graceful fallbacks
SANDBOX_AVAILABLE = False
LOGGING_AVAILABLE = False
RATE_LIMITER_AVAILABLE = False
REQUEST_VALIDATOR_AVAILABLE = False

try:
    from .security_sandbox import SecuritySandbox
    SANDBOX_AVAILABLE = True
except ImportError:
    pass

try:
    from .security_logger import get_security_logger, SecurityEventSeverity
    LOGGING_AVAILABLE = True
except ImportError:
    pass

try:
    from .rate_limiter import get_rate_limiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    pass

try:
    from .request_validator import get_request_validator
    REQUEST_VALIDATOR_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SecurityContext:
    """Container for security context information."""
    client_ip: str = ""
    user_agent: str = ""
    operation_type: str = "unknown"
    risk_level: str = "medium"
    start_time: float = 0.0
    
    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()


class SecurityContextManager:
    """
    Simplified security context manager that integrates available security features.
    """
    
    def __init__(self, context: SecurityContext):
        self.context = context
        self.sandbox = None
        self.logger = None
        self.rate_limiter = None
        self.validator = None
        self.temp_files = []
        self.temp_dirs = []
        self.processes = []
        self.cleanup_handlers = []
        
        # Initialize available security components
        self._initialize_security_components()
    
    def _initialize_security_components(self):
        """Initialize available security components."""
        try:
            if SANDBOX_AVAILABLE:
                self.sandbox = SecuritySandbox()
            
            if LOGGING_AVAILABLE:
                self.logger = get_security_logger()
            
            if RATE_LIMITER_AVAILABLE:
                self.rate_limiter = get_rate_limiter()
            
            if REQUEST_VALIDATOR_AVAILABLE:
                self.validator = get_request_validator()
                
        except Exception:
            # Don't fail if security components can't be initialized
            pass
    
    def __enter__(self):
        """Enter the security context."""
        try:
            # Log context start
            if self.logger:
                self.logger.log_security_violation(
                    violation_type="context_start",
                    details={
                        "operation_type": self.context.operation_type,
                        "risk_level": self.context.risk_level
                    },
                    source_ip=self.context.client_ip or None,
                    severity=SecurityEventSeverity.INFO
                )
            
            return self
            
        except Exception as e:
            self._emergency_cleanup()
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the security context with cleanup."""
        try:
            # Calculate operation duration
            duration = time.time() - self.context.start_time
            
            # Log context end
            if self.logger:
                self.logger.log_security_violation(
                    violation_type="context_end",
                    details={
                        "operation_type": self.context.operation_type,
                        "duration_seconds": duration,
                        "success": exc_type is None,
                        "error": str(exc_val) if exc_val else None
                    },
                    source_ip=self.context.client_ip or None,
                    severity=SecurityEventSeverity.INFO if exc_type is None else SecurityEventSeverity.WARNING
                )
            
            # Perform cleanup
            self._cleanup_all_resources()
            
        except Exception as cleanup_error:
            # Don't let cleanup errors mask original exceptions
            if self.logger:
                self.logger.log_security_violation(
                    violation_type="cleanup_error",
                    details={"error": str(cleanup_error)},
                    severity=SecurityEventSeverity.ERROR
                )
    
    def create_temp_file(self, content: str, suffix: str = '.tmp', prefix: str = None) -> str:
        """
        Create a secure temporary file with automatic cleanup.
        
        Args:
            content: Content to write to the file
            suffix: File suffix
            prefix: Optional prefix for the file name
            
        Returns:
            Path to the created file
        """
        if self.sandbox:
            # Use sandbox for secure file creation
            temp_file = self.sandbox.create_secure_temp_file(content, suffix, prefix)
        else:
            # Fallback to basic temp file creation
            temp_file = self._create_basic_temp_file(content, suffix, prefix)
        
        # Track for cleanup
        self.temp_files.append(temp_file)
        
        # Log file creation
        if self.logger:
            self.logger.log_file_upload(
                file_name=os.path.basename(temp_file),
                file_size=len(content),
                source_ip=self.context.client_ip or None
            )
        
        return temp_file
    
    def create_temp_directory(self, prefix: str = None) -> str:
        """
        Create a secure temporary directory with automatic cleanup.
        
        Args:
            prefix: Optional prefix for the directory name
            
        Returns:
            Path to the created directory
        """
        if self.sandbox:
            # Use sandbox for secure directory creation
            temp_dir = self.sandbox.create_secure_temp_directory(prefix)
        else:
            # Fallback to basic temp directory creation
            temp_dir = self._create_basic_temp_dir(prefix)
        
        # Track for cleanup
        self.temp_dirs.append(temp_dir)
        
        return temp_dir
    
    def create_isolated_process(self, command: List[str], working_dir: str = None, 
                              env_vars: Dict[str, str] = None, timeout: int = 300) -> subprocess.Popen:
        """
        Create an isolated process with security monitoring.
        
        Args:
            command: Command and arguments to execute
            working_dir: Working directory for the process
            env_vars: Environment variables
            timeout: Process timeout in seconds
            
        Returns:
            Process object
        """
        if self.sandbox:
            # Use sandbox for process isolation
            process = self.sandbox.create_isolated_process(command, working_dir, env_vars)
        else:
            # Fallback to basic process creation
            process = self._create_basic_process(command, working_dir, env_vars)
        
        # Track for cleanup
        self.processes.append(process)
        
        # Log process creation
        if self.logger:
            self.logger.log_process_isolation(
                command=command,
                process_id=process.pid
            )
        
        # Set up timeout handler
        if timeout > 0:
            self._setup_process_timeout(process, timeout)
        
        return process
    
    def validate_request(self, request_data: Dict[str, Any]) -> bool:
        """
        Validate a request for security issues.
        
        Args:
            request_data: Request data to validate
            
        Returns:
            True if request is valid, False otherwise
        """
        if not self.validator:
            return True  # No validator available, allow request
        
        try:
            result = self.validator.validate_request(request_data, self.context.client_ip)
            
            if not result.is_valid:
                # Log security violation
                if self.logger:
                    self.logger.log_security_violation(
                        violation_type=result.error_code,
                        details={
                            "error_message": result.error_message,
                            "risk_level": result.risk_level,
                            "request_details": result.details
                        },
                        source_ip=self.context.client_ip or None,
                        severity=self._risk_level_to_severity(result.risk_level)
                    )
                
                return False
            
            return True
            
        except Exception as e:
            # Log validation error
            if self.logger:
                self.logger.log_security_violation(
                    violation_type="validation_error",
                    details={"error": str(e)},
                    source_ip=self.context.client_ip or None,
                    severity=SecurityEventSeverity.ERROR
                )
            
            # Fail secure - reject request on validation error
            return False
    
    def check_rate_limit(self) -> bool:
        """
        Check if the current client is within rate limits.
        
        Returns:
            True if within limits, False if rate limited
        """
        if not self.rate_limiter or not self.context.client_ip:
            return True  # No rate limiter or IP, allow request
        
        try:
            status = self.rate_limiter.check_rate_limit(
                self.context.client_ip, 
                self.context.user_agent
            )
            
            if not status.allowed:
                # Log rate limit hit
                if self.logger:
                    self.logger.log_security_violation(
                        violation_type="rate_limit_hit",
                        details={
                            "retry_after": status.retry_after,
                            "remaining_requests": status.remaining_requests
                        },
                        source_ip=self.context.client_ip,
                        severity=SecurityEventSeverity.WARNING
                    )
                
                return False
            
            return True
            
        except Exception as e:
            # Log rate limit error
            if self.logger:
                self.logger.log_security_violation(
                    violation_type="rate_limit_error",
                    details={"error": str(e)},
                    source_ip=self.context.client_ip or None,
                    severity=SecurityEventSeverity.ERROR
                )
            
            # Fail open for rate limiting errors
            return True
    
    def add_cleanup_handler(self, handler: Callable[[], None]):
        """
        Add a custom cleanup handler.
        
        Args:
            handler: Function to call during cleanup
        """
        self.cleanup_handlers.append(handler)
    
    def _create_basic_temp_file(self, content: str, suffix: str, prefix: str) -> str:
        """Create a basic temporary file without sandbox."""
        temp_dir = tempfile.mkdtemp(prefix=prefix or 'secure_')
        temp_file = os.path.join(temp_dir, f'file_{os.urandom(8).hex()}{suffix}')
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Set restrictive permissions
        os.chmod(temp_file, 0o600)
        
        return temp_file
    
    def _create_basic_temp_dir(self, prefix: str) -> str:
        """Create a basic temporary directory without sandbox."""
        temp_dir = tempfile.mkdtemp(prefix=prefix or 'secure_')
        os.chmod(temp_dir, 0o700)
        return temp_dir
    
    def _create_basic_process(self, command: List[str], working_dir: str, env_vars: Dict[str, str]) -> subprocess.Popen:
        """Create a basic process without sandbox isolation."""
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)
        
        return subprocess.Popen(
            command,
            cwd=working_dir,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True  # Basic isolation
        )
    
    def _setup_process_timeout(self, process: subprocess.Popen, timeout: int):
        """Set up a timeout for a process."""
        def timeout_handler():
            time.sleep(timeout)
            if process.poll() is None:  # Process still running
                try:
                    if self.sandbox:
                        self.sandbox.terminate_process_group(process)
                    else:
                        process.terminate()
                        process.wait(timeout=5)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
        
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()
    
    def _risk_level_to_severity(self, risk_level: str):
        """Convert risk level to logging severity."""
        if not LOGGING_AVAILABLE:
            return None
        
        mapping = {
            'low': SecurityEventSeverity.INFO,
            'medium': SecurityEventSeverity.WARNING,
            'high': SecurityEventSeverity.ERROR,
            'critical': SecurityEventSeverity.CRITICAL
        }
        return mapping.get(risk_level, SecurityEventSeverity.WARNING)
    
    def _cleanup_all_resources(self):
        """Clean up all tracked resources."""
        # Run custom cleanup handlers first
        for handler in self.cleanup_handlers:
            try:
                handler()
            except Exception:
                pass
        
        # Terminate processes
        for process in self.processes:
            try:
                if process.poll() is None:  # Process still running
                    if self.sandbox:
                        self.sandbox.terminate_process_group(process)
                    else:
                        process.terminate()
                        process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        
        # Use sandbox cleanup if available
        if self.sandbox:
            try:
                self.sandbox.cleanup_all_temp_files()
            except Exception:
                pass
    
    def _emergency_cleanup(self):
        """Emergency cleanup in case of initialization failure."""
        try:
            self._cleanup_all_resources()
        except Exception:
            # Last resort cleanup
            pass


# Convenience context managers

@contextmanager
def secure_file_operation(client_ip: str = "", operation_type: str = "file_operation"):
    """
    Context manager for secure file operations.
    
    Args:
        client_ip: Client IP address for logging/rate limiting
        operation_type: Type of operation for logging
        
    Yields:
        SecurityContextManager instance
    """
    context = SecurityContext(
        client_ip=client_ip,
        operation_type=operation_type,
        risk_level="medium"
    )
    
    with SecurityContextManager(context) as manager:
        yield manager


@contextmanager
def secure_process_execution(command: List[str], client_ip: str = "", 
                           working_dir: str = None, timeout: int = 300):
    """
    Context manager for secure process execution.
    
    Args:
        command: Command to execute
        client_ip: Client IP address for logging/rate limiting
        working_dir: Working directory for the process
        timeout: Process timeout in seconds
        
    Yields:
        Tuple of (SecurityContextManager, process)
    """
    context = SecurityContext(
        client_ip=client_ip,
        operation_type="process_execution",
        risk_level="high"
    )
    
    with SecurityContextManager(context) as manager:
        process = manager.create_isolated_process(command, working_dir, timeout=timeout)
        yield manager, process


@contextmanager
def secure_request_handling(request_data: Dict[str, Any], client_ip: str = "", 
                          user_agent: str = ""):
    """
    Context manager for secure request handling with validation and rate limiting.
    
    Args:
        request_data: Request data to validate
        client_ip: Client IP address
        user_agent: Client user agent
        
    Yields:
        SecurityContextManager instance
        
    Raises:
        SecurityError: If request validation fails or rate limit exceeded
    """
    context = SecurityContext(
        client_ip=client_ip,
        user_agent=user_agent,
        operation_type="request_handling",
        risk_level="medium"
    )
    
    with SecurityContextManager(context) as manager:
        # Check rate limits first
        if not manager.check_rate_limit():
            raise SecurityError("Rate limit exceeded")
        
        # Validate request
        if not manager.validate_request(request_data):
            raise SecurityError("Request validation failed")
        
        yield manager


@contextmanager
def secure_notebook_analysis(notebook_path: str, client_ip: str = ""):
    """
    Context manager for secure notebook analysis operations.
    
    Args:
        notebook_path: Path to notebook file
        client_ip: Client IP address for logging/rate limiting
        
    Yields:
        SecurityContextManager instance
    """
    context = SecurityContext(
        client_ip=client_ip,
        operation_type="notebook_analysis",
        risk_level="medium"
    )
    
    with SecurityContextManager(context) as manager:
        # Validate notebook file if it's a local path
        if os.path.exists(notebook_path):
            file_size = os.path.getsize(notebook_path)
            request_data = {
                'files': {
                    'notebook': {
                        'filename': os.path.basename(notebook_path),
                        'size': file_size,
                        'content': b'',  # Don't load full content for validation
                        'content_type': 'application/json' if notebook_path.endswith('.ipynb') else 'text/plain'
                    }
                }
            }
            
            if not manager.validate_request(request_data):
                raise SecurityError("Notebook validation failed")
        
        yield manager


class SecurityError(Exception):
    """Exception raised for security-related errors."""
    pass


# Utility functions

def get_current_security_context():
    """
    Get the current security context manager (Flask integration).
    
    Returns:
        SecurityContextManager instance or None
    """
    try:
        from flask import g
        return getattr(g, 'security_manager', None)
    except ImportError:
        return None


def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "info"):
    """
    Log a security event using the current context.
    
    Args:
        event_type: Type of security event
        details: Event details
        severity: Event severity (info, warning, error, critical)
    """
    manager = get_current_security_context()
    if manager and manager.logger:
        severity_map = {
            'info': SecurityEventSeverity.INFO,
            'warning': SecurityEventSeverity.WARNING,
            'error': SecurityEventSeverity.ERROR,
            'critical': SecurityEventSeverity.CRITICAL
        }
        
        manager.logger.log_security_violation(
            violation_type=event_type,
            details=details,
            severity=severity_map.get(severity, SecurityEventSeverity.INFO)
        ) 