#!/usr/bin/env python3
"""
Secure Error Handler Module

Provides comprehensive secure error handling that prevents information disclosure
while maintaining useful error reporting for debugging and monitoring.
"""

import os
import re
import sys
import traceback
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib
import json

# Optional dependency - graceful fallback
try:
    from .security_logger import get_security_logger, SecurityEventSeverity
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False


class ErrorSeverity(Enum):
    """Error severity levels for secure error handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for secure handling."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    FILE_OPERATION = "file_operation"
    NETWORK = "network"
    SYSTEM = "system"
    SECURITY = "security"
    PROCESSING = "processing"
    CONFIGURATION = "configuration"


@dataclass
class SecureErrorResponse:
    """Secure error response structure."""
    public_message: str
    error_code: str
    severity: ErrorSeverity
    category: ErrorCategory
    http_status: int
    details: Optional[Dict[str, Any]] = None
    internal_message: Optional[str] = None
    error_id: Optional[str] = None


class SecureErrorHandler:
    """
    Comprehensive secure error handler that prevents information disclosure.
    """
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.logger = None
        
        # Initialize security logger if available
        if LOGGING_AVAILABLE:
            try:
                self.logger = get_security_logger()
            except Exception:
                pass
        
        # Define secure error mappings
        self._initialize_error_mappings()
    
    def _initialize_error_mappings(self):
        """Initialize secure error mappings."""
        self._error_mappings = {
            # File operation errors
            'FileNotFoundError': SecureErrorResponse(
                public_message="The requested file could not be found.",
                error_code="FILE_NOT_FOUND",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.FILE_OPERATION,
                http_status=404
            ),
            'PermissionError': SecureErrorResponse(
                public_message="Access denied. Please check your permissions.",
                error_code="ACCESS_DENIED",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.AUTHORIZATION,
                http_status=403
            ),
            'IsADirectoryError': SecureErrorResponse(
                public_message="Invalid file path provided.",
                error_code="INVALID_FILE_PATH",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.FILE_OPERATION,
                http_status=400
            ),
            'OSError': SecureErrorResponse(
                public_message="File system operation failed.",
                error_code="FILE_SYSTEM_ERROR",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM,
                http_status=500
            ),
            
            # Data validation errors
            'JSONDecodeError': SecureErrorResponse(
                public_message="Invalid file format. Please check the file content.",
                error_code="INVALID_JSON",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.VALIDATION,
                http_status=400
            ),
            'UnicodeDecodeError': SecureErrorResponse(
                public_message="File encoding error. Please use UTF-8 encoding.",
                error_code="ENCODING_ERROR",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.VALIDATION,
                http_status=400
            ),
            'ValueError': SecureErrorResponse(
                public_message="Invalid input data provided.",
                error_code="INVALID_INPUT",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.VALIDATION,
                http_status=400
            ),
            'TypeError': SecureErrorResponse(
                public_message="Invalid data type provided.",
                error_code="INVALID_DATA_TYPE",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.VALIDATION,
                http_status=400
            ),
            'SyntaxError': SecureErrorResponse(
                public_message="Invalid syntax in the provided code.",
                error_code="SYNTAX_ERROR",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.VALIDATION,
                http_status=400
            ),
            
            # Network errors
            'ConnectionError': SecureErrorResponse(
                public_message="Network connection failed. Please try again.",
                error_code="CONNECTION_ERROR",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                http_status=503
            ),
            'TimeoutError': SecureErrorResponse(
                public_message="Request timed out. Please try again.",
                error_code="TIMEOUT_ERROR",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                http_status=504
            ),
            'HTTPError': SecureErrorResponse(
                public_message="External service error. Please try again later.",
                error_code="EXTERNAL_SERVICE_ERROR",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                http_status=502
            ),
            
            # Security errors
            'SecurityError': SecureErrorResponse(
                public_message="Security validation failed.",
                error_code="SECURITY_ERROR",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SECURITY,
                http_status=403
            ),
            'AuthenticationError': SecureErrorResponse(
                public_message="Authentication required.",
                error_code="AUTHENTICATION_REQUIRED",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.AUTHENTICATION,
                http_status=401
            ),
            'AuthorizationError': SecureErrorResponse(
                public_message="Access denied.",
                error_code="ACCESS_DENIED",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.AUTHORIZATION,
                http_status=403
            ),
            
            # System errors
            'MemoryError': SecureErrorResponse(
                public_message="System resource limit exceeded.",
                error_code="RESOURCE_LIMIT_EXCEEDED",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SYSTEM,
                http_status=507
            ),
            'ImportError': SecureErrorResponse(
                public_message="System configuration error.",
                error_code="CONFIGURATION_ERROR",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.CONFIGURATION,
                http_status=500
            ),
            'ModuleNotFoundError': SecureErrorResponse(
                public_message="System configuration error.",
                error_code="CONFIGURATION_ERROR",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.CONFIGURATION,
                http_status=500
            ),
        }
        
        # Define sensitive patterns that should never be exposed
        self._sensitive_patterns = [
            r'/[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+/',  # File paths
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',      # Email addresses
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',                          # IP addresses
            r'\b[A-Za-z0-9]{20,}\b',                                 # API keys/tokens
            r'password[=:]\s*[^\s]+',                                # Passwords
            r'secret[=:]\s*[^\s]+',                                  # Secrets
            r'token[=:]\s*[^\s]+',                                   # Tokens
            r'key[=:]\s*[^\s]+',                                     # Keys
            r'\\[a-zA-Z]:\\\\',                                      # Windows paths
            r'/home/[a-zA-Z0-9_\-]+',                               # Home directories
            r'/tmp/[a-zA-Z0-9_\-]+',                                # Temp directories
        ]
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> SecureErrorResponse:
        """
        Handle an error securely, returning sanitized error information.
        
        Args:
            error: The exception to handle
            context: Optional context information
            
        Returns:
            SecureErrorResponse with sanitized error information
        """
        # Generate unique error ID for tracking
        error_id = self._generate_error_id(error, context)
        
        # Get error type name
        error_type = type(error).__name__
        
        # Get base error response
        if error_type in self._error_mappings:
            response = self._error_mappings[error_type]
        else:
            # Default fallback for unknown errors
            response = SecureErrorResponse(
                public_message="An unexpected error occurred. Please try again.",
                error_code="UNKNOWN_ERROR",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM,
                http_status=500
            )
        
        # Create response copy with error ID
        secure_response = SecureErrorResponse(
            public_message=response.public_message,
            error_code=response.error_code,
            severity=response.severity,
            category=response.category,
            http_status=response.http_status,
            error_id=error_id
        )
        
        # Add internal message if in debug mode
        if self.debug_mode:
            secure_response.internal_message = self._sanitize_error_message(str(error))
            secure_response.details = {
                "error_type": error_type,
                "context": context or {}
            }
        
        # Log the error securely
        self._log_error_securely(error, secure_response, context)
        
        return secure_response
    
    def _generate_error_id(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a unique error ID for tracking."""
        # Create hash from error details
        error_data = f"{type(error).__name__}:{str(error)}:{datetime.now().isoformat()}"
        if context:
            error_data += f":{json.dumps(context, sort_keys=True)}"
        
        # Generate short hash for error ID
        error_hash = hashlib.md5(error_data.encode()).hexdigest()[:8]
        return f"ERR_{error_hash.upper()}"
    
    def _sanitize_error_message(self, message: str) -> str:
        """Sanitize error message to remove sensitive information."""
        sanitized = message
        
        # Remove sensitive patterns
        for pattern in self._sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        # Remove common sensitive keywords
        sensitive_keywords = [
            'password', 'secret', 'token', 'key', 'auth', 'credential',
            'session', 'cookie', 'header', 'api_key', 'private'
        ]
        
        for keyword in sensitive_keywords:
            # Replace sensitive values but keep the context
            pattern = rf'({keyword}[=:]\s*)[^\s]+'
            sanitized = re.sub(pattern, r'\1[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _log_error_securely(self, error: Exception, response: SecureErrorResponse, context: Optional[Dict[str, Any]] = None):
        """Log error securely with appropriate detail level."""
        if not self.logger:
            return
        
        try:
            # Prepare log data
            log_data = {
                "error_id": response.error_id,
                "error_type": type(error).__name__,
                "error_code": response.error_code,
                "severity": response.severity.value,
                "category": response.category.value,
                "http_status": response.http_status,
                "context": context or {}
            }
            
            # Add full error details for internal logging
            if self.debug_mode:
                log_data["error_message"] = str(error)
                log_data["traceback"] = traceback.format_exc()
            else:
                log_data["error_message"] = self._sanitize_error_message(str(error))
            
            # Log with appropriate severity
            if response.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                severity = SecurityEventSeverity.ERROR
            elif response.severity == ErrorSeverity.MEDIUM:
                severity = SecurityEventSeverity.WARNING
            else:
                severity = SecurityEventSeverity.INFO
            
            self.logger.log_security_violation(
                violation_type=f"error_{response.category.value}",
                details=log_data,
                severity=severity
            )
            
        except Exception:
            # Don't let logging errors break error handling
            pass
    
    def create_http_response(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], int]:
        """
        Create an HTTP response from an error.
        
        Args:
            error: The exception to handle
            context: Optional context information
            
        Returns:
            (response_dict, http_status_code)
        """
        secure_response = self.handle_error(error, context)
        
        response_dict = {
            "error": secure_response.public_message,
            "error_code": secure_response.error_code,
            "error_id": secure_response.error_id
        }
        
        # Add additional details in debug mode
        if self.debug_mode and secure_response.details:
            response_dict["debug_details"] = secure_response.details
        
        if self.debug_mode and secure_response.internal_message:
            response_dict["internal_message"] = secure_response.internal_message
        
        return response_dict, secure_response.http_status
    
    def create_flask_response(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Create a Flask JSON response from an error.
        
        Args:
            error: The exception to handle
            context: Optional context information
            
        Returns:
            Flask jsonify response with appropriate status code
        """
        try:
            from flask import jsonify
            response_dict, status_code = self.create_http_response(error, context)
            return jsonify(response_dict), status_code
        except ImportError:
            # Fallback if Flask not available
            return self.create_http_response(error, context)
    
    def handle_validation_error(self, field: str, value: Any, reason: str) -> SecureErrorResponse:
        """
        Handle validation errors with specific field information.
        
        Args:
            field: The field that failed validation
            value: The invalid value (will be sanitized)
            reason: The reason for validation failure
            
        Returns:
            SecureErrorResponse for validation error
        """
        # Sanitize the value to prevent information disclosure
        sanitized_value = self._sanitize_value_for_logging(value)
        
        return SecureErrorResponse(
            public_message=f"Validation failed for field '{field}': {reason}",
            error_code="VALIDATION_ERROR",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            http_status=400,
            details={
                "field": field,
                "reason": reason,
                "value": sanitized_value if self.debug_mode else "[REDACTED]"
            },
            error_id=self._generate_error_id(ValueError(reason), {"field": field})
        )
    
    def handle_rate_limit_error(self, client_ip: str, retry_after: int) -> SecureErrorResponse:
        """
        Handle rate limit errors.
        
        Args:
            client_ip: Client IP address
            retry_after: Seconds to wait before retrying
            
        Returns:
            SecureErrorResponse for rate limit error
        """
        return SecureErrorResponse(
            public_message="Rate limit exceeded. Please try again later.",
            error_code="RATE_LIMIT_EXCEEDED",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.RATE_LIMIT,
            http_status=429,
            details={
                "retry_after": retry_after,
                "client_ip": client_ip if self.debug_mode else "[REDACTED]"
            },
            error_id=self._generate_error_id(Exception("Rate limit exceeded"), {"client_ip": client_ip})
        )
    
    def _sanitize_value_for_logging(self, value: Any) -> str:
        """Sanitize a value for safe logging."""
        if isinstance(value, str):
            if len(value) > 100:
                return f"{value[:50]}...[TRUNCATED]...{value[-50:]}"
            return self._sanitize_error_message(value)
        else:
            return str(type(value).__name__)


# Global secure error handler instance
_secure_error_handler = None

def get_secure_error_handler(debug_mode: bool = False) -> SecureErrorHandler:
    """Get the global secure error handler instance."""
    global _secure_error_handler
    if _secure_error_handler is None or _secure_error_handler.debug_mode != debug_mode:
        _secure_error_handler = SecureErrorHandler(debug_mode=debug_mode)
    return _secure_error_handler


def handle_error_securely(error: Exception, context: Optional[Dict[str, Any]] = None, debug_mode: bool = False) -> SecureErrorResponse:
    """
    Convenience function to handle errors securely.
    
    Args:
        error: The exception to handle
        context: Optional context information
        debug_mode: Whether to include debug information
        
    Returns:
        SecureErrorResponse with sanitized error information
    """
    handler = get_secure_error_handler(debug_mode)
    return handler.handle_error(error, context)


def create_secure_flask_response(error: Exception, context: Optional[Dict[str, Any]] = None, debug_mode: bool = False):
    """
    Convenience function to create a secure Flask response from an error.
    
    Args:
        error: The exception to handle
        context: Optional context information
        debug_mode: Whether to include debug information
        
    Returns:
        Flask jsonify response with appropriate status code
    """
    handler = get_secure_error_handler(debug_mode)
    return handler.create_flask_response(error, context)


# Custom exception classes for better error categorization
class ValidationError(Exception):
    """Exception for validation errors."""
    pass

class AuthenticationError(Exception):
    """Exception for authentication errors."""
    pass

class AuthorizationError(Exception):
    """Exception for authorization errors."""
    pass

class SecurityError(Exception):
    """Exception for security-related errors."""
    pass

class RateLimitError(Exception):
    """Exception for rate limiting errors."""
    pass

class ConfigurationError(Exception):
    """Exception for configuration errors."""
    pass 