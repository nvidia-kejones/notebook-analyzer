#!/usr/bin/env python3
"""
Request Validator Module

Provides comprehensive request validation with security checks, size limits,
content validation, and attack prevention for the notebook analyzer.
"""

import os
import re
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
import ipaddress
from datetime import datetime, timedelta

# Optional dependency - graceful fallback if not available
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    magic = None
    HAS_MAGIC = False


@dataclass
class RequestLimits:
    """Configuration for request size and structure limits."""
    max_file_size_mb: int
    max_url_length: int
    max_json_depth: int
    max_json_keys: int
    max_header_count: int
    max_header_value_length: int
    max_form_fields: int
    max_form_field_length: int
    allowed_content_types: List[str]
    blocked_user_agents: List[str]
    rate_limit_window_seconds: int
    max_requests_per_window: int


@dataclass
class ValidationResult:
    """Result of request validation."""
    is_valid: bool
    error_message: str
    error_code: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    details: Dict[str, Any]
    sanitized_data: Optional[Dict[str, Any]] = None


class RequestValidator:
    """
    Comprehensive request validator with security checks and attack prevention.
    """
    
    def __init__(self, enable_rate_limiting: bool = True):
        self.enable_rate_limiting = enable_rate_limiting
        
        # Initialize request limits
        self.limits = RequestLimits(
            max_file_size_mb=16,  # 16MB max file size
            max_url_length=2048,  # 2KB max URL length
            max_json_depth=10,    # Maximum JSON nesting depth
            max_json_keys=1000,   # Maximum JSON keys
            max_header_count=50,  # Maximum HTTP headers
            max_header_value_length=4096,  # 4KB max header value
            max_form_fields=20,   # Maximum form fields
            max_form_field_length=10240,  # 10KB max form field
            allowed_content_types=[
                'application/json',
                'multipart/form-data',
                'application/x-www-form-urlencoded',
                'text/plain',
                'application/octet-stream'
            ],
            blocked_user_agents=[
                'sqlmap', 'nikto', 'nmap', 'masscan', 'nessus',
                'openvas', 'w3af', 'skipfish', 'burp', 'zap',
                'havij', 'acunetix', 'webscarab', 'paros'
            ],
            rate_limit_window_seconds=60,
            max_requests_per_window=100
        )
        
        # Rate limiting storage (in production, use Redis or database)
        self._rate_limit_storage = {}
        
        # Compile regex patterns for performance
        self._patterns = {
            'sql_injection': re.compile(
                r'(union\s+select|insert\s+into|delete\s+from|drop\s+table|'
                r'script\s*>|javascript:|vbscript:|onload\s*=|onerror\s*=|'
                r'<\s*script|</\s*script|eval\s*\(|expression\s*\()',
                re.IGNORECASE
            ),
            'xss_patterns': re.compile(
                r'(<\s*script|</\s*script|javascript:|vbscript:|onload\s*=|'
                r'onerror\s*=|onclick\s*=|onmouseover\s*=|alert\s*\(|'
                r'document\.cookie|document\.write)',
                re.IGNORECASE
            ),
            'path_traversal': re.compile(
                r'(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c|\.\.%2f|\.\.%5c)',
                re.IGNORECASE
            ),
            'command_injection': re.compile(
                r'(;|\||&|`|\$\(|>\s*&|<\s*&|\|\s*tee|nc\s+-|curl\s+|wget\s+|'
                r'bash\s+-|sh\s+-|python\s+-c|perl\s+-e|ruby\s+-e)',
                re.IGNORECASE
            ),
            'ldap_injection': re.compile(
                r'(\*\)|&\||!\(|\|\(|\)\(|objectclass=|\|\||&&)',
                re.IGNORECASE
            ),
            'nosql_injection': re.compile(
                r'(\$where|\$ne|\$gt|\$lt|\$regex|\$or|\$and|\$in|\$nin)',
                re.IGNORECASE
            )
        }
        
        # Dangerous file extensions
        self._dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.app', '.deb', '.pkg', '.dmg', '.iso', '.msi', '.dll',
            '.so', '.dylib', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1'
        }
        
        # Allowed file extensions for uploads
        self._allowed_extensions = {'.ipynb', '.py', '.txt', '.json', '.md'}
    
    def validate_request(self, request_data: Dict[str, Any], 
                        client_ip: Optional[str] = None) -> ValidationResult:
        """
        Comprehensive request validation.
        
        Args:
            request_data: Dictionary containing request information
            client_ip: Client IP address for rate limiting
            
        Returns:
            ValidationResult with validation status and details
        """
        try:
            # Rate limiting check
            if self.enable_rate_limiting and client_ip:
                rate_limit_result = self._check_rate_limit(client_ip)
                if not rate_limit_result.is_valid:
                    return rate_limit_result
            
            # Validate request structure
            structure_result = self._validate_request_structure(request_data)
            if not structure_result.is_valid:
                return structure_result
            
            # Validate headers
            if 'headers' in request_data:
                headers_result = self._validate_headers(request_data['headers'])
                if not headers_result.is_valid:
                    return headers_result
            
            # Validate content based on type
            if 'content_type' in request_data and 'data' in request_data:
                content_result = self._validate_content(
                    request_data['data'],
                    request_data['content_type']
                )
                if not content_result.is_valid:
                    return content_result
            
            # Validate file uploads
            if 'files' in request_data:
                files_result = self._validate_file_uploads(request_data['files'])
                if not files_result.is_valid:
                    return files_result
            
            # Validate URLs
            if 'url' in request_data:
                url_result = self._validate_url(request_data['url'])
                if not url_result.is_valid:
                    return url_result
            
            # Security pattern scanning
            security_result = self._scan_security_patterns(request_data)
            if not security_result.is_valid:
                return security_result
            
            # All validations passed
            return ValidationResult(
                is_valid=True,
                error_message="",
                error_code="",
                risk_level="low",
                details={"validation": "passed", "checks": "all"},
                sanitized_data=self._sanitize_request_data(request_data)
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}",
                error_code="VALIDATION_ERROR",
                risk_level="high",
                details={"exception": str(e)}
            )
    
    def _check_rate_limit(self, client_ip: str) -> ValidationResult:
        """Check rate limiting for client IP."""
        try:
            current_time = time.time()
            window_start = current_time - self.limits.rate_limit_window_seconds
            
            # Clean old entries
            if client_ip in self._rate_limit_storage:
                self._rate_limit_storage[client_ip] = [
                    timestamp for timestamp in self._rate_limit_storage[client_ip]
                    if timestamp > window_start
                ]
            else:
                self._rate_limit_storage[client_ip] = []
            
            # Check current request count
            request_count = len(self._rate_limit_storage[client_ip])
            
            if request_count >= self.limits.max_requests_per_window:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Rate limit exceeded: {request_count} requests in {self.limits.rate_limit_window_seconds} seconds",
                    error_code="RATE_LIMIT_EXCEEDED",
                    risk_level="medium",
                    details={
                        "client_ip": client_ip,
                        "request_count": request_count,
                        "window_seconds": self.limits.rate_limit_window_seconds
                    }
                )
            
            # Record this request
            self._rate_limit_storage[client_ip].append(current_time)
            
            return ValidationResult(
                is_valid=True,
                error_message="",
                error_code="",
                risk_level="low",
                details={"rate_limit": "passed"}
            )
            
        except Exception as e:
            # Don't fail on rate limiting errors
            return ValidationResult(
                is_valid=True,
                error_message="",
                error_code="",
                risk_level="low",
                details={"rate_limit": "error", "error": str(e)}
            )
    
    def _validate_request_structure(self, request_data: Dict[str, Any]) -> ValidationResult:
        """Validate basic request structure and size limits."""
        
        # Check if request data is too large
        try:
            request_size = len(str(request_data))
            max_request_size = 50 * 1024 * 1024  # 50MB max request
            
            if request_size > max_request_size:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Request too large: {request_size} bytes (max: {max_request_size})",
                    error_code="REQUEST_TOO_LARGE",
                    risk_level="medium",
                    details={"request_size": request_size, "max_size": max_request_size}
                )
        except Exception:
            pass  # Continue with other checks
        
        # Validate JSON depth if JSON data present
        if 'json' in request_data:
            depth_result = self._validate_json_depth(request_data['json'])
            if not depth_result.is_valid:
                return depth_result
        
        return ValidationResult(
            is_valid=True,
            error_message="",
            error_code="",
            risk_level="low",
            details={"structure": "valid"}
        )
    
    def _validate_json_depth(self, json_data: Any, current_depth: int = 0) -> ValidationResult:
        """Validate JSON nesting depth to prevent DoS attacks."""
        if current_depth > self.limits.max_json_depth:
            return ValidationResult(
                is_valid=False,
                error_message=f"JSON nesting too deep: {current_depth} (max: {self.limits.max_json_depth})",
                error_code="JSON_TOO_DEEP",
                risk_level="medium",
                details={"depth": current_depth, "max_depth": self.limits.max_json_depth}
            )
        
        if isinstance(json_data, dict):
            if len(json_data) > self.limits.max_json_keys:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Too many JSON keys: {len(json_data)} (max: {self.limits.max_json_keys})",
                    error_code="TOO_MANY_JSON_KEYS",
                    risk_level="medium",
                    details={"key_count": len(json_data), "max_keys": self.limits.max_json_keys}
                )
            
            for value in json_data.values():
                result = self._validate_json_depth(value, current_depth + 1)
                if not result.is_valid:
                    return result
        
        elif isinstance(json_data, list):
            for item in json_data:
                result = self._validate_json_depth(item, current_depth + 1)
                if not result.is_valid:
                    return result
        
        return ValidationResult(
            is_valid=True,
            error_message="",
            error_code="",
            risk_level="low",
            details={"json_depth": "valid"}
        )
    
    def _validate_headers(self, headers: Dict[str, str]) -> ValidationResult:
        """Validate HTTP headers for security issues."""
        
        # Check header count
        if len(headers) > self.limits.max_header_count:
            return ValidationResult(
                is_valid=False,
                error_message=f"Too many headers: {len(headers)} (max: {self.limits.max_header_count})",
                error_code="TOO_MANY_HEADERS",
                risk_level="medium",
                details={"header_count": len(headers), "max_headers": self.limits.max_header_count}
            )
        
        # Check individual headers
        for header_name, header_value in headers.items():
            # Check header value length
            if len(header_value) > self.limits.max_header_value_length:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Header value too long: {header_name}",
                    error_code="HEADER_VALUE_TOO_LONG",
                    risk_level="medium",
                    details={
                        "header": header_name,
                        "length": len(header_value),
                        "max_length": self.limits.max_header_value_length
                    }
                )
            
            # Check for malicious user agents
            if header_name.lower() == 'user-agent':
                user_agent = header_value.lower()
                for blocked_agent in self.limits.blocked_user_agents:
                    if blocked_agent in user_agent:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Blocked user agent detected: {blocked_agent}",
                            error_code="BLOCKED_USER_AGENT",
                            risk_level="high",
                            details={"user_agent": header_value, "blocked_pattern": blocked_agent}
                        )
            
            # Check for suspicious headers
            suspicious_headers = ['x-forwarded-for', 'x-real-ip', 'x-originating-ip']
            if header_name.lower() in suspicious_headers:
                # Validate IP addresses in forwarded headers
                ip_result = self._validate_forwarded_ips(header_value)
                if not ip_result.is_valid:
                    return ip_result
        
        return ValidationResult(
            is_valid=True,
            error_message="",
            error_code="",
            risk_level="low",
            details={"headers": "valid"}
        )
    
    def _validate_forwarded_ips(self, ip_header: str) -> ValidationResult:
        """Validate IP addresses in forwarded headers."""
        try:
            # Split by comma for multiple IPs
            ips = [ip.strip() for ip in ip_header.split(',')]
            
            for ip in ips:
                try:
                    ip_obj = ipaddress.ip_address(ip)
                    
                    # Check for private/reserved IPs that shouldn't be forwarded
                    if ip_obj.is_private or ip_obj.is_reserved or ip_obj.is_loopback:
                        continue  # These are acceptable
                    
                except ValueError:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Invalid IP address in header: {ip}",
                        error_code="INVALID_FORWARDED_IP",
                        risk_level="medium",
                        details={"invalid_ip": ip, "header_value": ip_header}
                    )
            
            return ValidationResult(
                is_valid=True,
                error_message="",
                error_code="",
                risk_level="low",
                details={"forwarded_ips": "valid"}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Error validating forwarded IPs: {str(e)}",
                error_code="FORWARDED_IP_ERROR",
                risk_level="medium",
                details={"error": str(e)}
            )
    
    def _validate_content(self, data: Any, content_type: str) -> ValidationResult:
        """Validate request content based on content type."""
        
        # Check if content type is allowed
        content_type_main = content_type.split(';')[0].strip().lower()
        if content_type_main not in self.limits.allowed_content_types:
            return ValidationResult(
                is_valid=False,
                error_message=f"Content type not allowed: {content_type_main}",
                error_code="INVALID_CONTENT_TYPE",
                risk_level="medium",
                details={"content_type": content_type_main, "allowed": self.limits.allowed_content_types}
            )
        
        # Validate JSON content
        if content_type_main == 'application/json':
            if isinstance(data, str):
                try:
                    json_data = json.loads(data)
                    return self._validate_json_depth(json_data)
                except json.JSONDecodeError as e:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Invalid JSON format: {str(e)}",
                        error_code="INVALID_JSON",
                        risk_level="medium",
                        details={"json_error": str(e)}
                    )
            elif isinstance(data, (dict, list)):
                return self._validate_json_depth(data)
        
        # Validate form data
        elif content_type_main == 'application/x-www-form-urlencoded':
            if isinstance(data, dict):
                if len(data) > self.limits.max_form_fields:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Too many form fields: {len(data)} (max: {self.limits.max_form_fields})",
                        error_code="TOO_MANY_FORM_FIELDS",
                        risk_level="medium",
                        details={"field_count": len(data), "max_fields": self.limits.max_form_fields}
                    )
                
                for field_name, field_value in data.items():
                    if isinstance(field_value, str) and len(field_value) > self.limits.max_form_field_length:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Form field too long: {field_name}",
                            error_code="FORM_FIELD_TOO_LONG",
                            risk_level="medium",
                            details={
                                "field": field_name,
                                "length": len(field_value),
                                "max_length": self.limits.max_form_field_length
                            }
                        )
        
        return ValidationResult(
            is_valid=True,
            error_message="",
            error_code="",
            risk_level="low",
            details={"content": "valid"}
        )
    
    def _validate_file_uploads(self, files: Dict[str, Any]) -> ValidationResult:
        """Validate file uploads for security."""
        
        for file_key, file_info in files.items():
            # Extract file information
            filename = file_info.get('filename', '')
            file_size = file_info.get('size', 0)
            file_content = file_info.get('content', b'')
            content_type = file_info.get('content_type', '')
            
            # Validate filename
            filename_result = self._validate_filename(filename)
            if not filename_result.is_valid:
                return filename_result
            
            # Validate file size
            max_size_bytes = self.limits.max_file_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File too large: {filename} ({file_size} bytes, max: {max_size_bytes})",
                    error_code="FILE_TOO_LARGE",
                    risk_level="medium",
                    details={
                        "filename": filename,
                        "size": file_size,
                        "max_size": max_size_bytes
                    }
                )
            
            # Validate file content type
            if HAS_MAGIC and file_content:
                detected_type = magic.from_buffer(file_content[:1024], mime=True)
                type_result = self._validate_file_type(filename, content_type, detected_type)
                if not type_result.is_valid:
                    return type_result
            
            # Validate file content for malicious patterns
            if file_content:
                content_result = self._validate_file_content(filename, file_content)
                if not content_result.is_valid:
                    return content_result
        
        return ValidationResult(
            is_valid=True,
            error_message="",
            error_code="",
            risk_level="low",
            details={"files": "valid"}
        )
    
    def _validate_filename(self, filename: str) -> ValidationResult:
        """Validate uploaded filename for security."""
        
        if not filename:
            return ValidationResult(
                is_valid=False,
                error_message="Empty filename",
                error_code="EMPTY_FILENAME",
                risk_level="medium",
                details={"filename": filename}
            )
        
        # Check for path traversal in filename
        if '..' in filename or '/' in filename or '\\' in filename:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid characters in filename: {filename}",
                error_code="INVALID_FILENAME",
                risk_level="high",
                details={"filename": filename, "issue": "path_traversal"}
            )
        
        # Check for dangerous extensions
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in self._dangerous_extensions:
            return ValidationResult(
                is_valid=False,
                error_message=f"Dangerous file extension: {file_ext}",
                error_code="DANGEROUS_FILE_EXTENSION",
                risk_level="high",
                details={"filename": filename, "extension": file_ext}
            )
        
        # Check if extension is allowed
        if file_ext not in self._allowed_extensions:
            return ValidationResult(
                is_valid=False,
                error_message=f"File extension not allowed: {file_ext}",
                error_code="EXTENSION_NOT_ALLOWED",
                risk_level="medium",
                details={
                    "filename": filename,
                    "extension": file_ext,
                    "allowed": list(self._allowed_extensions)
                }
            )
        
        # Check filename length
        if len(filename) > 255:
            return ValidationResult(
                is_valid=False,
                error_message=f"Filename too long: {len(filename)} characters",
                error_code="FILENAME_TOO_LONG",
                risk_level="medium",
                details={"filename": filename, "length": len(filename)}
            )
        
        return ValidationResult(
            is_valid=True,
            error_message="",
            error_code="",
            risk_level="low",
            details={"filename": "valid"}
        )
    
    def _validate_file_type(self, filename: str, declared_type: str, detected_type: str) -> ValidationResult:
        """Validate file type consistency."""
        
        # Map file extensions to expected MIME types
        expected_types = {
            '.ipynb': ['application/json', 'text/plain'],
            '.py': ['text/plain', 'text/x-python'],
            '.txt': ['text/plain'],
            '.json': ['application/json', 'text/plain'],
            '.md': ['text/plain', 'text/markdown']
        }
        
        file_ext = os.path.splitext(filename)[1].lower()
        expected = expected_types.get(file_ext, [])
        
        # Check if detected type matches expected
        if expected and detected_type not in expected:
            # Allow some flexibility for text files
            if detected_type.startswith('text/') and any(exp.startswith('text/') for exp in expected):
                return ValidationResult(
                    is_valid=True,
                    error_message="",
                    error_code="",
                    risk_level="low",
                    details={"file_type": "valid_text_variant"}
                )
            
            return ValidationResult(
                is_valid=False,
                error_message=f"File type mismatch: {filename} (detected: {detected_type}, expected: {expected})",
                error_code="FILE_TYPE_MISMATCH",
                risk_level="high",
                details={
                    "filename": filename,
                    "detected_type": detected_type,
                    "expected_types": expected,
                    "declared_type": declared_type
                }
            )
        
        return ValidationResult(
            is_valid=True,
            error_message="",
            error_code="",
            risk_level="low",
            details={"file_type": "valid"}
        )
    
    def _validate_file_content(self, filename: str, content: bytes) -> ValidationResult:
        """Validate file content for malicious patterns."""
        
        try:
            # Try to decode as text
            text_content = content.decode('utf-8', errors='ignore')
            
            # Check for suspicious patterns in content
            security_result = self._scan_text_for_patterns(text_content, f"file:{filename}")
            if not security_result.is_valid:
                return security_result
            
            # Additional checks for specific file types
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.ipynb':
                # Validate JSON structure for notebooks
                try:
                    notebook_data = json.loads(text_content)
                    if not isinstance(notebook_data, dict) or 'cells' not in notebook_data:
                        return ValidationResult(
                            is_valid=False,
                            error_message="Invalid notebook structure",
                            error_code="INVALID_NOTEBOOK",
                            risk_level="medium",
                            details={"filename": filename, "issue": "structure"}
                        )
                except json.JSONDecodeError:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Invalid JSON in notebook file",
                        error_code="INVALID_NOTEBOOK_JSON",
                        risk_level="medium",
                        details={"filename": filename}
                    )
            
            elif file_ext == '.py':
                # Basic Python syntax validation
                try:
                    import ast
                    ast.parse(text_content)
                except SyntaxError as e:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Python syntax error: {str(e)}",
                        error_code="PYTHON_SYNTAX_ERROR",
                        risk_level="medium",
                        details={"filename": filename, "syntax_error": str(e)}
                    )
            
            return ValidationResult(
                is_valid=True,
                error_message="",
                error_code="",
                risk_level="low",
                details={"file_content": "valid"}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Error validating file content: {str(e)}",
                error_code="FILE_CONTENT_ERROR",
                risk_level="medium",
                details={"filename": filename, "error": str(e)}
            )
    
    def _validate_url(self, url: str) -> ValidationResult:
        """Validate URL for security issues."""
        
        # Check URL length
        if len(url) > self.limits.max_url_length:
            return ValidationResult(
                is_valid=False,
                error_message=f"URL too long: {len(url)} characters (max: {self.limits.max_url_length})",
                error_code="URL_TOO_LONG",
                risk_level="medium",
                details={"url_length": len(url), "max_length": self.limits.max_url_length}
            )
        
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid URL format: {str(e)}",
                error_code="INVALID_URL_FORMAT",
                risk_level="medium",
                details={"url": url, "error": str(e)}
            )
        
        # Check scheme
        allowed_schemes = ['http', 'https']
        if parsed.scheme.lower() not in allowed_schemes:
            return ValidationResult(
                is_valid=False,
                error_message=f"URL scheme not allowed: {parsed.scheme}",
                error_code="INVALID_URL_SCHEME",
                risk_level="high",
                details={"scheme": parsed.scheme, "allowed": allowed_schemes}
            )
        
        # Check for localhost/private IPs (prevent SSRF)
        if parsed.hostname:
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private or ip.is_loopback or ip.is_reserved:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Private/local IP not allowed: {parsed.hostname}",
                        error_code="PRIVATE_IP_NOT_ALLOWED",
                        risk_level="high",
                        details={"hostname": parsed.hostname, "ip": str(ip)}
                    )
            except ValueError:
                # Not an IP address, check for localhost patterns
                hostname_lower = parsed.hostname.lower()
                blocked_hosts = ['localhost', '127.', '0.0.0.0', '::1', 'local', 'internal']
                for blocked in blocked_hosts:
                    if blocked in hostname_lower:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Blocked hostname pattern: {parsed.hostname}",
                            error_code="BLOCKED_HOSTNAME",
                            risk_level="high",
                            details={"hostname": parsed.hostname, "pattern": blocked}
                        )
        
        # Check for malicious patterns in URL
        url_result = self._scan_text_for_patterns(url, "url")
        if not url_result.is_valid:
            return url_result
        
        return ValidationResult(
            is_valid=True,
            error_message="",
            error_code="",
            risk_level="low",
            details={"url": "valid"}
        )
    
    def _scan_security_patterns(self, request_data: Dict[str, Any]) -> ValidationResult:
        """Scan request data for security attack patterns."""
        
        # Convert request data to text for scanning
        text_data = str(request_data)
        
        return self._scan_text_for_patterns(text_data, "request")
    
    def _scan_text_for_patterns(self, text: str, context: str) -> ValidationResult:
        """Scan text for malicious patterns."""
        
        text_lower = text.lower()
        
        # Check each pattern type
        for pattern_name, pattern in self._patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                risk_levels = {
                    'sql_injection': 'critical',
                    'xss_patterns': 'high',
                    'command_injection': 'critical',
                    'path_traversal': 'high',
                    'ldap_injection': 'high',
                    'nosql_injection': 'high'
                }
                
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Security pattern detected: {pattern_name} in {context}",
                    error_code=f"SECURITY_PATTERN_{pattern_name.upper()}",
                    risk_level=risk_levels.get(pattern_name, 'medium'),
                    details={
                        "pattern_type": pattern_name,
                        "context": context,
                        "matches": matches[:5],  # Limit to first 5 matches
                        "match_count": len(matches)
                    }
                )
        
        return ValidationResult(
            is_valid=True,
            error_message="",
            error_code="",
            risk_level="low",
            details={"security_patterns": "clean"}
        )
    
    def _sanitize_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data by removing/escaping dangerous content."""
        
        sanitized = {}
        
        for key, value in request_data.items():
            if isinstance(value, str):
                # Basic HTML entity encoding for XSS prevention
                sanitized_value = (value
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;'))
                sanitized[key] = sanitized_value
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_request_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_request_data(item) if isinstance(item, dict)
                    else str(item).replace('<', '&lt;').replace('>', '&gt;')
                    if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation configuration and statistics."""
        
        return {
            "limits": {
                "max_file_size_mb": self.limits.max_file_size_mb,
                "max_url_length": self.limits.max_url_length,
                "max_json_depth": self.limits.max_json_depth,
                "max_json_keys": self.limits.max_json_keys,
                "allowed_extensions": list(self._allowed_extensions),
                "blocked_extensions": list(self._dangerous_extensions)
            },
            "security_patterns": list(self._patterns.keys()),
            "rate_limiting": {
                "enabled": self.enable_rate_limiting,
                "window_seconds": self.limits.rate_limit_window_seconds,
                "max_requests": self.limits.max_requests_per_window,
                "active_clients": len(self._rate_limit_storage)
            },
            "features": {
                "magic_file_detection": HAS_MAGIC,
                "ip_validation": True,
                "content_type_validation": True,
                "security_pattern_scanning": True
            }
        }


# Global validator instance
_request_validator = None

def get_request_validator(enable_rate_limiting: bool = True) -> RequestValidator:
    """Get the global request validator instance."""
    global _request_validator
    if _request_validator is None:
        _request_validator = RequestValidator(enable_rate_limiting=enable_rate_limiting)
    return _request_validator

def validate_flask_request(request, client_ip: Optional[str] = None) -> ValidationResult:
    """
    Validate a Flask request object.
    
    Args:
        request: Flask request object
        client_ip: Client IP address
        
    Returns:
        ValidationResult
    """
    validator = get_request_validator()
    
    # Extract request data
    request_data = {
        'method': request.method,
        'url': request.url,
        'headers': dict(request.headers),
        'content_type': request.content_type or '',
    }
    
    # Add form data if present
    if request.form:
        request_data['data'] = dict(request.form)
    
    # Add JSON data if present
    if request.is_json:
        try:
            request_data['json'] = request.get_json()
        except Exception:
            pass
    
    # Add file data if present
    if request.files:
        files_data = {}
        for file_key, file_obj in request.files.items():
            if file_obj.filename:
                files_data[file_key] = {
                    'filename': file_obj.filename,
                    'content_type': file_obj.content_type or '',
                    'size': len(file_obj.read()),  # This consumes the stream
                    'content': file_obj.stream.read()  # Read content for validation
                }
                file_obj.stream.seek(0)  # Reset stream for later use
        request_data['files'] = files_data
    
    return validator.validate_request(request_data, client_ip) 