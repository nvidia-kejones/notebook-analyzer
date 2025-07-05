#!/usr/bin/env python3
"""
Security Event Logger

Provides structured logging for security events and audit trails.
This module handles all security-related logging in a consistent format.
"""

import json
import time
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum


class SecurityEventType(Enum):
    """Types of security events that can be logged."""
    FILE_UPLOAD = "file_upload"
    FILE_VALIDATION = "file_validation"
    SECURITY_VIOLATION = "security_violation"
    PROCESS_ISOLATION = "process_isolation"
    RATE_LIMIT_HIT = "rate_limit_hit"
    AUTHENTICATION = "authentication"
    ACCESS_DENIED = "access_denied"
    SYSTEM_ERROR = "system_error"
    RESOURCE_LIMIT = "resource_limit"
    NETWORK_BLOCK = "network_block"


class SecurityEventSeverity(Enum):
    """Severity levels for security events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Structured security event data."""
    event_type: SecurityEventType
    severity: SecurityEventSeverity
    message: str
    timestamp: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    violation_details: Optional[Dict[str, Any]] = None
    process_info: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class SecurityLogger:
    """
    Centralized security event logger with structured output.
    
    Provides consistent logging for all security-related events
    with configurable output formats and destinations.
    """
    
    def __init__(self, log_to_file: bool = True, log_to_console: bool = False,
                 log_file_path: Optional[str] = None, max_log_size_mb: int = 100):
        """
        Initialize the security logger.
        
        Args:
            log_to_file: Whether to log to a file
            log_to_console: Whether to log to console (for development)
            log_file_path: Custom log file path (defaults to /tmp/security.log)
            max_log_size_mb: Maximum log file size before rotation
        """
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        
        # Set up log file path
        if log_file_path:
            self.log_file_path = log_file_path
        else:
            # Use /tmp for temporary storage (works on most systems)
            self.log_file_path = "/tmp/notebook_analyzer_security.log"
        
        # Ensure log directory exists
        if self.log_to_file:
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, mode=0o700)
                except OSError:
                    # If we can't create the directory, fall back to console logging
                    self.log_to_file = False
                    self.log_to_console = True
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
    
    def _format_event(self, event: SecurityEvent) -> str:
        """Format a security event as JSON."""
        event_dict = asdict(event)
        # Convert enums to strings
        event_dict['event_type'] = event.event_type.value
        event_dict['severity'] = event.severity.value
        return json.dumps(event_dict, separators=(',', ':'))
    
    def _write_to_file(self, formatted_event: str):
        """Write event to log file with rotation if needed."""
        try:
            # Check if log rotation is needed
            if os.path.exists(self.log_file_path):
                file_size = os.path.getsize(self.log_file_path)
                if file_size > self.max_log_size_bytes:
                    self._rotate_log_file()
            
            # Append to log file
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(formatted_event + '\n')
                f.flush()  # Ensure immediate write
                
        except (OSError, IOError) as e:
            # If file logging fails, fall back to console
            if self.log_to_console:
                print(f"Security log write failed: {e}", file=sys.stderr)
                print(f"Event: {formatted_event}", file=sys.stderr)
    
    def _rotate_log_file(self):
        """Rotate the log file when it gets too large."""
        try:
            # Simple rotation: move current log to .old
            old_log_path = self.log_file_path + '.old'
            if os.path.exists(old_log_path):
                os.remove(old_log_path)
            os.rename(self.log_file_path, old_log_path)
        except OSError:
            # If rotation fails, just truncate the current file
            try:
                open(self.log_file_path, 'w').close()
            except OSError:
                pass
    
    def _write_to_console(self, formatted_event: str):
        """Write event to console (for development)."""
        print(f"SECURITY: {formatted_event}", file=sys.stderr)
    
    def log_event(self, event: SecurityEvent):
        """
        Log a security event.
        
        Args:
            event: The security event to log
        """
        # Ensure timestamp is set
        if not event.timestamp:
            event.timestamp = self._get_timestamp()
        
        # Format the event
        formatted_event = self._format_event(event)
        
        # Write to configured destinations
        if self.log_to_file:
            self._write_to_file(formatted_event)
        
        if self.log_to_console:
            self._write_to_console(formatted_event)
    
    def log_file_upload(self, file_name: str, file_size: int, source_ip: Optional[str] = None,
                       user_agent: Optional[str] = None, request_id: Optional[str] = None):
        """Log a file upload event."""
        event = SecurityEvent(
            event_type=SecurityEventType.FILE_UPLOAD,
            severity=SecurityEventSeverity.INFO,
            message=f"File uploaded: {file_name} ({file_size} bytes)",
            timestamp=self._get_timestamp(),
            source_ip=source_ip,
            user_agent=user_agent,
            file_name=file_name,
            file_size=file_size,
            request_id=request_id
        )
        self.log_event(event)
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any],
                              file_name: Optional[str] = None, source_ip: Optional[str] = None,
                              severity: SecurityEventSeverity = SecurityEventSeverity.ERROR):
        """Log a security violation."""
        event = SecurityEvent(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            severity=severity,
            message=f"Security violation detected: {violation_type}",
            timestamp=self._get_timestamp(),
            source_ip=source_ip,
            file_name=file_name,
            violation_details=details
        )
        self.log_event(event)
    
    def log_file_validation(self, file_name: str, is_safe: bool, error_msg: Optional[str] = None,
                           source_ip: Optional[str] = None):
        """Log file validation results."""
        severity = SecurityEventSeverity.INFO if is_safe else SecurityEventSeverity.WARNING
        message = f"File validation: {file_name} - {'SAFE' if is_safe else 'BLOCKED'}"
        if error_msg:
            message += f" ({error_msg})"
        
        event = SecurityEvent(
            event_type=SecurityEventType.FILE_VALIDATION,
            severity=severity,
            message=message,
            timestamp=self._get_timestamp(),
            source_ip=source_ip,
            file_name=file_name,
            additional_data={'validation_result': is_safe, 'error_message': error_msg}
        )
        self.log_event(event)
    
    def log_process_isolation(self, command: List[str], process_id: int = None,
                             resource_usage: Dict[str, Any] = None):
        """Log process isolation events."""
        event = SecurityEvent(
            event_type=SecurityEventType.PROCESS_ISOLATION,
            severity=SecurityEventSeverity.INFO,
            message=f"Process isolated: {' '.join(command[:3])}{'...' if len(command) > 3 else ''}",
            timestamp=self._get_timestamp(),
            process_info={
                'command': command,
                'process_id': process_id,
                'resource_usage': resource_usage
            }
        )
        self.log_event(event)
    
    def log_rate_limit_hit(self, source_ip: str, endpoint: str, limit_type: str = "requests"):
        """Log rate limit violations."""
        event = SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_HIT,
            severity=SecurityEventSeverity.WARNING,
            message=f"Rate limit exceeded: {limit_type} from {source_ip} to {endpoint}",
            timestamp=self._get_timestamp(),
            source_ip=source_ip,
            additional_data={'endpoint': endpoint, 'limit_type': limit_type}
        )
        self.log_event(event)
    
    def log_resource_limit(self, resource_type: str, current_value: float, limit_value: float,
                          process_id: int = None):
        """Log resource limit violations."""
        event = SecurityEvent(
            event_type=SecurityEventType.RESOURCE_LIMIT,
            severity=SecurityEventSeverity.WARNING,
            message=f"Resource limit exceeded: {resource_type} {current_value} > {limit_value}",
            timestamp=self._get_timestamp(),
            process_info={'process_id': process_id},
            additional_data={
                'resource_type': resource_type,
                'current_value': current_value,
                'limit_value': limit_value
            }
        )
        self.log_event(event)
    
    def get_recent_events(self, hours: int = 24, event_type: SecurityEventType = None) -> List[Dict]:
        """
        Get recent security events from the log file.
        
        Args:
            hours: Number of hours to look back
            event_type: Filter by specific event type
            
        Returns:
            List of security events as dictionaries
        """
        if not self.log_to_file or not os.path.exists(self.log_file_path):
            return []
        
        events = []
        cutoff_time = time.time() - (hours * 3600)
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        
                        # Parse timestamp and check if within range
                        event_time = datetime.fromisoformat(event_data.get('timestamp', ''))
                        if event_time.timestamp() < cutoff_time:
                            continue
                        
                        # Filter by event type if specified
                        if event_type and event_data.get('event_type') != event_type.value:
                            continue
                        
                        events.append(event_data)
                        
                    except (json.JSONDecodeError, ValueError):
                        continue  # Skip malformed lines
                        
        except (OSError, IOError):
            pass
        
        return events
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get a summary of recent security events.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with security event statistics
        """
        events = self.get_recent_events(hours)
        
        summary = {
            'total_events': len(events),
            'time_period_hours': hours,
            'event_types': {},
            'severity_counts': {},
            'top_source_ips': {},
            'violations': 0,
            'file_uploads': 0
        }
        
        for event in events:
            # Count by event type
            event_type = event.get('event_type', 'unknown')
            summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1
            
            # Count by severity
            severity = event.get('severity', 'unknown')
            summary['severity_counts'][severity] = summary['severity_counts'].get(severity, 0) + 1
            
            # Track source IPs
            source_ip = event.get('source_ip')
            if source_ip:
                summary['top_source_ips'][source_ip] = summary['top_source_ips'].get(source_ip, 0) + 1
            
            # Count specific event types
            if event_type == SecurityEventType.SECURITY_VIOLATION.value:
                summary['violations'] += 1
            elif event_type == SecurityEventType.FILE_UPLOAD.value:
                summary['file_uploads'] += 1
        
        return summary


# Global security logger instance
_security_logger = None

def get_security_logger() -> SecurityLogger:
    """Get the global security logger instance."""
    global _security_logger
    if _security_logger is None:
        # Default configuration - log to file only in production
        log_to_console = os.environ.get('SECURITY_LOG_CONSOLE', '').lower() == 'true'
        log_file_path = os.environ.get('SECURITY_LOG_PATH')
        
        _security_logger = SecurityLogger(
            log_to_file=True,
            log_to_console=log_to_console,
            log_file_path=log_file_path
        )
    return _security_logger


def log_security_event(event_type: SecurityEventType, message: str, 
                      severity: SecurityEventSeverity = SecurityEventSeverity.INFO,
                      **kwargs):
    """
    Convenience function to log a security event.
    
    Args:
        event_type: Type of security event
        message: Event message
        severity: Event severity
        **kwargs: Additional event data
    """
    logger = get_security_logger()
    event = SecurityEvent(
        event_type=event_type,
        severity=severity,
        message=message,
        timestamp=logger._get_timestamp(),
        **kwargs
    )
    logger.log_event(event) 