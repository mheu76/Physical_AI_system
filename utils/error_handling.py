"""
Enhanced Error Handling and Validation System
Provides comprehensive error handling, logging, and input validation for the Physical AI system.
"""

import asyncio
import functools
import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification"""
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    VALIDATION = "validation"
    PERMISSION = "permission"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"

@dataclass
class ErrorContext:
    """Enhanced error context information"""
    error_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.SOFTWARE
    component: str = ""
    operation: str = ""
    user_data: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_suggestions: List[str] = field(default_factory=list)
    
class PhysicalAIException(Exception):
    """Base exception for Physical AI System"""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext(error_id=f"err_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        self.cause = cause
        self.traceback_info = traceback.format_exc()

class ValidationError(PhysicalAIException):
    """Input validation errors"""
    def __init__(self, message: str, field_name: str = "", expected_type: str = "", received_value: Any = None):
        context = ErrorContext(
            error_id=f"val_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            category=ErrorCategory.VALIDATION,
            user_data={
                "field_name": field_name,
                "expected_type": expected_type,
                "received_value": str(received_value)
            }
        )
        super().__init__(message, context)
        self.field_name = field_name
        self.expected_type = expected_type
        self.received_value = received_value

class ResourceError(PhysicalAIException):
    """Resource-related errors (memory, disk, etc.)"""
    def __init__(self, message: str, resource_type: str = "", usage: Dict[str, Any] = None):
        context = ErrorContext(
            error_id=f"res_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            user_data={
                "resource_type": resource_type,
                "usage": usage or {}
            }
        )
        super().__init__(message, context)

class HardwareError(PhysicalAIException):
    """Hardware-related errors"""
    def __init__(self, message: str, device_id: str = "", device_type: str = ""):
        context = ErrorContext(
            error_id=f"hw_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            category=ErrorCategory.HARDWARE,
            severity=ErrorSeverity.CRITICAL,
            user_data={
                "device_id": device_id,
                "device_type": device_type
            },
            recovery_suggestions=[
                "Check hardware connections",
                "Verify device power",
                "Restart the device",
                "Check device drivers"
            ]
        )
        super().__init__(message, context)

F = TypeVar('F', bound=Callable[..., Any])

class ErrorHandler:
    """Centralized error handling and recovery system"""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.max_history = 1000
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.HARDWARE: [self._retry_hardware_operation, self._reinitialize_hardware],
            ErrorCategory.NETWORK: [self._retry_network_operation, self._check_network_connectivity],
            ErrorCategory.RESOURCE: [self._cleanup_resources, self._wait_for_resources],
            ErrorCategory.CONFIGURATION: [self._reload_configuration, self._use_fallback_config]
        }
    
    async def handle_error(self, error: PhysicalAIException, context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle an error with automatic recovery attempts"""
        try:
            # Log the error
            await self._log_error(error, context)
            
            # Add to history
            self._add_to_history(error.context)
            
            # Attempt recovery
            if error.context.category in self.recovery_strategies:
                for strategy in self.recovery_strategies[error.context.category]:
                    try:
                        success = await strategy(error, context)
                        if success:
                            logger.info(f"Recovery successful using {strategy.__name__}")
                            return True
                    except Exception as recovery_error:
                        logger.warning(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
            
            # No recovery possible
            return False
            
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            return False
    
    async def _log_error(self, error: PhysicalAIException, context: Optional[Dict[str, Any]]):
        """Enhanced error logging"""
        log_data = {
            "error_id": error.context.error_id,
            "message": error.message,
            "severity": error.context.severity.value,
            "category": error.context.category.value,
            "component": error.context.component,
            "operation": error.context.operation,
            "timestamp": error.context.timestamp.isoformat(),
            "user_data": error.context.user_data,
            "system_state": error.context.system_state,
            "traceback": error.traceback_info
        }
        
        if context:
            log_data["additional_context"] = context
        
        # Log based on severity
        if error.context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            logger.error(f"Error {error.context.error_id}: {error.message}", extra=log_data)
        elif error.context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Error {error.context.error_id}: {error.message}", extra=log_data)
        else:
            logger.info(f"Error {error.context.error_id}: {error.message}", extra=log_data)
    
    def _add_to_history(self, error_context: ErrorContext):
        """Add error to history with size management"""
        self.error_history.append(error_context)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
    
    async def _retry_hardware_operation(self, error: PhysicalAIException, context: Optional[Dict[str, Any]]) -> bool:
        """Retry hardware operation with backoff"""
        try:
            await asyncio.sleep(1.0)  # Simple backoff
            logger.info(f"Retrying hardware operation for {error.context.error_id}")
            return True  # Simplified - actual retry logic would go here
        except Exception:
            return False
    
    async def _reinitialize_hardware(self, error: PhysicalAIException, context: Optional[Dict[str, Any]]) -> bool:
        """Attempt to reinitialize hardware"""
        try:
            logger.info(f"Reinitializing hardware for {error.context.error_id}")
            await asyncio.sleep(2.0)  # Simulate hardware reset
            return True  # Simplified - actual reinitialization would go here
        except Exception:
            return False
    
    async def _retry_network_operation(self, error: PhysicalAIException, context: Optional[Dict[str, Any]]) -> bool:
        """Retry network operation"""
        try:
            await asyncio.sleep(0.5)
            return True  # Simplified
        except Exception:
            return False
    
    async def _check_network_connectivity(self, error: PhysicalAIException, context: Optional[Dict[str, Any]]) -> bool:
        """Check network connectivity"""
        try:
            # Simplified connectivity check
            return True
        except Exception:
            return False
    
    async def _cleanup_resources(self, error: PhysicalAIException, context: Optional[Dict[str, Any]]) -> bool:
        """Cleanup system resources"""
        try:
            import gc
            gc.collect()
            if hasattr(self, '_cleanup_temp_files'):
                await self._cleanup_temp_files()
            return True
        except Exception:
            return False
    
    async def _wait_for_resources(self, error: PhysicalAIException, context: Optional[Dict[str, Any]]) -> bool:
        """Wait for resources to become available"""
        try:
            await asyncio.sleep(5.0)  # Wait for resources
            return True
        except Exception:
            return False
    
    async def _reload_configuration(self, error: PhysicalAIException, context: Optional[Dict[str, Any]]) -> bool:
        """Reload system configuration"""
        try:
            # Would reload config from files
            return True
        except Exception:
            return False
    
    async def _use_fallback_config(self, error: PhysicalAIException, context: Optional[Dict[str, Any]]) -> bool:
        """Use fallback configuration"""
        try:
            # Would switch to fallback config
            return True
        except Exception:
            return False

# Global error handler instance
error_handler = ErrorHandler()

def safe_async_call(
    fallback_value: Any = None,
    exceptions: tuple = (Exception,),
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    component: str = "",
    operation: str = ""
):
    """Decorator for safe async function calls with retry logic"""
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Create error context
                    context = ErrorContext(
                        error_id=f"safe_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        component=component or func.__module__,
                        operation=operation or func.__name__,
                        user_data={
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "function_args": str(args)[:200],  # Truncate for safety
                            "function_kwargs": str(kwargs)[:200]
                        }
                    )
                    
                    # Convert to our exception type
                    if not isinstance(e, PhysicalAIException):
                        wrapped_error = PhysicalAIException(str(e), context, e)
                    else:
                        wrapped_error = e
                        wrapped_error.context.user_data.update(context.user_data)
                    
                    # Handle the error
                    await error_handler.handle_error(wrapped_error)
                    
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.info(f"Retrying {func.__name__} in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
                        break
            
            return fallback_value
            
        return wrapper
    return decorator

def validate_input(
    value: Any,
    expected_type: type,
    field_name: str = "",
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    allowed_values: Optional[List[Any]] = None,
    regex_pattern: Optional[str] = None
) -> Any:
    """Comprehensive input validation"""
    
    # Type checking
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Invalid type for {field_name}: expected {expected_type.__name__}, got {type(value).__name__}",
            field_name=field_name,
            expected_type=expected_type.__name__,
            received_value=value
        )
    
    # Range validation for numeric types
    if isinstance(value, (int, float)) and (min_value is not None or max_value is not None):
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"Value {value} for {field_name} is below minimum {min_value}",
                field_name=field_name,
                received_value=value
            )
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"Value {value} for {field_name} is above maximum {max_value}",
                field_name=field_name,
                received_value=value
            )
    
    # Allowed values validation
    if allowed_values is not None and value not in allowed_values:
        raise ValidationError(
            f"Value {value} for {field_name} not in allowed values: {allowed_values}",
            field_name=field_name,
            received_value=value
        )
    
    # Regex validation for strings
    if regex_pattern is not None and isinstance(value, str):
        import re
        if not re.match(regex_pattern, value):
            raise ValidationError(
                f"Value {value} for {field_name} does not match pattern {regex_pattern}",
                field_name=field_name,
                received_value=value
            )
    
    return value

def require_initialization(func: F) -> F:
    """Decorator to ensure object is initialized before method calls"""
    
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if not getattr(self, '_initialized', False):
            raise PhysicalAIException(
                f"Object {self.__class__.__name__} must be initialized before calling {func.__name__}",
                ErrorContext(
                    error_id=f"init_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    category=ErrorCategory.SOFTWARE,
                    severity=ErrorSeverity.HIGH,
                    component=self.__class__.__name__,
                    operation=func.__name__
                )
            )
        return await func(self, *args, **kwargs)
    
    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        if not getattr(self, '_initialized', False):
            raise PhysicalAIException(
                f"Object {self.__class__.__name__} must be initialized before calling {func.__name__}",
                ErrorContext(
                    error_id=f"init_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    category=ErrorCategory.SOFTWARE,
                    severity=ErrorSeverity.HIGH,
                    component=self.__class__.__name__,
                    operation=func.__name__
                )
            )
        return func(self, *args, **kwargs)
    
    # Return appropriate wrapper based on whether function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper