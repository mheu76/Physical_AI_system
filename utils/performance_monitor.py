"""
Performance Monitoring and Optimization System
Provides comprehensive performance monitoring, profiling, and optimization for the Physical AI system.
"""

import asyncio
import functools
import gc
import logging
import psutil
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar
import threading
import weakref

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    category: str = "general"
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class ResourceUsage:
    """System resource usage snapshot"""
    cpu_percent: float
    memory_percent: float
    memory_used: int  # bytes
    memory_available: int  # bytes
    disk_io_read: int  # bytes
    disk_io_write: int  # bytes
    network_sent: int  # bytes  
    network_recv: int  # bytes
    timestamp: datetime = field(default_factory=datetime.now)
    process_count: int = 0
    thread_count: int = 0

@dataclass
class FunctionProfile:
    """Function execution profile"""
    function_name: str
    module_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_called: Optional[datetime] = None
    error_count: int = 0
    memory_peak: int = 0  # bytes
    
    def update(self, execution_time: float, memory_used: int = 0, had_error: bool = False):
        """Update profile with new execution data"""
        self.total_calls += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.total_calls
        self.last_called = datetime.now()
        self.memory_peak = max(self.memory_peak, memory_used)
        if had_error:
            self.error_count += 1

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, max_metrics: int = 10000, sampling_interval: float = 1.0):
        self.max_metrics = max_metrics
        self.sampling_interval = sampling_interval
        
        # Metrics storage
        self.metrics: deque = deque(maxlen=max_metrics)
        self.resource_history: deque = deque(maxlen=1000)
        self.function_profiles: Dict[str, FunctionProfile] = {}
        
        # Performance tracking
        self.start_time = time.time()
        self.last_gc_time = time.time()
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Thread-safe locks
        self._metrics_lock = threading.RLock()
        self._profiles_lock = threading.RLock()
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'function_time': 5.0,  # seconds
            'error_rate': 0.1  # 10%
        }
        
        # Initialize psutil for system monitoring
        self.process = psutil.Process()
        self._last_io_counters = None
        self._last_net_counters = None
        
    async def start_monitoring(self):
        """Start the performance monitoring loop"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        tracemalloc.start()
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop the performance monitoring loop"""
        self.monitoring_active = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        tracemalloc.stop()
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                resource_usage = await self._collect_resource_usage()
                self.resource_history.append(resource_usage)
                
                # Check for alerts
                await self._check_alerts(resource_usage)
                
                # Garbage collection monitoring
                await self._monitor_garbage_collection()
                
                # Wait for next sampling
                await asyncio.sleep(self.sampling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.sampling_interval)
    
    async def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current system resource usage"""
        try:
            # CPU and Memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # Disk I/O
            io_counters = self.process.io_counters()
            if self._last_io_counters:
                disk_io_read = io_counters.read_bytes - self._last_io_counters.read_bytes
                disk_io_write = io_counters.write_bytes - self._last_io_counters.write_bytes
            else:
                disk_io_read = disk_io_write = 0
            self._last_io_counters = io_counters
            
            # Network I/O (system-wide)
            net_counters = psutil.net_io_counters()
            if self._last_net_counters:
                network_sent = net_counters.bytes_sent - self._last_net_counters.bytes_sent
                network_recv = net_counters.bytes_recv - self._last_net_counters.bytes_recv
            else:
                network_sent = network_recv = 0
            self._last_net_counters = net_counters
            
            # Process and thread counts
            try:
                process_count = len(self.process.children(recursive=True)) + 1
                thread_count = self.process.num_threads()
            except psutil.NoSuchProcess:
                process_count = thread_count = 0
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used=memory_info.rss,
                memory_available=psutil.virtual_memory().available,
                disk_io_read=disk_io_read,
                disk_io_write=disk_io_write,
                network_sent=network_sent,
                network_recv=network_recv,
                process_count=process_count,
                thread_count=thread_count
            )
            
        except Exception as e:
            logger.warning(f"Failed to collect resource usage: {e}")
            return ResourceUsage(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    async def _check_alerts(self, usage: ResourceUsage):
        """Check for performance alerts"""
        alerts = []
        
        if usage.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {usage.cpu_percent:.1f}%")
        
        if usage.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {usage.memory_percent:.1f}%")
        
        # Check function performance
        with self._profiles_lock:
            for profile in self.function_profiles.values():
                if profile.total_calls > 0:
                    error_rate = profile.error_count / profile.total_calls
                    if error_rate > self.alert_thresholds['error_rate']:
                        alerts.append(f"High error rate for {profile.function_name}: {error_rate:.1%}")
                    
                    if profile.max_time > self.alert_thresholds['function_time']:
                        alerts.append(f"Slow function {profile.function_name}: {profile.max_time:.2f}s")
        
        if alerts:
            for alert in alerts:
                logger.warning(f"Performance alert: {alert}")
    
    async def _monitor_garbage_collection(self):
        """Monitor and optimize garbage collection"""
        current_time = time.time()
        
        # Force GC every 60 seconds
        if current_time - self.last_gc_time > 60:
            gc_start = time.time()
            collected = gc.collect()
            gc_time = time.time() - gc_start
            
            self.last_gc_time = current_time
            
            await self.record_metric(PerformanceMetric(
                name="garbage_collection_time",
                value=gc_time,
                unit="seconds",
                category="memory"
            ))
            
            await self.record_metric(PerformanceMetric(
                name="garbage_collection_objects",
                value=collected,
                unit="count",
                category="memory"
            ))
            
            if gc_time > 0.1:  # Log slow GC
                logger.info(f"Garbage collection took {gc_time:.3f}s, collected {collected} objects")
    
    async def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        with self._metrics_lock:
            self.metrics.append(metric)
    
    def get_resource_usage(self, duration_minutes: int = 5) -> List[ResourceUsage]:
        """Get recent resource usage history"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [usage for usage in self.resource_history if usage.timestamp >= cutoff_time]
    
    def get_function_profiles(self, sort_by: str = "avg_time") -> List[FunctionProfile]:
        """Get function performance profiles sorted by specified metric"""
        with self._profiles_lock:
            profiles = list(self.function_profiles.values())
            return sorted(profiles, key=lambda p: getattr(p, sort_by, 0), reverse=True)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_usage = self.resource_history[-1] if self.resource_history else None
        
        # Calculate average resource usage
        recent_usage = self.get_resource_usage(5)
        avg_cpu = sum(u.cpu_percent for u in recent_usage) / len(recent_usage) if recent_usage else 0
        avg_memory = sum(u.memory_percent for u in recent_usage) / len(recent_usage) if recent_usage else 0
        
        # Top slow functions
        slow_functions = self.get_function_profiles("max_time")[:5]
        
        # Error-prone functions
        error_functions = [p for p in self.function_profiles.values() if p.error_count > 0]
        error_functions.sort(key=lambda p: p.error_count / p.total_calls if p.total_calls > 0 else 0, reverse=True)
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "current_resource_usage": current_usage.__dict__ if current_usage else {},
            "average_cpu_percent": avg_cpu,
            "average_memory_percent": avg_memory,
            "total_functions_profiled": len(self.function_profiles),
            "slow_functions": [
                {
                    "name": p.function_name,
                    "avg_time": p.avg_time,
                    "max_time": p.max_time,
                    "total_calls": p.total_calls
                } for p in slow_functions
            ],
            "error_prone_functions": [
                {
                    "name": p.function_name,
                    "error_rate": p.error_count / p.total_calls if p.total_calls > 0 else 0,
                    "error_count": p.error_count,
                    "total_calls": p.total_calls
                } for p in error_functions[:5]
            ],
            "memory_peak_mb": max((u.memory_used for u in recent_usage), default=0) / 1024 / 1024
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Perform automatic performance optimizations"""
        optimizations = []
        
        # Force garbage collection if memory usage is high
        if self.resource_history:
            latest_usage = self.resource_history[-1]
            if latest_usage.memory_percent > 75:
                gc_start = time.time()
                collected = gc.collect()
                gc_time = time.time() - gc_start
                optimizations.append(f"Garbage collection: {collected} objects in {gc_time:.3f}s")
        
        # Clear old metrics
        with self._metrics_lock:
            if len(self.metrics) > self.max_metrics * 0.8:
                remove_count = len(self.metrics) - int(self.max_metrics * 0.5)
                for _ in range(remove_count):
                    self.metrics.popleft()
                optimizations.append(f"Cleared {remove_count} old metrics")
        
        # Reset function profiles for functions with many errors
        with self._profiles_lock:
            reset_functions = []
            for name, profile in list(self.function_profiles.items()):
                if profile.total_calls > 100 and profile.error_count / profile.total_calls > 0.5:
                    reset_functions.append(name)
                    del self.function_profiles[name]
            
            if reset_functions:
                optimizations.append(f"Reset profiles for error-prone functions: {reset_functions}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": optimizations,
            "optimization_count": len(optimizations)
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

F = TypeVar('F', bound=Callable[..., Any])

def profile_function(
    include_memory: bool = False,
    sample_rate: float = 1.0,
    category: str = "function"
):
    """Decorator to profile function performance"""
    
    def decorator(func: F) -> F:
        profile_key = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Sample based on rate
            if sample_rate < 1.0 and time.time() % (1.0 / sample_rate) > 1.0:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            start_memory = 0
            
            if include_memory:
                try:
                    start_memory = tracemalloc.get_traced_memory()[0]
                except:
                    pass
            
            had_error = False
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                had_error = True
                raise
            finally:
                # Record performance data
                execution_time = time.time() - start_time
                memory_used = 0
                
                if include_memory:
                    try:
                        current_memory = tracemalloc.get_traced_memory()[0]
                        memory_used = current_memory - start_memory
                    except:
                        pass
                
                # Update or create profile
                with performance_monitor._profiles_lock:
                    if profile_key not in performance_monitor.function_profiles:
                        performance_monitor.function_profiles[profile_key] = FunctionProfile(
                            function_name=func.__name__,
                            module_name=func.__module__
                        )
                    
                    performance_monitor.function_profiles[profile_key].update(
                        execution_time, memory_used, had_error
                    )
                
                # Record metric
                await performance_monitor.record_metric(PerformanceMetric(
                    name=f"function_execution_time",
                    value=execution_time,
                    unit="seconds",
                    category=category,
                    tags={
                        "function": func.__name__,
                        "module": func.__module__,
                        "had_error": str(had_error)
                    }
                ))
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            start_time = time.time()
            start_memory = 0
            
            if include_memory:
                try:
                    start_memory = tracemalloc.get_traced_memory()[0]
                except:
                    pass
            
            had_error = False
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                had_error = True
                raise
            finally:
                execution_time = time.time() - start_time
                memory_used = 0
                
                if include_memory:
                    try:
                        current_memory = tracemalloc.get_traced_memory()[0]
                        memory_used = current_memory - start_memory
                    except:
                        pass
                
                with performance_monitor._profiles_lock:
                    if profile_key not in performance_monitor.function_profiles:
                        performance_monitor.function_profiles[profile_key] = FunctionProfile(
                            function_name=func.__name__,
                            module_name=func.__module__
                        )
                    
                    performance_monitor.function_profiles[profile_key].update(
                        execution_time, memory_used, had_error
                    )
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

@asynccontextmanager
async def performance_context(operation_name: str, category: str = "operation"):
    """Async context manager for performance monitoring"""
    start_time = time.time()
    start_memory = 0
    
    try:
        start_memory = tracemalloc.get_traced_memory()[0]
    except:
        pass
    
    had_error = False
    try:
        yield
    except Exception as e:
        had_error = True
        raise
    finally:
        execution_time = time.time() - start_time
        memory_used = 0
        
        try:
            current_memory = tracemalloc.get_traced_memory()[0]
            memory_used = current_memory - start_memory
        except:
            pass
        
        await performance_monitor.record_metric(PerformanceMetric(
            name=f"operation_time",
            value=execution_time,
            unit="seconds",
            category=category,
            tags={
                "operation": operation_name,
                "had_error": str(had_error),
                "memory_used": str(memory_used)
            }
        ))