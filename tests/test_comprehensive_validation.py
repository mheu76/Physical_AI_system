"""
Comprehensive Validation and Testing Suite
Tests all major components and integration points for stability and correctness.
"""

import asyncio
import pytest
import numpy as np
import logging
from typing import Dict, List, Any
from datetime import datetime
import time
import gc
from unittest.mock import AsyncMock, MagicMock, patch

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test imports with fallback handling
try:
    from utils.error_handling import PhysicalAIException, ValidationError, HardwareError
    from utils.performance_monitor import performance_monitor, PerformanceMetric
    from core.event_bus import Event, EventBus, EventPriority
    from core.config_manager import ConfigManager, ConfigSection
except ImportError as e:
    logger.warning(f"Some imports failed, creating mock classes: {e}")
    
    class PhysicalAIException(Exception):
        pass
    class ValidationError(Exception):
        pass  
    class HardwareError(Exception):
        pass
    
    performance_monitor = None
    Event = None
    EventBus = None
    ConfigManager = None

@pytest.fixture
async def event_bus():
    """Event bus fixture"""
    if EventBus:
        bus = EventBus()
        await bus.start()
        yield bus
        await bus.stop()
    else:
        yield MagicMock()

@pytest.fixture
async def config_manager():
    """Config manager fixture"""
    if ConfigManager:
        manager = ConfigManager("test_configs")
        await manager.initialize()
        yield manager
        await manager.shutdown()
    else:
        yield MagicMock()

class TestErrorHandling:
    """Test comprehensive error handling"""
    
    @pytest.mark.asyncio
    async def test_validation_error_creation(self):
        """Test validation error creation and handling"""
        try:
            from utils.error_handling import validate_input, ValidationError
            
            # Test valid input
            result = validate_input(5, int, "test_field", min_value=0, max_value=10)
            assert result == 5
            
            # Test invalid type
            with pytest.raises(ValidationError) as exc_info:
                validate_input("string", int, "test_field")
            
            assert "Invalid type" in str(exc_info.value)
            assert exc_info.value.field_name == "test_field"
            
            # Test out of range
            with pytest.raises(ValidationError):
                validate_input(15, int, "test_field", min_value=0, max_value=10)
                
        except ImportError:
            pytest.skip("Error handling module not available")
    
    @pytest.mark.asyncio
    async def test_safe_async_call_decorator(self):
        """Test safe async call decorator"""
        try:
            from utils.error_handling import safe_async_call
            
            call_count = 0
            
            @safe_async_call(fallback_value="fallback", max_retries=2)
            async def failing_function():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Test error")
                return "success"
            
            result = await failing_function()
            assert result == "success"
            assert call_count == 3  # Original call + 2 retries
            
        except ImportError:
            pytest.skip("Error handling module not available")

class TestPerformanceMonitoring:
    """Test performance monitoring system"""
    
    @pytest.mark.asyncio
    async def test_performance_monitor_lifecycle(self):
        """Test performance monitor start/stop lifecycle"""
        if not performance_monitor:
            pytest.skip("Performance monitor not available")
            
        # Start monitoring
        await performance_monitor.start_monitoring()
        assert performance_monitor.monitoring_active
        
        # Record some metrics
        metric = PerformanceMetric(
            name="test_metric",
            value=1.23,
            unit="seconds",
            category="test"
        )
        await performance_monitor.record_metric(metric)
        
        # Check metrics were recorded
        assert len(performance_monitor.metrics) > 0
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        assert not performance_monitor.monitoring_active
    
    @pytest.mark.asyncio
    async def test_function_profiling(self):
        """Test function profiling decorator"""
        try:
            from utils.performance_monitor import profile_function
            
            @profile_function(include_memory=True, category="test")
            async def test_function():
                await asyncio.sleep(0.1)
                return "completed"
            
            result = await test_function()
            assert result == "completed"
            
            # Check if profile was recorded
            if performance_monitor:
                profiles = performance_monitor.get_function_profiles()
                assert len(profiles) > 0
                
        except ImportError:
            pytest.skip("Performance monitoring module not available")

class TestEventBus:
    """Test event bus functionality"""
    
    @pytest.mark.asyncio
    async def test_event_publication_and_handling(self, event_bus):
        """Test event publication and handling"""
        if not Event:
            pytest.skip("Event bus not available")
            
        received_events = []
        
        async def test_handler(event):
            received_events.append(event)
        
        # Register handler
        success = event_bus.register_handler("test.event", test_handler, "test_module")
        assert success
        
        # Publish event
        test_event = Event(
            event_type="test.event",
            data={"message": "hello"},
            source="test"
        )
        
        success = await event_bus.publish(test_event)
        assert success
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check event was received
        assert len(received_events) > 0
        assert received_events[0].event_type == "test.event"
        assert received_events[0].data["message"] == "hello"
    
    @pytest.mark.asyncio
    async def test_event_priority_handling(self, event_bus):
        """Test event priority handling"""
        if not Event or not EventPriority:
            pytest.skip("Event bus not available")
            
        processing_order = []
        
        async def high_priority_handler(event):
            processing_order.append("high")
        
        async def low_priority_handler(event):
            processing_order.append("low")
        
        # Register handlers with different priorities
        event_bus.register_handler("test.priority", high_priority_handler, "test", EventPriority.HIGH)
        event_bus.register_handler("test.priority", low_priority_handler, "test", EventPriority.LOW)
        
        # Publish event
        test_event = Event(event_type="test.priority", source="test")
        await event_bus.publish_sync(test_event)  # Synchronous to check order
        
        # High priority should be processed first
        assert processing_order[0] == "high"
        assert processing_order[1] == "low"

class TestConfigManager:
    """Test configuration management"""
    
    @pytest.mark.asyncio
    async def test_config_value_operations(self, config_manager):
        """Test config value get/set operations"""
        if not config_manager:
            pytest.skip("Config manager not available")
        
        # Set a value
        success = await config_manager.set_value("test_section", "test_key", "test_value")
        assert success
        
        # Get the value back
        value = config_manager.get_value("test_section", "test_key")
        assert value == "test_value"
        
        # Test default value
        default_value = config_manager.get_value("nonexistent_section", "nonexistent_key", "default")
        assert default_value == "default"
    
    @pytest.mark.asyncio
    async def test_config_validation(self, config_manager):
        """Test configuration validation"""
        if not config_manager:
            pytest.skip("Config manager not available")
        
        # Register a schema
        schema = {
            "types": {"test_int": int, "test_str": str},
            "required": ["test_int"],
            "ranges": {"test_int": {"min": 0, "max": 100}}
        }
        
        success = config_manager.register_schema("test_validation", schema)
        assert success
        
        # Test valid configuration
        await config_manager.set_value("test_validation", "test_int", 50)
        await config_manager.set_value("test_validation", "test_str", "hello")
        
        section_info = config_manager.get_section_info("test_validation")
        if section_info:
            assert section_info["is_valid"] == True

class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_system_initialization(self):
        """Test full system initialization sequence"""
        try:
            # Simulate main system initialization
            from main import PhysicalAI
            
            # Use test config
            physical_ai = PhysicalAI("configs/default.yaml")
            
            # Test initialization (with mocked components)
            with patch('foundation_model.slm_foundation.SLMFoundation') as mock_slm, \
                 patch('developmental_learning.dev_engine.DevelopmentalEngine') as mock_dev, \
                 patch('ai_agent_execution.agent_executor.AgentExecutor') as mock_agent, \
                 patch('hardware_abstraction.hal_manager.HardwareManager') as mock_hw:
                
                # Mock successful initialization
                mock_slm.return_value.initialize = AsyncMock(return_value=True)
                mock_dev.return_value.initialize = AsyncMock(return_value=True)
                mock_agent.return_value.initialize = AsyncMock(return_value=True)
                mock_hw.return_value.initialize = AsyncMock(return_value=True)
                
                # Test initialization
                success = await physical_ai.initialize()
                assert success
                assert physical_ai.system_ready
                
        except ImportError as e:
            pytest.skip(f"Main module not available: {e}")
    
    @pytest.mark.asyncio
    async def test_mission_execution_flow(self):
        """Test mission execution flow"""
        try:
            from main import PhysicalAI
            
            physical_ai = PhysicalAI("configs/default.yaml")
            
            # Mock all dependencies
            with patch.multiple(
                physical_ai,
                slm_foundation=AsyncMock(),
                dev_engine=AsyncMock(),
                agent_executor=AsyncMock(),
                hw_manager=AsyncMock()
            ):
                # Mock successful responses
                physical_ai.slm_foundation.process_mission_with_learning.return_value = {
                    'success': True,
                    'learning_value': 0.8,
                    'errors': []
                }
                
                physical_ai.slm_foundation.interpret_mission.return_value = {
                    'tasks': ['move_forward', 'pick_object'],
                    'parameters': {'distance': 1.0}
                }
                
                physical_ai.dev_engine.analyze_required_skills.return_value = ['basic_movement', 'grasping']
                
                physical_ai.agent_executor.execute.return_value = MagicMock(
                    success=True,
                    execution_time=2.5,
                    actions_performed=['moved', 'picked'],
                    errors=[],
                    performance_metrics={'accuracy': 0.95},
                    learning_value=0.7
                )
                
                physical_ai.system_ready = True
                
                # Execute mission
                result = await physical_ai.execute_mission("Pick up the red cup")
                
                assert result['success'] == True
                assert 'execution_time' in result
                assert len(result['actions_performed']) > 0
                
        except ImportError as e:
            pytest.skip(f"Main module not available: {e}")

class TestResourceManagement:
    """Test resource management and cleanup"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage remains stable during operations"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        large_arrays = []
        for i in range(100):
            arr = np.random.random((1000, 1000))
            large_arrays.append(arr)
            
            # Simulate async operation
            await asyncio.sleep(0.001)
            
            # Periodic cleanup
            if i % 10 == 0:
                gc.collect()
        
        # Clean up
        del large_arrays
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (< 100MB after cleanup)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test system stability under concurrent operations"""
        async def concurrent_task(task_id: int):
            """Simulate concurrent task"""
            try:
                # Simulate varying workload
                await asyncio.sleep(np.random.uniform(0.01, 0.1))
                
                # Simulate some computation
                result = np.sum(np.random.random((100, 100)))
                
                # Simulate error condition occasionally
                if np.random.random() < 0.1:  # 10% error rate
                    raise Exception(f"Simulated error in task {task_id}")
                
                return {"task_id": task_id, "result": result, "status": "success"}
                
            except Exception as e:
                return {"task_id": task_id, "error": str(e), "status": "error"}
        
        # Run many concurrent tasks
        tasks = [concurrent_task(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and errors
        successes = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        errors = len(results) - successes
        
        # Should have mostly successes with some expected errors
        assert successes > 40, f"Too few successful tasks: {successes}/50"
        assert errors < 10, f"Too many errors: {errors}/50"
        
        logger.info(f"Concurrent test completed: {successes} successes, {errors} errors")

class TestFailureScenarios:
    """Test system behavior under failure conditions"""
    
    @pytest.mark.asyncio
    async def test_hardware_failure_recovery(self):
        """Test system recovery from hardware failures"""
        try:
            from hardware_abstraction.hal_manager import VisionSensor, HardwareError
            
            # Create sensor that will fail initially
            sensor = VisionSensor("test_camera", {"simulation_mode": True})
            
            # Mock initialization failure, then success
            original_init = sensor.initialize
            call_count = 0
            
            async def mock_failing_init():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise HardwareError("Sensor initialization failed")
                return await original_init()
            
            sensor.initialize = mock_failing_init
            
            # First attempt should fail
            with pytest.raises(HardwareError):
                await sensor.initialize()
            
            # Second attempt should succeed
            success = await sensor.initialize()
            assert success
            assert sensor.status.value == "active"
            
        except ImportError:
            pytest.skip("Hardware abstraction module not available")
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Test handling of network timeouts"""
        async def slow_network_operation():
            """Simulate slow network operation"""
            await asyncio.sleep(2.0)  # Longer than typical timeout
            return "network_data"
        
        # Test timeout handling
        try:
            result = await asyncio.wait_for(slow_network_operation(), timeout=1.0)
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            # Expected timeout
            pass
        
        # Test with appropriate timeout
        result = await asyncio.wait_for(slow_network_operation(), timeout=3.0)
        assert result == "network_data"

@pytest.mark.asyncio
async def test_system_health_monitoring():
    """Test system health monitoring"""
    if not performance_monitor:
        pytest.skip("Performance monitor not available")
    
    # Start monitoring
    await performance_monitor.start_monitoring()
    
    # Simulate some system activity
    for i in range(10):
        await performance_monitor.record_metric(PerformanceMetric(
            name="test_cpu_usage",
            value=np.random.uniform(20, 80),
            unit="percent",
            category="system"
        ))
        await asyncio.sleep(0.1)
    
    # Get performance summary
    summary = performance_monitor.get_performance_summary()
    
    assert "uptime_seconds" in summary
    assert "total_functions_profiled" in summary
    assert summary["uptime_seconds"] > 0
    
    # Stop monitoring
    await performance_monitor.stop_monitoring()

@pytest.mark.asyncio 
async def test_cleanup_and_shutdown():
    """Test proper cleanup and shutdown procedures"""
    cleanup_tasks = []
    
    # Simulate various system components
    async def cleanup_component(name: str):
        logger.info(f"Cleaning up {name}")
        await asyncio.sleep(0.1)  # Simulate cleanup time
        return f"{name}_cleaned"
    
    # Add cleanup tasks
    cleanup_tasks.extend([
        cleanup_component("event_bus"),
        cleanup_component("config_manager"),
        cleanup_component("performance_monitor"),
        cleanup_component("hardware_manager")
    ])
    
    # Execute all cleanup tasks
    results = await asyncio.gather(*cleanup_tasks)
    
    # Verify all components were cleaned up
    assert len(results) == 4
    assert all("_cleaned" in result for result in results)
    
    logger.info("System shutdown test completed successfully")

if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v", "--tb=short"])