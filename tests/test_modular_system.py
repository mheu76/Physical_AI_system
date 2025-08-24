"""
Integration Tests for Modular Physical AI System
Tests the functionality and performance of the modular architecture
"""

import asyncio
import pytest
import time
from typing import Dict, Any

from core import (event_bus, plugin_manager, mission_broker, config_manager, 
                 Event, EventPriority, Mission, MissionPriority)


class TestModularSystem:
    """Test suite for the modular system components"""
    
    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Setup and teardown for each test"""
        # Setup
        await config_manager.initialize()
        await event_bus.start()
        await plugin_manager.initialize({})
        await mission_broker.start()
        
        yield
        
        # Teardown
        await mission_broker.stop()
        await event_bus.stop()
        await config_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_event_bus_functionality(self):
        """Test event bus basic functionality"""
        events_received = []
        
        async def test_handler(event: Event):
            events_received.append(event)
        
        # Register handler
        event_bus.register_handler("test.event", test_handler, "test_module")
        
        # Publish event
        test_event = Event(
            event_type="test.event",
            data={"message": "Hello World"},
            source="test"
        )
        
        await event_bus.publish(test_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        assert len(events_received) == 1
        assert events_received[0].data["message"] == "Hello World"
    
    @pytest.mark.asyncio
    async def test_plugin_manager_functionality(self):
        """Test plugin manager functionality"""
        # Test plugin discovery
        discovered_plugins = plugin_manager.loader.discover_plugins()
        assert isinstance(discovered_plugins, list)
        
        # Test plugin loading (with mock plugin)
        mock_plugin_config = {"test": True}
        success = await plugin_manager.load_plugin("TestPlugin", mock_plugin_config)
        # Should fail since TestPlugin doesn't exist, but shouldn't crash
        assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_mission_broker_functionality(self):
        """Test mission broker functionality"""
        # Create test mission
        mission = Mission(
            name="test_mission",
            description="Test mission",
            parameters={"test": True},
            source="test"
        )
        
        # Submit mission
        success = await mission_broker.submit_mission(mission)
        assert success is False  # Should fail since no plugins handle this mission
        
        # Check mission status
        status = await mission_broker.get_mission_status(mission.mission_id)
        assert status is None  # Mission should not be found since it failed to submit
    
    @pytest.mark.asyncio
    async def test_config_manager_functionality(self):
        """Test configuration manager functionality"""
        # Test setting and getting values
        config_manager.set_value("test_section", "test_key", "test_value")
        value = config_manager.get_value("test_section", "test_key")
        assert value == "test_value"
        
        # Test getting non-existent value
        default_value = config_manager.get_value("test_section", "non_existent", "default")
        assert default_value == "default"
        
        # Test getting entire section
        section = config_manager.get_section("test_section")
        assert section["test_key"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_plugin_capability_registration(self):
        """Test plugin capability registration"""
        # Register a test capability
        await event_bus.publish(Event(
            event_type="plugin.capability.register",
            data={
                "plugin_name": "TestPlugin",
                "capability": "test_capability"
            },
            source="test"
        ))
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check if capability was registered
        capabilities = mission_broker.router.get_all_capabilities()
        assert "test_capability" in capabilities
        assert "TestPlugin" in capabilities["test_capability"]
    
    @pytest.mark.asyncio
    async def test_mission_routing(self):
        """Test mission routing functionality"""
        # Register a test capability first
        await event_bus.publish(Event(
            event_type="plugin.capability.register",
            data={
                "plugin_name": "TestPlugin",
                "capability": "test_capability"
            },
            source="test"
        ))
        
        await asyncio.sleep(0.1)
        
        # Create mission that should route to the registered capability
        mission = Mission(
            name="test_capability",
            description="Test mission",
            parameters={"test": True},
            source="test"
        )
        
        # Test routing
        target_plugin = mission_broker.router.route_mission(mission)
        assert target_plugin == "TestPlugin"
    
    @pytest.mark.asyncio
    async def test_mission_queue_priority(self):
        """Test mission queue priority handling"""
        # Create missions with different priorities
        high_priority_mission = Mission(
            name="high_priority",
            priority=MissionPriority.HIGH,
            parameters={},
            source="test"
        )
        
        low_priority_mission = Mission(
            name="low_priority",
            priority=MissionPriority.LOW,
            parameters={},
            source="test"
        )
        
        # Enqueue missions
        await mission_broker.queue.enqueue_mission(high_priority_mission)
        await mission_broker.queue.enqueue_mission(low_priority_mission)
        
        # Dequeue missions and check order
        first_mission = await mission_broker.queue.dequeue_mission()
        second_mission = await mission_broker.queue.dequeue_mission()
        
        # High priority should come first
        assert first_mission.name == "high_priority"
        assert second_mission.name == "low_priority"
    
    @pytest.mark.asyncio
    async def test_event_priority_handling(self):
        """Test event priority handling"""
        events_processed = []
        
        async def low_priority_handler(event: Event):
            events_processed.append(("low", event.event_type))
        
        async def high_priority_handler(event: Event):
            events_processed.append(("high", event.event_type))
        
        # Register handlers with different priorities
        event_bus.register_handler("priority.test", low_priority_handler, "test", EventPriority.LOW)
        event_bus.register_handler("priority.test", high_priority_handler, "test", EventPriority.HIGH)
        
        # Publish event
        test_event = Event(
            event_type="priority.test",
            data={},
            source="test"
        )
        
        await event_bus.publish_sync(test_event)
        
        # High priority should be processed first
        assert len(events_processed) == 2
        assert events_processed[0][0] == "high"
        assert events_processed[1][0] == "low"
    
    @pytest.mark.asyncio
    async def test_system_integration(self):
        """Test full system integration"""
        # Test complete workflow
        events_received = []
        
        async def integration_handler(event: Event):
            events_received.append(event.event_type)
        
        # Register handler for integration test
        event_bus.register_handler("integration.test", integration_handler, "test")
        
        # Simulate a complete workflow
        # 1. Configuration update
        config_manager.set_value("integration", "test_value", "success")
        
        # 2. Event publication
        await event_bus.publish(Event(
            event_type="integration.test",
            data={"step": 1},
            source="test"
        ))
        
        # 3. Mission submission
        mission = Mission(
            name="integration_test",
            parameters={"test": True},
            source="test"
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify events were processed
        assert "integration.test" in events_received
        
        # Verify configuration was set
        value = config_manager.get_value("integration", "test_value")
        assert value == "success"


class TestModularSystemPerformance:
    """Performance tests for the modular system"""
    
    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Setup and teardown for performance tests"""
        await config_manager.initialize()
        await event_bus.start()
        await plugin_manager.initialize({})
        await mission_broker.start()
        
        yield
        
        await mission_broker.stop()
        await event_bus.stop()
        await config_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_event_throughput(self):
        """Test event processing throughput"""
        events_processed = 0
        
        async def throughput_handler(event: Event):
            nonlocal events_processed
            events_processed += 1
        
        # Register handler
        event_bus.register_handler("throughput.test", throughput_handler, "test")
        
        # Measure throughput
        start_time = time.time()
        num_events = 1000
        
        # Publish events
        for i in range(num_events):
            await event_bus.publish(Event(
                event_type="throughput.test",
                data={"index": i},
                source="test"
            ))
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate throughput
        throughput = events_processed / duration
        
        print(f"Event throughput: {throughput:.2f} events/second")
        print(f"Processed {events_processed}/{num_events} events in {duration:.2f} seconds")
        
        # Should process at least 100 events per second
        assert throughput > 100, f"Throughput {throughput} events/sec is too low"
        assert events_processed >= num_events * 0.9, f"Only processed {events_processed}/{num_events} events"
    
    @pytest.mark.asyncio
    async def test_mission_processing_speed(self):
        """Test mission processing speed"""
        missions_completed = 0
        
        async def mission_handler(event: Event):
            nonlocal missions_completed
            missions_completed += 1
            
            # Simulate mission completion
            await event_bus.publish(Event(
                event_type="mission.complete",
                data={
                    "mission_id": event.data.get("mission_id"),
                    "result": {"status": "completed"}
                },
                source="test"
            ))
        
        # Register mission handler
        event_bus.register_handler("mission.execute", mission_handler, "test")
        
        # Register capability
        await event_bus.publish(Event(
            event_type="plugin.capability.register",
            data={
                "plugin_name": "TestPlugin",
                "capability": "test_mission"
            },
            source="test"
        ))
        
        await asyncio.sleep(0.1)
        
        # Measure mission processing speed
        start_time = time.time()
        num_missions = 100
        
        # Submit missions
        for i in range(num_missions):
            mission = Mission(
                name="test_mission",
                parameters={"index": i},
                source="test"
            )
            await mission_broker.submit_mission(mission)
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate processing speed
        speed = missions_completed / duration
        
        print(f"Mission processing speed: {speed:.2f} missions/second")
        print(f"Completed {missions_completed}/{num_missions} missions in {duration:.2f} seconds")
        
        # Should process at least 10 missions per second
        assert speed > 10, f"Processing speed {speed} missions/sec is too low"
        assert missions_completed >= num_missions * 0.8, f"Only completed {missions_completed}/{num_missions} missions"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate load
        events_processed = 0
        
        async def memory_test_handler(event: Event):
            nonlocal events_processed
            events_processed += 1
        
        event_bus.register_handler("memory.test", memory_test_handler, "test")
        
        # Publish many events
        for i in range(1000):
            await event_bus.publish(Event(
                event_type="memory.test",
                data={"index": i, "data": "x" * 100},  # 100 bytes per event
                source="test"
            ))
        
        await asyncio.sleep(1.0)
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (+{memory_increase:.2f}MB)")
        print(f"Events processed: {events_processed}")
        
        # Memory increase should be reasonable (less than 100MB for 1000 events)
        assert memory_increase < 100, f"Memory increase {memory_increase:.2f}MB is too high"
        assert events_processed >= 900, f"Only processed {events_processed}/1000 events"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test system behavior under concurrent operations"""
        results = []
        
        async def concurrent_operation(operation_id: int):
            """Simulate a concurrent operation"""
            try:
                # Publish event
                await event_bus.publish(Event(
                    event_type="concurrent.test",
                    data={"operation_id": operation_id},
                    source="test"
                ))
                
                # Submit mission
                mission = Mission(
                    name="concurrent_mission",
                    parameters={"operation_id": operation_id},
                    source="test"
                )
                success = await mission_broker.submit_mission(mission)
                
                results.append((operation_id, "success" if success else "failed"))
                
            except Exception as e:
                results.append((operation_id, f"error: {e}"))
        
        # Register handlers
        event_bus.register_handler("concurrent.test", lambda e: None, "test")
        
        await event_bus.publish(Event(
            event_type="plugin.capability.register",
            data={
                "plugin_name": "TestPlugin",
                "capability": "concurrent_mission"
            },
            source="test"
        ))
        
        await asyncio.sleep(0.1)
        
        # Run concurrent operations
        start_time = time.time()
        num_operations = 50
        
        tasks = [concurrent_operation(i) for i in range(num_operations)]
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful_operations = sum(1 for _, result in results if result == "success")
        failed_operations = sum(1 for _, result in results if result == "failed")
        error_operations = sum(1 for _, result in results if result.startswith("error"))
        
        print(f"Concurrent operations completed in {duration:.2f} seconds")
        print(f"Successful: {successful_operations}, Failed: {failed_operations}, Errors: {error_operations}")
        
        # Should handle concurrent operations without errors
        assert error_operations == 0, f"Had {error_operations} errors during concurrent operations"
        assert len(results) == num_operations, f"Only completed {len(results)}/{num_operations} operations"
        assert duration < 10, f"Concurrent operations took too long: {duration:.2f} seconds"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
