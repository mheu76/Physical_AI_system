# Modular Physical AI System Guide

## Overview

The Modular Physical AI System is a highly modular, event-driven architecture that transforms the original monolithic system into a flexible, extensible platform. This guide explains how to use, configure, and extend the system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 🎯 Mission Broker                    │
│              (Event-Driven Hub)                     │
├─────────────────────────────────────────────────────┤
│           📦 Plugin Manager & Registry              │
├─────────────────────────────────────────────────────┤
│  🤖 Foundation   🌱 Learning   ⚡ Execution   🔌 HAL  │
│    Plugins       Plugins      Plugins      Plugins  │
├─────────────────────────────────────────────────────┤
│              🔄 Event Bus & Messaging               │
├─────────────────────────────────────────────────────┤
│               📋 Configuration Layer                │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **Event Bus**: Central communication hub for all system components
2. **Plugin Manager**: Handles plugin lifecycle and discovery
3. **Mission Broker**: Manages and routes missions to appropriate plugins
4. **Configuration Manager**: Centralized configuration management
5. **Core Plugins**: Foundation, Learning, Execution, and HAL plugins

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)
- Configuration file (`configs/modular.yaml`)

### Running the System

#### Basic Usage

```bash
# Run the system
python modular_main.py

# Run with demo
python modular_main.py --demo

# Run with custom config
python modular_main.py --config configs/custom.yaml

# Run with debug logging
python modular_main.py --log-level DEBUG
```

#### Programmatic Usage

```python
from modular_main import ModularPhysicalAI

async def main():
    # Create system instance
    system = ModularPhysicalAI()
    
    # Initialize system
    if await system.initialize():
        # Submit a mission
        mission_id = await system.submit_mission("text_generation", {
            "prompt": "Explain robotics",
            "max_length": 200
        })
        
        # Get system status
        status = await system.get_system_status()
        print(f"System status: {status}")
        
        # Start the system
        await system.start()
    else:
        print("Failed to initialize system")

# Run the system
asyncio.run(main())
```

## Plugin Development

### Creating a New Plugin

1. **Create Plugin File Structure**:
```
plugins/
├── my_category/
│   ├── __init__.py
│   └── my_plugin.py
```

2. **Implement Plugin Interface**:

```python
# plugins/my_category/my_plugin.py

# Plugin metadata
# PLUGIN_NAME: MyCustomPlugin
# PLUGIN_VERSION: 1.0.0
# PLUGIN_DESCRIPTION: My custom plugin
# PLUGIN_AUTHOR: Your Name
# PLUGIN_CATEGORY: my_category
# PLUGIN_DEPENDENCIES: 
# PLUGIN_ENTRY_POINT: MyCustomPlugin

from core import PluginInterface, event_bus, config_manager
from typing import Dict, Any

class MyCustomPlugin(PluginInterface):
    def __init__(self):
        self.capabilities = ["my_capability"]
        self._running = False
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        self.config = config
        return True
    
    async def start(self) -> bool:
        """Start the plugin"""
        # Register capabilities
        for capability in self.capabilities:
            await event_bus.publish(event_bus.Event(
                event_type="plugin.capability.register",
                data={
                    "plugin_name": "MyCustomPlugin",
                    "capability": capability
                },
                source="MyCustomPlugin"
            ))
        
        # Register mission handler
        event_bus.register_handler(
            "mission.execute", 
            self._handle_mission_execute, 
            "MyCustomPlugin"
        )
        
        self._running = True
        return True
    
    async def stop(self) -> bool:
        """Stop the plugin"""
        self._running = False
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        return {
            "name": "My Custom Plugin",
            "version": "1.0.0",
            "running": self._running,
            "capabilities": self.capabilities
        }
    
    async def _handle_mission_execute(self, event: event_bus.Event):
        """Handle mission execution"""
        mission_data = event.data
        mission_name = mission_data.get("mission_name")
        
        if mission_name in self.capabilities:
            # Execute capability
            result = await self._execute_my_capability(mission_data.get("parameters", {}))
            
            # Publish completion event
            await event_bus.publish(event_bus.Event(
                event_type="mission.complete",
                data={
                    "mission_id": mission_data.get("mission_id"),
                    "result": result
                },
                source="MyCustomPlugin"
            ))
    
    async def _execute_my_capability(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plugin's capability"""
        return {
            "status": "completed",
            "result": "My capability executed successfully"
        }

# Plugin entry point
MyCustomPlugin = MyCustomPlugin
```

3. **Add Configuration**:

```yaml
# configs/modular.yaml
my_category:
  my_plugin:
    enabled: true
    custom_setting: "value"
```

### Plugin Best Practices

1. **Error Handling**: Always handle exceptions gracefully
2. **Resource Management**: Clean up resources in the `stop()` method
3. **Event Publishing**: Use appropriate event types and data structures
4. **Configuration**: Validate configuration in `initialize()`
5. **Logging**: Use the logging module for debugging and monitoring

## Mission System

### Submitting Missions

```python
# Submit a mission
mission_id = await system.submit_mission("text_generation", {
    "prompt": "Hello, world!",
    "max_length": 100,
    "temperature": 0.7
})

# Check mission status
status = await system.get_mission_status(mission_id)
print(f"Mission status: {status.status}")
```

### Mission Types

#### Text Generation
```python
mission_params = {
    "prompt": "Your prompt here",
    "max_length": 200,
    "temperature": 0.7
}
```

#### Skill Acquisition
```python
mission_params = {
    "skill_name": "object_recognition",
    "experience_value": 5.0,
    "context": "Learning session"
}
```

#### Motion Control
```python
mission_params = {
    "motion_type": "move_to_position",
    "parameters": {
        "target_position": [1.0, 2.0, 0.5],
        "velocity": 0.5
    }
}
```

#### Sensor Management
```python
mission_params = {
    "operation": "read",
    "sensor_id": "temperature_sensor_01"
}
```

## Event System

### Publishing Events

```python
from core import Event, EventPriority

# Publish a simple event
await event_bus.publish(Event(
    event_type="my.event",
    data={"key": "value"},
    source="my_module"
))

# Publish with priority
await event_bus.publish(Event(
    event_type="critical.event",
    data={"alert": "System failure"},
    source="monitoring",
    priority=EventPriority.CRITICAL
))
```

### Handling Events

```python
async def my_event_handler(event: Event):
    print(f"Received event: {event.event_type}")
    print(f"Data: {event.data}")

# Register handler
event_bus.register_handler("my.event", my_event_handler, "my_module")

# Unregister handler
event_bus.unregister_handler("my.event", "my_module")
```

### Event Types

- `mission.submit`: New mission submitted
- `mission.complete`: Mission completed
- `plugin.capability.register`: Plugin capability registered
- `system.health`: System health status
- `config.changed`: Configuration changed

## Configuration Management

### Reading Configuration

```python
from core import config_manager

# Get a specific value
value = config_manager.get_value("foundation_model", "temperature", 0.7)

# Get entire section
foundation_config = config_manager.get_section("foundation_model")

# Set a value
config_manager.set_value("my_section", "my_key", "my_value")
```

### Configuration Structure

```yaml
system:
  name: "Modular Physical AI System"
  version: "3.0.0-modular"
  debug: false

event_bus:
  max_history: 1000
  queue_size: 10000

plugin_manager:
  auto_discovery: true
  plugin_dirs: ["plugins"]

# Plugin-specific configurations
foundation_model:
  model_name: "microsoft/Phi-3.5-mini-instruct"
  temperature: 0.7
  max_length: 2048
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_modular_system.py

# Run with verbose output
pytest tests/ -v

# Run performance tests only
pytest tests/test_modular_system.py::TestModularSystemPerformance -v
```

### Writing Tests

```python
import pytest
from core import event_bus, mission_broker

class TestMyPlugin:
    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        await event_bus.start()
        yield
        await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_my_plugin_functionality(self):
        # Test your plugin functionality
        pass
```

## Performance Optimization

### Event Throughput

- Use appropriate event priorities
- Avoid blocking operations in event handlers
- Consider batching events when possible

### Memory Management

- Clean up resources in plugin `stop()` methods
- Use weak references for long-lived objects
- Monitor memory usage with the built-in metrics

### Mission Processing

- Use appropriate mission priorities
- Implement timeout handling
- Consider parallel mission execution

## Monitoring and Debugging

### System Status

```python
# Get comprehensive system status
status = await system.get_system_status()
print(f"Uptime: {status['system']['uptime']}")
print(f"Missions processed: {status['statistics']['missions_processed']}")
print(f"Plugins loaded: {status['statistics']['plugins_loaded']}")
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Plugin-specific logging
logger = logging.getLogger(__name__)
logger.info("Plugin started")
logger.error("An error occurred")
```

### Health Monitoring

The system automatically publishes health events every 30 seconds:

```python
async def health_monitor(event: Event):
    if event.event_type == "system.health":
        health_data = event.data
        if not health_data["system"]["running"]:
            logger.warning("System health check failed")

event_bus.register_handler("system.health", health_monitor, "monitor")
```

## Security Considerations

### Plugin Security

- Validate all plugin inputs
- Implement proper error handling
- Use sandboxing for untrusted plugins
- Validate plugin signatures

### Event Security

- Validate event data
- Implement rate limiting
- Use authentication for sensitive events
- Monitor for suspicious event patterns

### Configuration Security

- Validate configuration schemas
- Use environment variables for secrets
- Implement configuration encryption
- Audit configuration changes

## Troubleshooting

### Common Issues

1. **Plugin Not Loading**:
   - Check plugin metadata format
   - Verify plugin file structure
   - Check for import errors

2. **Mission Not Executing**:
   - Verify capability registration
   - Check mission routing
   - Review plugin status

3. **Event Not Received**:
   - Check event handler registration
   - Verify event type spelling
   - Check event priority

4. **Configuration Not Applied**:
   - Verify configuration file format
   - Check configuration validation
   - Review configuration reloading

### Debug Mode

Enable debug mode for detailed logging:

```bash
python modular_main.py --log-level DEBUG
```

### Performance Issues

1. **High Memory Usage**:
   - Check for memory leaks in plugins
   - Review event history size
   - Monitor plugin resource usage

2. **Slow Event Processing**:
   - Check event handler performance
   - Review event queue size
   - Consider event batching

3. **Mission Timeouts**:
   - Review mission complexity
   - Check plugin performance
   - Adjust timeout settings

## Extending the System

### Adding New Capabilities

1. Create a new plugin with the capability
2. Register the capability with the mission broker
3. Implement mission handling logic
4. Add configuration options
5. Write tests for the capability

### Custom Event Types

1. Define event type constants
2. Document event data structure
3. Implement event handlers
4. Add event validation
5. Update monitoring and logging

### Integration with External Systems

1. Create integration plugins
2. Implement API clients
3. Handle authentication and security
4. Add error handling and retry logic
5. Monitor integration health

## Best Practices

### Code Organization

- Keep plugins focused and single-purpose
- Use clear naming conventions
- Document all public interfaces
- Follow PEP 8 style guidelines

### Error Handling

- Always handle exceptions gracefully
- Provide meaningful error messages
- Implement proper cleanup on errors
- Log errors with appropriate detail

### Performance

- Profile code for bottlenecks
- Use async/await appropriately
- Implement caching where beneficial
- Monitor resource usage

### Testing

- Write comprehensive unit tests
- Implement integration tests
- Use mocking for external dependencies
- Test error conditions

### Documentation

- Document all public APIs
- Provide usage examples
- Keep documentation up to date
- Include troubleshooting guides

## Conclusion

The Modular Physical AI System provides a flexible, extensible foundation for building sophisticated robotics and AI applications. By following this guide and the established patterns, you can effectively use and extend the system to meet your specific requirements.

For additional support and examples, refer to the test files and existing plugin implementations in the codebase.
