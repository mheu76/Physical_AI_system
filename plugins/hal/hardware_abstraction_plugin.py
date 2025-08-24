# PLUGIN_NAME: HardwareAbstractionPlugin
# PLUGIN_VERSION: 1.0.0
# PLUGIN_DESCRIPTION: Hardware Abstraction Layer Plugin
# PLUGIN_AUTHOR: Physical AI Team
# PLUGIN_CATEGORY: hal
# PLUGIN_DEPENDENCIES: 
# PLUGIN_ENTRY_POINT: HardwareAbstractionPlugin

"""
Hardware Abstraction Layer Plugin
Provides sensor management, actuator control, and hardware monitoring capabilities
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from core import PluginInterface, event_bus, config_manager, mission_broker

# Import the existing hardware abstraction module
try:
    from hardware_abstraction.hal_manager import HardwareManager
except ImportError:
    # Fallback for when the original module is not available
    class HardwareManager:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.initialized = False
            self.sensors = {}
            self.actuators = {}
            self.hardware_status = "operational"
        
        async def initialize(self) -> bool:
            self.initialized = True
            return True
        
        async def read_sensor(self, sensor_id: str) -> Dict[str, Any]:
            # Simulate sensor reading
            return {
                "sensor_id": sensor_id,
                "value": 25.0,
                "unit": "celsius",
                "timestamp": asyncio.get_event_loop().time()
            }
        
        async def control_actuator(self, actuator_id: str, command: str, parameters: Dict[str, Any]) -> bool:
            # Simulate actuator control
            await asyncio.sleep(0.1)
            return True
        
        async def get_hardware_status(self) -> str:
            return "operational"

logger = logging.getLogger(__name__)


class HardwareAbstractionPlugin(PluginInterface):
    """Hardware Abstraction Layer Plugin"""
    
    def __init__(self):
        self.hardware_manager: Optional[HardwareManager] = None
        self.config: Dict[str, Any] = {}
        self.capabilities = [
            "sensor_management",
            "actuator_control",
            "hardware_monitoring",
            "device_management",
            "hardware_diagnostics"
        ]
        self._running = False
        self._hardware_monitoring_task: Optional[asyncio.Task] = None
        self._sensor_readings: Dict[str, List[Dict[str, Any]]] = {}
        self._actuator_states: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the hardware abstraction plugin"""
        try:
            self.config = config
            
            # Initialize the hardware manager
            self.hardware_manager = HardwareManager(config)
            success = await self.hardware_manager.initialize()
            
            if success:
                logger.info("Hardware Abstraction Plugin initialized successfully")
                return True
            else:
                logger.error("Failed to initialize hardware manager")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize hardware abstraction plugin: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the hardware abstraction plugin"""
        try:
            if not self.hardware_manager:
                logger.error("Hardware manager not initialized")
                return False
            
            # Register capabilities with mission broker
            for capability in self.capabilities:
                await event_bus.publish(event_bus.Event(
                    event_type="plugin.capability.register",
                    data={
                        "plugin_name": "HardwareAbstractionPlugin",
                        "capability": capability
                    },
                    source="HardwareAbstractionPlugin"
                ))
            
            # Register mission execution handler
            event_bus.register_handler(
                "mission.execute", 
                self._handle_mission_execute, 
                "HardwareAbstractionPlugin"
            )
            
            # Start hardware monitoring loop
            self._hardware_monitoring_task = asyncio.create_task(self._hardware_monitoring_loop())
            
            self._running = True
            logger.info("Hardware Abstraction Plugin started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start hardware abstraction plugin: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the hardware abstraction plugin"""
        try:
            self._running = False
            
            # Stop hardware monitoring task
            if self._hardware_monitoring_task:
                self._hardware_monitoring_task.cancel()
                try:
                    await self._hardware_monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Unregister capabilities
            for capability in self.capabilities:
                await event_bus.publish(event_bus.Event(
                    event_type="plugin.capability.unregister",
                    data={
                        "plugin_name": "HardwareAbstractionPlugin",
                        "capability": capability
                    },
                    source="HardwareAbstractionPlugin"
                ))
            
            # Unregister event handler
            event_bus.unregister_handler("mission.execute", "HardwareAbstractionPlugin")
            
            logger.info("Hardware Abstraction Plugin stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop hardware abstraction plugin: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        return {
            "name": "Hardware Abstraction Plugin",
            "version": "1.0.0",
            "running": self._running,
            "initialized": self.hardware_manager is not None,
            "capabilities": self.capabilities,
            "hardware_status": self.hardware_manager.hardware_status if self.hardware_manager else "unknown",
            "sensors_count": len(self._sensor_readings),
            "actuators_count": len(self._actuator_states),
            "config": self.config
        }
    
    async def _handle_mission_execute(self, event: event_bus.Event):
        """Handle mission execution events"""
        try:
            mission_data = event.data
            mission_id = mission_data.get("mission_id")
            mission_name = mission_data.get("mission_name")
            parameters = mission_data.get("parameters", {})
            
            # Check if this mission is for us
            if mission_name not in self.capabilities:
                return
            
            logger.info(f"Executing mission {mission_id}: {mission_name}")
            
            # Execute the appropriate capability
            result = None
            error_message = None
            
            try:
                if mission_name == "sensor_management":
                    result = await self._manage_sensor(parameters)
                elif mission_name == "actuator_control":
                    result = await self._control_actuator(parameters)
                elif mission_name == "hardware_monitoring":
                    result = await self._monitor_hardware(parameters)
                elif mission_name == "device_management":
                    result = await self._manage_device(parameters)
                elif mission_name == "hardware_diagnostics":
                    result = await self._run_diagnostics(parameters)
                else:
                    error_message = f"Unknown capability: {mission_name}"
                    
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error executing mission {mission_id}: {e}")
            
            # Publish mission completion event
            await event_bus.publish(event_bus.Event(
                event_type="mission.complete",
                data={
                    "mission_id": mission_id,
                    "result": result,
                    "error_message": error_message
                },
                source="HardwareAbstractionPlugin"
            ))
            
        except Exception as e:
            logger.error(f"Error handling mission execution: {e}")
    
    async def _manage_sensor(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Manage sensor operations"""
        try:
            operation = parameters.get("operation", "")
            sensor_id = parameters.get("sensor_id", "")
            
            if not operation or not sensor_id:
                raise ValueError("Operation and sensor_id are required for sensor management")
            
            if operation == "read":
                # Read sensor data
                sensor_data = await self.hardware_manager.read_sensor(sensor_id)
                
                # Store reading history
                if sensor_id not in self._sensor_readings:
                    self._sensor_readings[sensor_id] = []
                
                self._sensor_readings[sensor_id].append(sensor_data)
                
                # Keep only recent readings (last 100)
                if len(self._sensor_readings[sensor_id]) > 100:
                    self._sensor_readings[sensor_id] = self._sensor_readings[sensor_id][-100:]
                
                return {
                    "operation": operation,
                    "sensor_id": sensor_id,
                    "data": sensor_data,
                    "readings_count": len(self._sensor_readings[sensor_id])
                }
            
            elif operation == "configure":
                # Configure sensor
                config_params = parameters.get("config", {})
                return {
                    "operation": operation,
                    "sensor_id": sensor_id,
                    "config": config_params,
                    "status": "configured"
                }
            
            elif operation == "status":
                # Get sensor status
                readings = self._sensor_readings.get(sensor_id, [])
                return {
                    "operation": operation,
                    "sensor_id": sensor_id,
                    "status": "active" if readings else "inactive",
                    "readings_count": len(readings),
                    "last_reading": readings[-1] if readings else None
                }
            
            else:
                raise ValueError(f"Unknown sensor operation: {operation}")
            
        except Exception as e:
            logger.error(f"Sensor management error: {e}")
            raise
    
    async def _control_actuator(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Control actuator operations"""
        try:
            actuator_id = parameters.get("actuator_id", "")
            command = parameters.get("command", "")
            command_params = parameters.get("parameters", {})
            
            if not actuator_id or not command:
                raise ValueError("Actuator_id and command are required for actuator control")
            
            # Execute actuator command
            success = await self.hardware_manager.control_actuator(actuator_id, command, command_params)
            
            # Update actuator state
            self._actuator_states[actuator_id] = {
                "last_command": command,
                "last_parameters": command_params,
                "last_execution_time": asyncio.get_event_loop().time(),
                "status": "success" if success else "failed"
            }
            
            return {
                "actuator_id": actuator_id,
                "command": command,
                "parameters": command_params,
                "success": success,
                "execution_time": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Actuator control error: {e}")
            raise
    
    async def _monitor_hardware(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor hardware status"""
        try:
            hardware_status = await self.hardware_manager.get_hardware_status()
            
            # Collect sensor status
            sensor_status = {}
            for sensor_id, readings in self._sensor_readings.items():
                sensor_status[sensor_id] = {
                    "active": len(readings) > 0,
                    "readings_count": len(readings),
                    "last_reading": readings[-1] if readings else None
                }
            
            # Collect actuator status
            actuator_status = {}
            for actuator_id, state in self._actuator_states.items():
                actuator_status[actuator_id] = {
                    "last_command": state.get("last_command"),
                    "status": state.get("status"),
                    "last_execution_time": state.get("last_execution_time")
                }
            
            return {
                "hardware_status": hardware_status,
                "sensor_status": sensor_status,
                "actuator_status": actuator_status,
                "monitoring_timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Hardware monitoring error: {e}")
            raise
    
    async def _manage_device(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Manage device operations"""
        try:
            operation = parameters.get("operation", "")
            device_id = parameters.get("device_id", "")
            device_type = parameters.get("device_type", "")
            
            if not operation or not device_id:
                raise ValueError("Operation and device_id are required for device management")
            
            if operation == "register":
                # Register new device
                device_config = parameters.get("config", {})
                return {
                    "operation": operation,
                    "device_id": device_id,
                    "device_type": device_type,
                    "config": device_config,
                    "status": "registered"
                }
            
            elif operation == "unregister":
                # Unregister device
                return {
                    "operation": operation,
                    "device_id": device_id,
                    "status": "unregistered"
                }
            
            elif operation == "configure":
                # Configure device
                config_params = parameters.get("config", {})
                return {
                    "operation": operation,
                    "device_id": device_id,
                    "config": config_params,
                    "status": "configured"
                }
            
            elif operation == "status":
                # Get device status
                return {
                    "operation": operation,
                    "device_id": device_id,
                    "device_type": device_type,
                    "status": "active"
                }
            
            else:
                raise ValueError(f"Unknown device operation: {operation}")
            
        except Exception as e:
            logger.error(f"Device management error: {e}")
            raise
    
    async def _run_diagnostics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run hardware diagnostics"""
        try:
            diagnostic_type = parameters.get("type", "full")
            
            diagnostics_results = {
                "hardware_status": await self.hardware_manager.get_hardware_status(),
                "sensors_health": {},
                "actuators_health": {},
                "overall_health": "good"
            }
            
            # Check sensors health
            for sensor_id, readings in self._sensor_readings.items():
                if readings:
                    # Simple health check based on recent readings
                    recent_readings = readings[-10:]  # Last 10 readings
                    avg_value = sum(r.get("value", 0) for r in recent_readings) / len(recent_readings)
                    
                    diagnostics_results["sensors_health"][sensor_id] = {
                        "status": "healthy",
                        "average_value": avg_value,
                        "readings_count": len(readings)
                    }
                else:
                    diagnostics_results["sensors_health"][sensor_id] = {
                        "status": "no_data",
                        "average_value": 0,
                        "readings_count": 0
                    }
            
            # Check actuators health
            for actuator_id, state in self._actuator_states.items():
                diagnostics_results["actuators_health"][actuator_id] = {
                    "status": state.get("status", "unknown"),
                    "last_command": state.get("last_command"),
                    "last_execution_time": state.get("last_execution_time")
                }
            
            # Determine overall health
            if diagnostics_results["hardware_status"] != "operational":
                diagnostics_results["overall_health"] = "critical"
            elif any(s.get("status") == "no_data" for s in diagnostics_results["sensors_health"].values()):
                diagnostics_results["overall_health"] = "warning"
            
            return {
                "diagnostic_type": diagnostic_type,
                "results": diagnostics_results,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Hardware diagnostics error: {e}")
            raise
    
    async def _hardware_monitoring_loop(self):
        """Continuous hardware monitoring loop"""
        try:
            while self._running:
                try:
                    # Get hardware status
                    hardware_status = await self.hardware_manager.get_hardware_status()
                    
                    # Publish hardware status event
                    await event_bus.publish(event_bus.Event(
                        event_type="hardware.status",
                        data={
                            "status": hardware_status,
                            "timestamp": asyncio.get_event_loop().time()
                        },
                        source="HardwareAbstractionPlugin"
                    ))
                    
                    # Handle hardware issues
                    if hardware_status != "operational":
                        logger.warning(f"Hardware issue detected: {hardware_status}")
                        
                        # Publish hardware alert event
                        await event_bus.publish(event_bus.Event(
                            event_type="hardware.alert",
                            data={
                                "status": hardware_status,
                                "severity": "warning" if hardware_status == "degraded" else "critical",
                                "timestamp": asyncio.get_event_loop().time()
                            },
                            source="HardwareAbstractionPlugin"
                        ))
                    
                    # Periodic sensor readings for monitoring
                    for sensor_id in self._sensor_readings.keys():
                        try:
                            sensor_data = await self.hardware_manager.read_sensor(sensor_id)
                            self._sensor_readings[sensor_id].append(sensor_data)
                            
                            # Keep only recent readings
                            if len(self._sensor_readings[sensor_id]) > 100:
                                self._sensor_readings[sensor_id] = self._sensor_readings[sensor_id][-100:]
                                
                        except Exception as e:
                            logger.error(f"Error reading sensor {sensor_id}: {e}")
                    
                    # Wait before next monitoring cycle
                    await asyncio.sleep(5.0)  # Monitor every 5 seconds
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in hardware monitoring loop: {e}")
                    await asyncio.sleep(10.0)  # Wait longer on error
                    
        except asyncio.CancelledError:
            logger.info("Hardware monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Hardware monitoring loop failed: {e}")


# Plugin entry point
HardwareAbstractionPlugin = HardwareAbstractionPlugin
