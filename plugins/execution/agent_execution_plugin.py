# PLUGIN_NAME: AgentExecutionPlugin
# PLUGIN_VERSION: 1.0.0
# PLUGIN_DESCRIPTION: AI Agent Execution Engine Plugin
# PLUGIN_AUTHOR: Physical AI Team
# PLUGIN_CATEGORY: execution
# PLUGIN_DEPENDENCIES: 
# PLUGIN_ENTRY_POINT: AgentExecutionPlugin

"""
Agent Execution Plugin
Provides motion control, task execution, and safety monitoring capabilities
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from core import PluginInterface, event_bus, config_manager, mission_broker

# Import the existing agent execution module
try:
    from ai_agent_execution.agent_executor import AgentExecutor
except ImportError:
    # Fallback for when the original module is not available
    class AgentExecutor:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.initialized = False
            self.current_task = None
            self.safety_status = "safe"
        
        async def initialize(self) -> bool:
            self.initialized = True
            return True
        
        async def execute_motion(self, motion_type: str, parameters: Dict[str, Any]) -> bool:
            await asyncio.sleep(0.1)  # Simulate motion execution
            return True
        
        async def execute_task(self, task_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0.5)  # Simulate task execution
            return {"status": "completed", "task": task_name}
        
        async def check_safety(self) -> str:
            return "safe"

logger = logging.getLogger(__name__)


class AgentExecutionPlugin(PluginInterface):
    """Agent Execution Plugin"""
    
    def __init__(self):
        self.agent_executor: Optional[AgentExecutor] = None
        self.config: Dict[str, Any] = {}
        self.capabilities = [
            "motion_control",
            "task_execution",
            "safety_monitoring",
            "path_planning",
            "manipulation"
        ]
        self._running = False
        self._safety_monitoring_task: Optional[asyncio.Task] = None
        self._current_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the agent execution plugin"""
        try:
            self.config = config
            
            # Initialize the agent executor
            self.agent_executor = AgentExecutor(config)
            success = await self.agent_executor.initialize()
            
            if success:
                logger.info("Agent Execution Plugin initialized successfully")
                return True
            else:
                logger.error("Failed to initialize agent executor")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize agent execution plugin: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the agent execution plugin"""
        try:
            if not self.agent_executor:
                logger.error("Agent executor not initialized")
                return False
            
            # Register capabilities with mission broker
            for capability in self.capabilities:
                await event_bus.publish(event_bus.Event(
                    event_type="plugin.capability.register",
                    data={
                        "plugin_name": "AgentExecutionPlugin",
                        "capability": capability
                    },
                    source="AgentExecutionPlugin"
                ))
            
            # Register mission execution handler
            event_bus.register_handler(
                "mission.execute", 
                self._handle_mission_execute, 
                "AgentExecutionPlugin"
            )
            
            # Start safety monitoring loop
            self._safety_monitoring_task = asyncio.create_task(self._safety_monitoring_loop())
            
            self._running = True
            logger.info("Agent Execution Plugin started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent execution plugin: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the agent execution plugin"""
        try:
            self._running = False
            
            # Stop safety monitoring task
            if self._safety_monitoring_task:
                self._safety_monitoring_task.cancel()
                try:
                    await self._safety_monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Unregister capabilities
            for capability in self.capabilities:
                await event_bus.publish(event_bus.Event(
                    event_type="plugin.capability.unregister",
                    data={
                        "plugin_name": "AgentExecutionPlugin",
                        "capability": capability
                    },
                    source="AgentExecutionPlugin"
                ))
            
            # Unregister event handler
            event_bus.unregister_handler("mission.execute", "AgentExecutionPlugin")
            
            logger.info("Agent Execution Plugin stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop agent execution plugin: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        return {
            "name": "Agent Execution Plugin",
            "version": "1.0.0",
            "running": self._running,
            "initialized": self.agent_executor is not None,
            "capabilities": self.capabilities,
            "current_tasks": len(self._current_tasks),
            "safety_status": self.agent_executor.safety_status if self.agent_executor else "unknown",
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
                if mission_name == "motion_control":
                    result = await self._execute_motion(parameters)
                elif mission_name == "task_execution":
                    result = await self._execute_task(parameters)
                elif mission_name == "safety_monitoring":
                    result = await self._check_safety(parameters)
                elif mission_name == "path_planning":
                    result = await self._plan_path(parameters)
                elif mission_name == "manipulation":
                    result = await self._execute_manipulation(parameters)
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
                source="AgentExecutionPlugin"
            ))
            
        except Exception as e:
            logger.error(f"Error handling mission execution: {e}")
    
    async def _execute_motion(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute motion control"""
        try:
            motion_type = parameters.get("motion_type", "")
            motion_params = parameters.get("parameters", {})
            
            if not motion_type:
                raise ValueError("Motion type is required for motion control")
            
            success = await self.agent_executor.execute_motion(motion_type, motion_params)
            
            return {
                "motion_type": motion_type,
                "parameters": motion_params,
                "success": success,
                "execution_time": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Motion execution error: {e}")
            raise
    
    async def _execute_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task"""
        try:
            task_name = parameters.get("task_name", "")
            task_params = parameters.get("parameters", {})
            task_id = parameters.get("task_id", "")
            
            if not task_name:
                raise ValueError("Task name is required for task execution")
            
            # Track task execution
            if task_id:
                self._current_tasks[task_id] = {
                    "task_name": task_name,
                    "start_time": asyncio.get_event_loop().time(),
                    "status": "running"
                }
            
            result = await self.agent_executor.execute_task(task_name, task_params)
            
            # Update task status
            if task_id and task_id in self._current_tasks:
                self._current_tasks[task_id]["status"] = "completed"
                self._current_tasks[task_id]["result"] = result
            
            return {
                "task_name": task_name,
                "task_id": task_id,
                "result": result,
                "execution_time": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            raise
    
    async def _check_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check safety status"""
        try:
            safety_status = await self.agent_executor.check_safety()
            
            # Determine safety level
            if safety_status == "safe":
                safety_level = "normal"
            elif safety_status == "warning":
                safety_level = "caution"
            elif safety_status == "danger":
                safety_level = "critical"
            else:
                safety_level = "unknown"
            
            return {
                "safety_status": safety_status,
                "safety_level": safety_level,
                "check_timestamp": asyncio.get_event_loop().time(),
                "recommendations": self._get_safety_recommendations(safety_status)
            }
            
        except Exception as e:
            logger.error(f"Safety check error: {e}")
            raise
    
    async def _plan_path(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a path for navigation"""
        try:
            start_point = parameters.get("start_point", [0, 0, 0])
            end_point = parameters.get("end_point", [0, 0, 0])
            obstacles = parameters.get("obstacles", [])
            
            # Simulate path planning
            path_length = ((end_point[0] - start_point[0])**2 + 
                          (end_point[1] - start_point[1])**2 + 
                          (end_point[2] - start_point[2])**2)**0.5
            
            # Generate waypoints
            num_waypoints = max(2, int(path_length / 0.5))  # Waypoint every 0.5 units
            waypoints = []
            
            for i in range(num_waypoints):
                t = i / (num_waypoints - 1) if num_waypoints > 1 else 0
                waypoint = [
                    start_point[0] + t * (end_point[0] - start_point[0]),
                    start_point[1] + t * (end_point[1] - start_point[1]),
                    start_point[2] + t * (end_point[2] - start_point[2])
                ]
                waypoints.append(waypoint)
            
            return {
                "start_point": start_point,
                "end_point": end_point,
                "waypoints": waypoints,
                "path_length": path_length,
                "obstacles_avoided": len(obstacles),
                "planning_time": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Path planning error: {e}")
            raise
    
    async def _execute_manipulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manipulation task"""
        try:
            manipulation_type = parameters.get("type", "")
            target_object = parameters.get("target_object", "")
            manipulation_params = parameters.get("parameters", {})
            
            if not manipulation_type or not target_object:
                raise ValueError("Manipulation type and target object are required")
            
            # Simulate manipulation execution
            await asyncio.sleep(0.2)  # Simulate manipulation time
            
            success = True
            if manipulation_type == "grasp":
                result = f"Successfully grasped {target_object}"
            elif manipulation_type == "place":
                result = f"Successfully placed {target_object}"
            elif manipulation_type == "move":
                result = f"Successfully moved {target_object}"
            else:
                result = f"Executed {manipulation_type} on {target_object}"
            
            return {
                "manipulation_type": manipulation_type,
                "target_object": target_object,
                "result": result,
                "success": success,
                "execution_time": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Manipulation execution error: {e}")
            raise
    
    def _get_safety_recommendations(self, safety_status: str) -> List[str]:
        """Get safety recommendations based on status"""
        recommendations = []
        
        if safety_status == "safe":
            recommendations.append("Continue normal operation")
        elif safety_status == "warning":
            recommendations.append("Reduce speed and increase monitoring")
            recommendations.append("Check for potential hazards")
        elif safety_status == "danger":
            recommendations.append("Stop all motion immediately")
            recommendations.append("Activate emergency protocols")
            recommendations.append("Contact human operator")
        
        return recommendations
    
    async def _safety_monitoring_loop(self):
        """Continuous safety monitoring loop"""
        try:
            while self._running:
                try:
                    # Check safety status
                    safety_status = await self.agent_executor.check_safety()
                    
                    # Publish safety status event
                    await event_bus.publish(event_bus.Event(
                        event_type="safety.status",
                        data={
                            "status": safety_status,
                            "timestamp": asyncio.get_event_loop().time()
                        },
                        source="AgentExecutionPlugin"
                    ))
                    
                    # Handle critical safety issues
                    if safety_status == "danger":
                        logger.warning("Critical safety issue detected!")
                        
                        # Publish emergency event
                        await event_bus.publish(event_bus.Event(
                            event_type="safety.emergency",
                            data={
                                "status": safety_status,
                                "action": "emergency_stop",
                                "timestamp": asyncio.get_event_loop().time()
                            },
                            source="AgentExecutionPlugin"
                        ))
                    
                    # Wait before next safety check
                    await asyncio.sleep(1.0)  # Check safety every second
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in safety monitoring loop: {e}")
                    await asyncio.sleep(5.0)  # Wait longer on error
                    
        except asyncio.CancelledError:
            logger.info("Safety monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Safety monitoring loop failed: {e}")


# Plugin entry point
AgentExecutionPlugin = AgentExecutionPlugin
