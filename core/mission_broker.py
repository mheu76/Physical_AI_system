"""
Mission Broker - Event-Driven Mission Management Hub
Manages mission submission, routing, and execution
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from .event_bus import Event, event_bus

logger = logging.getLogger(__name__)


class MissionStatus(Enum):
    """Mission execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MissionPriority(Enum):
    """Mission priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Mission:
    """Mission data structure"""
    mission_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: MissionPriority = MissionPriority.NORMAL
    status: MissionStatus = MissionStatus.PENDING
    source: str = ""
    target_plugin: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timeout: Optional[float] = None  # seconds
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class MissionTemplate:
    """Mission template for common mission patterns"""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    target_plugin: str
    default_priority: MissionPriority = MissionPriority.NORMAL
    timeout: Optional[float] = None
    max_retries: int = 3


class MissionRouter:
    """Routes missions to appropriate plugins based on capabilities"""
    
    def __init__(self):
        self._plugin_capabilities: Dict[str, Set[str]] = {}
        self._capability_plugins: Dict[str, List[str]] = {}
    
    def register_plugin_capability(self, plugin_name: str, capability: str) -> bool:
        """Register a plugin capability"""
        try:
            if plugin_name not in self._plugin_capabilities:
                self._plugin_capabilities[plugin_name] = set()
            
            self._plugin_capabilities[plugin_name].add(capability)
            
            if capability not in self._capability_plugins:
                self._capability_plugins[capability] = []
            
            if plugin_name not in self._capability_plugins[capability]:
                self._capability_plugins[capability].append(plugin_name)
            
            logger.debug(f"Registered capability {capability} for plugin {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register capability {capability} for {plugin_name}: {e}")
            return False
    
    def unregister_plugin_capability(self, plugin_name: str, capability: str) -> bool:
        """Unregister a plugin capability"""
        try:
            if plugin_name in self._plugin_capabilities:
                self._plugin_capabilities[plugin_name].discard(capability)
            
            if capability in self._capability_plugins:
                if plugin_name in self._capability_plugins[capability]:
                    self._capability_plugins[capability].remove(plugin_name)
                
                if not self._capability_plugins[capability]:
                    del self._capability_plugins[capability]
            
            logger.debug(f"Unregistered capability {capability} for plugin {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister capability {capability} for {plugin_name}: {e}")
            return False
    
    def route_mission(self, mission: Mission) -> Optional[str]:
        """Route a mission to the appropriate plugin"""
        try:
            # If target plugin is specified, use it
            if mission.target_plugin:
                if mission.target_plugin in self._plugin_capabilities:
                    return mission.target_plugin
                else:
                    logger.warning(f"Target plugin {mission.target_plugin} not found")
                    return None
            
            # Try to route based on mission name (capability)
            if mission.name in self._capability_plugins:
                plugins = self._capability_plugins[mission.name]
                if plugins:
                    # For now, return the first available plugin
                    # In the future, could implement load balancing or selection logic
                    return plugins[0]
            
            logger.warning(f"No plugin found for mission {mission.name}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to route mission {mission.mission_id}: {e}")
            return None
    
    def get_plugin_capabilities(self, plugin_name: str) -> Set[str]:
        """Get capabilities of a plugin"""
        return self._plugin_capabilities.get(plugin_name, set())
    
    def get_capability_plugins(self, capability: str) -> List[str]:
        """Get plugins that support a capability"""
        return self._capability_plugins.get(capability, [])
    
    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get all registered capabilities and their plugins"""
        return self._capability_plugins.copy()


class MissionQueue:
    """Priority-based mission queue"""
    
    def __init__(self):
        self._queues: Dict[MissionPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MissionPriority
        }
        self._active_missions: Dict[str, Mission] = {}
        self._completed_missions: Dict[str, Mission] = {}
        self._max_completed_history = 1000
    
    async def enqueue_mission(self, mission: Mission) -> bool:
        """Add a mission to the appropriate priority queue"""
        try:
            await self._queues[mission.priority].put(mission)
            logger.debug(f"Enqueued mission {mission.mission_id} with priority {mission.priority.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue mission {mission.mission_id}: {e}")
            return False
    
    async def dequeue_mission(self) -> Optional[Mission]:
        """Get the next highest priority mission"""
        try:
            # Check queues in priority order (highest first)
            for priority in sorted(MissionPriority, key=lambda p: p.value, reverse=True):
                queue = self._queues[priority]
                if not queue.empty():
                    mission = await queue.get()
                    mission.status = MissionStatus.RUNNING
                    mission.started_at = datetime.now()
                    self._active_missions[mission.mission_id] = mission
                    
                    logger.debug(f"Dequeued mission {mission.mission_id} with priority {priority.name}")
                    return mission
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to dequeue mission: {e}")
            return None
    
    def complete_mission(self, mission_id: str, result: Dict[str, Any] = None, 
                        error_message: str = None) -> bool:
        """Mark a mission as completed"""
        try:
            if mission_id not in self._active_missions:
                logger.warning(f"Mission {mission_id} not found in active missions")
                return False
            
            mission = self._active_missions[mission_id]
            mission.completed_at = datetime.now()
            
            if error_message:
                mission.status = MissionStatus.FAILED
                mission.error_message = error_message
            else:
                mission.status = MissionStatus.COMPLETED
                mission.result = result
            
            # Move to completed history
            self._completed_missions[mission_id] = mission
            del self._active_missions[mission_id]
            
            # Maintain history size
            if len(self._completed_missions) > self._max_completed_history:
                oldest_id = min(self._completed_missions.keys(), 
                              key=lambda k: self._completed_missions[k].completed_at)
                del self._completed_missions[oldest_id]
            
            logger.debug(f"Completed mission {mission_id} with status {mission.status.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete mission {mission_id}: {e}")
            return False
    
    def cancel_mission(self, mission_id: str) -> bool:
        """Cancel an active mission"""
        try:
            if mission_id not in self._active_missions:
                return False
            
            mission = self._active_missions[mission_id]
            mission.status = MissionStatus.CANCELLED
            mission.completed_at = datetime.now()
            
            # Move to completed history
            self._completed_missions[mission_id] = mission
            del self._active_missions[mission_id]
            
            logger.info(f"Cancelled mission {mission_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel mission {mission_id}: {e}")
            return False
    
    def get_mission_status(self, mission_id: str) -> Optional[Mission]:
        """Get mission status by ID"""
        # Check active missions first
        if mission_id in self._active_missions:
            return self._active_missions[mission_id]
        
        # Check completed missions
        if mission_id in self._completed_missions:
            return self._completed_missions[mission_id]
        
        return None
    
    def get_active_missions(self) -> List[Mission]:
        """Get all active missions"""
        return list(self._active_missions.values())
    
    def get_completed_missions(self, limit: int = 100) -> List[Mission]:
        """Get recent completed missions"""
        missions = list(self._completed_missions.values())
        missions.sort(key=lambda m: m.completed_at or datetime.min, reverse=True)
        return missions[:limit]
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "active_missions": len(self._active_missions),
            "completed_missions": len(self._completed_missions),
            "queue_sizes": {
                priority.name: queue.qsize()
                for priority, queue in self._queues.items()
            }
        }


class MissionBroker:
    """Main mission broker orchestrating mission management"""
    
    def __init__(self):
        self.router = MissionRouter()
        self.queue = MissionQueue()
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        
        # Register event handlers
        event_bus.register_handler("mission.submit", self._handle_mission_submit, "mission_broker")
        event_bus.register_handler("mission.complete", self._handle_mission_complete, "mission_broker")
        event_bus.register_handler("plugin.capability.register", self._handle_plugin_capability, "mission_broker")
        event_bus.register_handler("plugin.capability.unregister", self._handle_plugin_capability_unregister, "mission_broker")
    
    async def start(self):
        """Start the mission broker"""
        if self._running:
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._process_missions())
        logger.info("Mission Broker started")
    
    async def stop(self):
        """Stop the mission broker"""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Mission Broker stopped")
    
    async def submit_mission(self, mission: Mission) -> bool:
        """Submit a mission for execution"""
        try:
            # Route the mission
            target_plugin = self.router.route_mission(mission)
            if not target_plugin:
                logger.error(f"No suitable plugin found for mission {mission.name}")
                return False
            
            mission.target_plugin = target_plugin
            
            # Enqueue the mission
            success = await self.queue.enqueue_mission(mission)
            if success:
                # Publish mission submitted event
                await event_bus.publish(Event(
                    event_type="mission.submitted",
                    data={
                        "mission_id": mission.mission_id,
                        "mission_name": mission.name,
                        "target_plugin": target_plugin
                    },
                    source="mission_broker"
                ))
                
                logger.info(f"Submitted mission {mission.mission_id} ({mission.name}) to {target_plugin}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to submit mission {mission.mission_id}: {e}")
            return False
    
    async def get_mission_status(self, mission_id: str) -> Optional[Mission]:
        """Get mission status"""
        return self.queue.get_mission_status(mission_id)
    
    async def cancel_mission(self, mission_id: str) -> bool:
        """Cancel a mission"""
        success = self.queue.cancel_mission(mission_id)
        if success:
            await event_bus.publish(Event(
                event_type="mission.cancelled",
                data={"mission_id": mission_id},
                source="mission_broker"
            ))
        return success
    
    async def _process_missions(self):
        """Main mission processing loop"""
        while self._running:
            try:
                # Get next mission
                mission = await self.queue.dequeue_mission()
                if not mission:
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute mission
                await self._execute_mission(mission)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in mission processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _execute_mission(self, mission: Mission):
        """Execute a mission"""
        try:
            # Publish mission started event
            await event_bus.publish(Event(
                event_type="mission.started",
                data={
                    "mission_id": mission.mission_id,
                    "mission_name": mission.name,
                    "target_plugin": mission.target_plugin
                },
                source="mission_broker"
            ))
            
            # Create mission execution event
            execution_event = Event(
                event_type="mission.execute",
                data={
                    "mission_id": mission.mission_id,
                    "mission_name": mission.name,
                    "parameters": mission.parameters,
                    "source": mission.source
                },
                source="mission_broker",
                target=mission.target_plugin
            )
            
            # Publish execution event
            await event_bus.publish(execution_event)
            
            # Set up timeout if specified
            if mission.timeout:
                asyncio.create_task(self._mission_timeout_handler(mission))
            
        except Exception as e:
            logger.error(f"Failed to execute mission {mission.mission_id}: {e}")
            self.queue.complete_mission(mission.mission_id, error_message=str(e))
    
    async def _mission_timeout_handler(self, mission: Mission):
        """Handle mission timeout"""
        try:
            await asyncio.sleep(mission.timeout)
            
            # Check if mission is still active
            current_mission = self.queue.get_mission_status(mission.mission_id)
            if current_mission and current_mission.status == MissionStatus.RUNNING:
                logger.warning(f"Mission {mission.mission_id} timed out after {mission.timeout}s")
                self.queue.complete_mission(mission.mission_id, error_message="Mission timeout")
                
                await event_bus.publish(Event(
                    event_type="mission.timeout",
                    data={"mission_id": mission.mission_id},
                    source="mission_broker"
                ))
                
        except asyncio.CancelledError:
            pass
    
    async def _handle_mission_submit(self, event: Event):
        """Handle mission submit event"""
        mission_data = event.data
        mission = Mission(
            name=mission_data.get("name", ""),
            description=mission_data.get("description", ""),
            parameters=mission_data.get("parameters", {}),
            priority=MissionPriority(mission_data.get("priority", MissionPriority.NORMAL.value)),
            source=event.source,
            timeout=mission_data.get("timeout")
        )
        await self.submit_mission(mission)
    
    async def _handle_mission_complete(self, event: Event):
        """Handle mission complete event"""
        mission_id = event.data.get("mission_id")
        result = event.data.get("result")
        error_message = event.data.get("error_message")
        
        if mission_id:
            self.queue.complete_mission(mission_id, result, error_message)
    
    async def _handle_plugin_capability(self, event: Event):
        """Handle plugin capability registration"""
        plugin_name = event.data.get("plugin_name")
        capability = event.data.get("capability")
        
        if plugin_name and capability:
            self.router.register_plugin_capability(plugin_name, capability)
    
    async def _handle_plugin_capability_unregister(self, event: Event):
        """Handle plugin capability unregistration"""
        plugin_name = event.data.get("plugin_name")
        capability = event.data.get("capability")
        
        if plugin_name and capability:
            self.router.unregister_plugin_capability(plugin_name, capability)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mission broker statistics"""
        return {
            "router": {
                "plugin_capabilities": len(self.router._plugin_capabilities),
                "capability_plugins": len(self.router._capability_plugins)
            },
            "queue": self.queue.get_queue_stats()
        }


# Global mission broker instance
mission_broker = MissionBroker()
