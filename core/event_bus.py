"""
Event Bus - Central Event-Driven Communication Hub
Provides asynchronous event publishing and subscription capabilities
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels for handling order"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Event data structure"""
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None


@dataclass
class EventHandler:
    """Event handler metadata"""
    handler: Callable[[Event], Any]
    module_name: str
    priority: EventPriority
    is_async: bool
    handler_id: str = field(default_factory=lambda: str(uuid4()))


class EventBus:
    """Central event bus for inter-module communication"""
    
    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "handlers_registered": 0
        }
    
    async def start(self):
        """Start the event bus processing loop"""
        if self._running:
            return
        
        self._running = True
        asyncio.create_task(self._process_events())
        logger.info("Event Bus started")
    
    async def stop(self):
        """Stop the event bus processing loop"""
        self._running = False
        logger.info("Event Bus stopped")
    
    def register_handler(self, event_type: str, handler: Callable[[Event], Any], 
                        module_name: str, priority: EventPriority = EventPriority.NORMAL, 
                        is_async: bool = True) -> bool:
        """Register an event handler"""
        try:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            
            event_handler = EventHandler(
                handler=handler,
                module_name=module_name,
                priority=priority,
                is_async=is_async
            )
            
            self._handlers[event_type].append(event_handler)
            # Sort by priority (highest first)
            self._handlers[event_type].sort(key=lambda h: h.priority.value, reverse=True)
            
            self._stats["handlers_registered"] += 1
            logger.debug(f"Registered handler for {event_type} from {module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register handler for {event_type}: {e}")
            return False
    
    def unregister_handler(self, event_type: str, module_name: str) -> bool:
        """Unregister all handlers for a module and event type"""
        try:
            if event_type in self._handlers:
                original_count = len(self._handlers[event_type])
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] 
                    if h.module_name != module_name
                ]
                removed_count = original_count - len(self._handlers[event_type])
                logger.debug(f"Unregistered {removed_count} handlers for {event_type} from {module_name}")
                return removed_count > 0
            return False
        except Exception as e:
            logger.error(f"Failed to unregister handler: {e}")
            return False
    
    async def publish(self, event: Event) -> bool:
        """Publish an event to the bus"""
        try:
            await self._event_queue.put(event)
            self._stats["events_published"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_type}: {e}")
            return False
    
    async def publish_sync(self, event: Event) -> bool:
        """Publish and process event synchronously"""
        try:
            if event.event_type in self._handlers:
                handlers = self._handlers[event.event_type].copy()
                for handler_info in handlers:
                    try:
                        if handler_info.is_async:
                            await handler_info.handler(event)
                        else:
                            handler_info.handler(event)
                    except Exception as e:
                        logger.error(f"Handler error in {handler_info.module_name}: {e}")
            
            self._stats["events_processed"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to process event {event.event_type}: {e}")
            return False
    
    async def _process_events(self):
        """Main event processing loop"""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                
                # Add to history
                self._event_history.append(event)
                if len(self._event_history) > self._max_history:
                    self._event_history.pop(0)
                
                # Process event
                await self.publish_sync(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            **self._stats,
            "active_handlers": sum(len(handlers) for handlers in self._handlers.values()),
            "event_types": list(self._handlers.keys()),
            "queue_size": self._event_queue.qsize(),
            "history_size": len(self._event_history)
        }
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get recent event history"""
        history = self._event_history
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        return history[-limit:]
    
    def clear_history(self):
        """Clear event history"""
        self._event_history.clear()


# Global event bus instance
event_bus = EventBus()
