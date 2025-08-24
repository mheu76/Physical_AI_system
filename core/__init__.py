"""
Core Module - Modular Physical AI System Core Components
Provides the foundational infrastructure for the modular system
"""

from .event_bus import EventBus, Event, EventPriority, event_bus
from .plugin_manager import PluginManager, PluginInterface, PluginInfo, plugin_manager
from .mission_broker import MissionBroker, Mission, MissionStatus, MissionPriority, mission_broker
from .config_manager import ConfigManager, ConfigSection, config_manager

__all__ = [
    # Event Bus
    "EventBus",
    "Event", 
    "EventPriority",
    "event_bus",
    
    # Plugin Manager
    "PluginManager",
    "PluginInterface", 
    "PluginInfo",
    "plugin_manager",
    
    # Mission Broker
    "MissionBroker",
    "Mission",
    "MissionStatus", 
    "MissionPriority",
    "mission_broker",
    
    # Config Manager
    "ConfigManager",
    "ConfigSection",
    "config_manager"
]

__version__ = "3.0.0-modular"
__author__ = "Physical AI Team"
__description__ = "Core infrastructure for modular Physical AI system"
