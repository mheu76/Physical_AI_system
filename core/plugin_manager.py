"""
Plugin Manager - Dynamic Plugin Lifecycle Management
Handles plugin discovery, loading, unloading, and registration
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from uuid import uuid4

from .event_bus import Event, event_bus

logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    """Abstract base class for all plugins"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration"""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the plugin"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the plugin"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        pass


@dataclass
class PluginInfo:
    """Plugin metadata information"""
    name: str
    version: str
    description: str
    author: str
    category: str
    entry_point: str
    file_path: Path
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None


@dataclass
class PluginInstance:
    """Loaded plugin instance"""
    info: PluginInfo
    instance: PluginInterface
    config: Dict[str, Any]
    status: str = "loaded"  # loaded, initialized, started, stopped, error
    load_time: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    error_message: Optional[str] = None


class PluginRegistry:
    """Plugin metadata registry"""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register_plugin(self, plugin_info: PluginInfo) -> bool:
        """Register a plugin"""
        try:
            self._plugins[plugin_info.name] = plugin_info
            
            if plugin_info.category not in self._categories:
                self._categories[plugin_info.category] = []
            self._categories[plugin_info.category].append(plugin_info.name)
            
            logger.debug(f"Registered plugin: {plugin_info.name} ({plugin_info.category})")
            return True
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_info.name}: {e}")
            return False
    
    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """Get plugin info by name"""
        return self._plugins.get(name)
    
    def get_plugins_by_category(self, category: str) -> List[PluginInfo]:
        """Get all plugins in a category"""
        plugin_names = self._categories.get(category, [])
        return [self._plugins[name] for name in plugin_names if name in self._plugins]
    
    def get_all_plugins(self) -> List[PluginInfo]:
        """Get all registered plugins"""
        return list(self._plugins.values())
    
    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin"""
        try:
            if name in self._plugins:
                plugin_info = self._plugins[name]
                if plugin_info.category in self._categories:
                    self._categories[plugin_info.category].remove(name)
                del self._plugins[name]
                logger.debug(f"Unregistered plugin: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unregister plugin {name}: {e}")
            return False


class PluginLoader:
    """Plugin loading and discovery"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or ["plugins"]
        self._discovered_plugins: Dict[str, PluginInfo] = {}
    
    def discover_plugins(self) -> List[PluginInfo]:
        """Discover all available plugins"""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
            
            for plugin_file in plugin_path.rglob("*.py"):
                if plugin_file.name.startswith("__"):
                    continue
                
                plugin_info = self._parse_plugin_file(plugin_file)
                if plugin_info:
                    discovered.append(plugin_info)
                    self._discovered_plugins[plugin_info.name] = plugin_info
        
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    def _parse_plugin_file(self, file_path: Path) -> Optional[PluginInfo]:
        """Parse plugin metadata from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from comments
            metadata = {}
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('# PLUGIN_'):
                    key, value = line[2:].split(':', 1)
                    metadata[key] = value.strip()
            
            if 'PLUGIN_NAME' not in metadata:
                return None
            
            return PluginInfo(
                name=metadata['PLUGIN_NAME'],
                version=metadata.get('PLUGIN_VERSION', '1.0.0'),
                description=metadata.get('PLUGIN_DESCRIPTION', ''),
                author=metadata.get('PLUGIN_AUTHOR', ''),
                category=metadata.get('PLUGIN_CATEGORY', 'general'),
                dependencies=metadata.get('PLUGIN_DEPENDENCIES', '').split(',') if metadata.get('PLUGIN_DEPENDENCIES') else [],
                entry_point=metadata.get('PLUGIN_ENTRY_POINT', ''),
                file_path=file_path
            )
            
        except Exception as e:
            logger.error(f"Failed to parse plugin file {file_path}: {e}")
            return None
    
    async def load_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> Optional[PluginInstance]:
        """Load a plugin instance"""
        try:
            plugin_info = self._discovered_plugins.get(plugin_name)
            if not plugin_info:
                logger.error(f"Plugin {plugin_name} not found")
                return None
            
            # Add plugin directory to Python path
            plugin_dir = str(plugin_info.file_path.parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # Import the module
            module_name = plugin_info.file_path.stem
            module = importlib.import_module(module_name)
            
            # Get the plugin class
            plugin_class = getattr(module, plugin_info.entry_point, None)
            if not plugin_class:
                logger.error(f"Entry point {plugin_info.entry_point} not found in {plugin_name}")
                return None
            
            # Check if it implements PluginInterface
            if not issubclass(plugin_class, PluginInterface):
                logger.error(f"Plugin {plugin_name} does not implement PluginInterface")
                return None
            
            # Create instance
            instance = plugin_class()
            
            plugin_instance = PluginInstance(
                info=plugin_info,
                instance=instance,
                config=config or {}
            )
            
            logger.info(f"Loaded plugin: {plugin_name}")
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return None
    
    async def unload_plugin(self, plugin_instance: PluginInstance) -> bool:
        """Unload a plugin instance"""
        try:
            # Stop the plugin if it's running
            if plugin_instance.status in ["started", "initialized"]:
                await plugin_instance.instance.stop()
            
            # Remove from Python path
            plugin_dir = str(plugin_instance.info.file_path.parent)
            if plugin_dir in sys.path:
                sys.path.remove(plugin_dir)
            
            logger.info(f"Unloaded plugin: {plugin_instance.info.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_instance.info.name}: {e}")
            return False


class PluginManager:
    """Main plugin manager orchestrating plugin lifecycle"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.registry = PluginRegistry()
        self.loader = PluginLoader(plugin_dirs)
        self._loaded_instances: Dict[str, PluginInstance] = {}
        
        # Register event handlers
        event_bus.register_handler("plugin.load", self._handle_plugin_load, "plugin_manager")
        event_bus.register_handler("plugin.unload", self._handle_plugin_unload, "plugin_manager")
        event_bus.register_handler("plugin.start", self._handle_plugin_start, "plugin_manager")
        event_bus.register_handler("plugin.stop", self._handle_plugin_stop, "plugin_manager")
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin manager"""
        try:
            # Discover plugins
            discovered_plugins = self.loader.discover_plugins()
            
            # Register discovered plugins
            for plugin_info in discovered_plugins:
                self.registry.register_plugin(plugin_info)
            
            # Publish discovery event
            await event_bus.publish(Event(
                event_type="plugin.discovered",
                data={"plugins": [p.name for p in discovered_plugins]},
                source="plugin_manager"
            ))
            
            logger.info(f"Plugin manager initialized with {len(discovered_plugins)} plugins")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin manager: {e}")
            return False
    
    async def load_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """Load a plugin"""
        try:
            if plugin_name in self._loaded_instances:
                logger.warning(f"Plugin {plugin_name} already loaded")
                return True
            
            plugin_instance = await self.loader.load_plugin(plugin_name, config)
            if not plugin_instance:
                return False
            
            # Initialize the plugin
            try:
                success = await plugin_instance.instance.initialize(plugin_instance.config)
                if success:
                    plugin_instance.status = "initialized"
                else:
                    plugin_instance.status = "error"
                    plugin_instance.error_message = "Initialization failed"
            except Exception as e:
                plugin_instance.status = "error"
                plugin_instance.error_message = str(e)
                logger.error(f"Plugin initialization error: {e}")
            
            self._loaded_instances[plugin_name] = plugin_instance
            
            # Publish load event
            await event_bus.publish(Event(
                event_type="plugin.loaded",
                data={"plugin_name": plugin_name, "status": plugin_instance.status},
                source="plugin_manager"
            ))
            
            return plugin_instance.status == "initialized"
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    async def start_plugin(self, plugin_name: str) -> bool:
        """Start a plugin"""
        try:
            plugin_instance = self._loaded_instances.get(plugin_name)
            if not plugin_instance:
                logger.error(f"Plugin {plugin_name} not loaded")
                return False
            
            if plugin_instance.status == "started":
                return True
            
            if plugin_instance.status != "initialized":
                logger.error(f"Plugin {plugin_name} not initialized (status: {plugin_instance.status})")
                return False
            
            success = await plugin_instance.instance.start()
            if success:
                plugin_instance.status = "started"
                plugin_instance.error_message = None
                
                # Publish start event
                await event_bus.publish(Event(
                    event_type="plugin.started",
                    data={"plugin_name": plugin_name},
                    source="plugin_manager"
                ))
                
                logger.info(f"Started plugin: {plugin_name}")
                return True
            else:
                plugin_instance.status = "error"
                plugin_instance.error_message = "Start failed"
                return False
                
        except Exception as e:
            logger.error(f"Failed to start plugin {plugin_name}: {e}")
            return False
    
    async def stop_plugin(self, plugin_name: str) -> bool:
        """Stop a plugin"""
        try:
            plugin_instance = self._loaded_instances.get(plugin_name)
            if not plugin_instance:
                return True  # Already not loaded
            
            if plugin_instance.status != "started":
                return True
            
            success = await plugin_instance.instance.stop()
            if success:
                plugin_instance.status = "initialized"
                
                # Publish stop event
                await event_bus.publish(Event(
                    event_type="plugin.stopped",
                    data={"plugin_name": plugin_name},
                    source="plugin_manager"
                ))
                
                logger.info(f"Stopped plugin: {plugin_name}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to stop plugin {plugin_name}: {e}")
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        try:
            plugin_instance = self._loaded_instances.get(plugin_name)
            if not plugin_instance:
                return True
            
            # Stop first if needed
            if plugin_instance.status == "started":
                await self.stop_plugin(plugin_name)
            
            # Unload
            success = await self.loader.unload_plugin(plugin_instance)
            if success:
                del self._loaded_instances[plugin_name]
                
                # Publish unload event
                await event_bus.publish(Event(
                    event_type="plugin.unloaded",
                    data={"plugin_name": plugin_name},
                    source="plugin_manager"
                ))
                
                logger.info(f"Unloaded plugin: {plugin_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def get_plugin_status(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get plugin status"""
        plugin_instance = self._loaded_instances.get(plugin_name)
        if not plugin_instance:
            return None
        
        return {
            "name": plugin_instance.info.name,
            "status": plugin_instance.status,
            "category": plugin_instance.info.category,
            "load_time": plugin_instance.load_time,
            "error_message": plugin_instance.error_message
        }
    
    def get_all_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all loaded plugins"""
        return {
            name: self.get_plugin_status(name)
            for name in self._loaded_instances.keys()
        }
    
    async def _handle_plugin_load(self, event: Event):
        """Handle plugin load event"""
        plugin_name = event.data.get("plugin_name")
        config = event.data.get("config")
        if plugin_name:
            await self.load_plugin(plugin_name, config)
    
    async def _handle_plugin_unload(self, event: Event):
        """Handle plugin unload event"""
        plugin_name = event.data.get("plugin_name")
        if plugin_name:
            await self.unload_plugin(plugin_name)
    
    async def _handle_plugin_start(self, event: Event):
        """Handle plugin start event"""
        plugin_name = event.data.get("plugin_name")
        if plugin_name:
            await self.start_plugin(plugin_name)
    
    async def _handle_plugin_stop(self, event: Event):
        """Handle plugin stop event"""
        plugin_name = event.data.get("plugin_name")
        if plugin_name:
            await self.stop_plugin(plugin_name)


# Global plugin manager instance
plugin_manager = PluginManager()
