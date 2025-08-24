"""
Configuration Manager - Centralized Configuration Management
Provides dynamic loading, validation, and runtime updates of system configuration
"""

import asyncio
import json
import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from watchdog.observers import Observer

from .event_bus import Event, event_bus

logger = logging.getLogger(__name__)


@dataclass
class ConfigSection:
    """Configuration section with metadata"""
    name: str
    data: Dict[str, Any]
    source_file: Path
    last_modified: float
    schema: Optional[Dict[str, Any]] = None
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)


class ConfigValidator:
    """Configuration schema validation"""
    
    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}
    
    def register_schema(self, section_name: str, schema: Dict[str, Any]) -> bool:
        """Register a validation schema for a configuration section"""
        try:
            self._schemas[section_name] = schema
            logger.debug(f"Registered schema for section: {section_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register schema for {section_name}: {e}")
            return False
    
    def validate_section(self, section_name: str, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a configuration section against its schema"""
        try:
            schema = self._schemas.get(section_name)
            if not schema:
                return True, []  # No schema means no validation required
            
            errors = []
            
            # Basic type checking
            for key, expected_type in schema.get("types", {}).items():
                if key in data:
                    if not isinstance(data[key], expected_type):
                        errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(data[key]).__name__}")
            
            # Required fields checking
            for required_field in schema.get("required", []):
                if required_field not in data:
                    errors.append(f"Missing required field: {required_field}")
            
            # Enum validation
            for key, allowed_values in schema.get("enums", {}).items():
                if key in data and data[key] not in allowed_values:
                    errors.append(f"Invalid value for {key}: {data[key]}. Allowed: {allowed_values}")
            
            # Range validation
            for key, range_info in schema.get("ranges", {}).items():
                if key in data:
                    value = data[key]
                    if "min" in range_info and value < range_info["min"]:
                        errors.append(f"Value for {key} ({value}) is below minimum {range_info['min']}")
                    if "max" in range_info and value > range_info["max"]:
                        errors.append(f"Value for {key} ({value}) is above maximum {range_info['max']}")
            
            # Custom validation functions
            for key, validation_func in schema.get("custom_validators", {}).items():
                if key in data:
                    try:
                        if not validation_func(data[key]):
                            errors.append(f"Custom validation failed for {key}")
                    except Exception as e:
                        errors.append(f"Custom validation error for {key}: {e}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Validation error for section {section_name}: {e}")
            return False, [f"Validation error: {e}"]
    
    def get_schema(self, section_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a section"""
        return self._schemas.get(section_name)
    
    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered schemas"""
        return self._schemas.copy()


class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            logger.info(f"Configuration file changed: {event.src_path}")
            # Schedule reload to avoid multiple rapid reloads
            asyncio.create_task(self.config_manager._schedule_reload(event.src_path))


class ConfigManager:
    """Main configuration manager"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.validator = ConfigValidator()
        self._sections: Dict[str, ConfigSection] = {}
        self._file_watcher: Optional[ConfigFileWatcher] = None
        self._observer: Optional[Observer] = None
        self._reload_tasks: Set[str] = set()
        
        # Register event handlers
        event_bus.register_handler("config.update", self._handle_config_update, "config_manager")
        event_bus.register_handler("config.reload", self._handle_config_reload, "config_manager")
    
    async def initialize(self) -> bool:
        """Initialize the configuration manager"""
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(exist_ok=True)
            
            # Load all configuration files
            await self._load_all_config_files()
            
            # Set up file watching
            await self._setup_file_watching()
            
            # Register default schemas
            self._register_default_schemas()
            
            logger.info(f"Configuration manager initialized with {len(self._sections)} sections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration manager: {e}")
            return False
    
    async def _load_all_config_files(self):
        """Load all configuration files from the config directory"""
        try:
            config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml")) + list(self.config_dir.glob("*.json"))
            
            for config_file in config_files:
                await self._load_config_file(config_file)
                
        except Exception as e:
            logger.error(f"Failed to load configuration files: {e}")
    
    async def _load_config_file(self, file_path: Path) -> bool:
        """Load a single configuration file"""
        try:
            if not file_path.exists():
                logger.warning(f"Configuration file not found: {file_path}")
                return False
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix in ['.yaml', '.yml']:
                    content = yaml.safe_load(f)
                else:
                    content = json.load(f)
            
            if not isinstance(content, dict):
                logger.error(f"Invalid configuration file format: {file_path}")
                return False
            
            # Process each section
            for section_name, section_data in content.items():
                if not isinstance(section_data, dict):
                    logger.warning(f"Invalid section format in {file_path}: {section_name}")
                    continue
                
                # Validate section
                is_valid, errors = self.validator.validate_section(section_name, section_data)
                
                # Create or update section
                config_section = ConfigSection(
                    name=section_name,
                    data=section_data,
                    source_file=file_path,
                    last_modified=file_path.stat().st_mtime,
                    schema=self.validator.get_schema(section_name),
                    is_valid=is_valid,
                    validation_errors=errors
                )
                
                self._sections[section_name] = config_section
                
                if not is_valid:
                    logger.warning(f"Configuration section {section_name} has validation errors: {errors}")
                else:
                    logger.debug(f"Loaded configuration section: {section_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration file {file_path}: {e}")
            return False
    
    async def _setup_file_watching(self):
        """Set up file system watching for configuration changes"""
        try:
            self._file_watcher = ConfigFileWatcher(self)
            self._observer = Observer()
            self._observer.schedule(self._file_watcher, str(self.config_dir), recursive=False)
            self._observer.start()
            
            logger.info("Configuration file watching enabled")
            
        except Exception as e:
            logger.error(f"Failed to set up file watching: {e}")
    
    def _register_default_schemas(self):
        """Register default validation schemas"""
        # System configuration schema
        self.validator.register_schema("system", {
            "types": {
                "debug": bool,
                "log_level": str,
                "max_workers": int,
                "timeout": (int, float)
            },
            "required": ["debug", "log_level"],
            "enums": {
                "log_level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            },
            "ranges": {
                "max_workers": {"min": 1, "max": 100},
                "timeout": {"min": 0.1, "max": 3600}
            }
        })
        
        # Event bus configuration schema
        self.validator.register_schema("event_bus", {
            "types": {
                "max_history": int,
                "queue_size": int,
                "enable_priority": bool
            },
            "ranges": {
                "max_history": {"min": 100, "max": 10000},
                "queue_size": {"min": 100, "max": 100000}
            }
        })
        
        # Plugin manager configuration schema
        self.validator.register_schema("plugin_manager", {
            "types": {
                "auto_discovery": bool,
                "plugin_dirs": list,
                "enable_hot_reload": bool
            }
        })
    
    async def _schedule_reload(self, file_path: str):
        """Schedule a configuration reload with debouncing"""
        try:
            # Cancel existing reload task for this file
            if file_path in self._reload_tasks:
                return
            
            self._reload_tasks.add(file_path)
            
            # Wait a bit to avoid multiple rapid reloads
            await asyncio.sleep(1.0)
            
            # Reload the file
            await self._load_config_file(Path(file_path))
            
            # Publish configuration changed event
            await event_bus.publish(Event(
                event_type="config.changed",
                data={"file_path": file_path},
                source="config_manager"
            ))
            
            self._reload_tasks.discard(file_path)
            
        except Exception as e:
            logger.error(f"Failed to reload configuration file {file_path}: {e}")
            self._reload_tasks.discard(file_path)
    
    def get_value(self, section_name: str, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        try:
            section = self._sections.get(section_name)
            if not section:
                return default
            
            return section.data.get(key, default)
            
        except Exception as e:
            logger.error(f"Failed to get config value {section_name}.{key}: {e}")
            return default
    
    def set_value(self, section_name: str, key: str, value: Any) -> bool:
        """Set a configuration value"""
        try:
            if section_name not in self._sections:
                # Create new section
                self._sections[section_name] = ConfigSection(
                    name=section_name,
                    data={},
                    source_file=Path("runtime"),
                    last_modified=asyncio.get_event_loop().time()
                )
            
            section = self._sections[section_name]
            section.data[key] = value
            section.last_modified = asyncio.get_event_loop().time()
            
            # Validate the updated section
            is_valid, errors = self.validator.validate_section(section_name, section.data)
            section.is_valid = is_valid
            section.validation_errors = errors
            
            # Publish update event
            asyncio.create_task(event_bus.publish(Event(
                event_type="config.updated",
                data={
                    "section": section_name,
                    "key": key,
                    "value": value,
                    "is_valid": is_valid
                },
                source="config_manager"
            )))
            
            logger.debug(f"Set config value: {section_name}.{key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set config value {section_name}.{key}: {e}")
            return False
    
    def get_section(self, section_name: str) -> Optional[Dict[str, Any]]:
        """Get entire configuration section"""
        section = self._sections.get(section_name)
        return section.data if section else None
    
    def get_all_sections(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration sections"""
        return {name: section.data for name, section in self._sections.items()}
    
    def get_section_info(self, section_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a configuration section"""
        section = self._sections.get(section_name)
        if not section:
            return None
        
        return {
            "name": section.name,
            "source_file": str(section.source_file),
            "last_modified": section.last_modified,
            "is_valid": section.is_valid,
            "validation_errors": section.validation_errors,
            "has_schema": section.schema is not None
        }
    
    def register_schema(self, section_name: str, schema: Dict[str, Any]) -> bool:
        """Register a validation schema"""
        success = self.validator.register_schema(section_name, schema)
        if success:
            # Re-validate existing section if it exists
            if section_name in self._sections:
                section = self._sections[section_name]
                is_valid, errors = self.validator.validate_section(section_name, section.data)
                section.is_valid = is_valid
                section.validation_errors = errors
                section.schema = schema
        
        return success
    
    async def save_section(self, section_name: str, file_path: Optional[Path] = None) -> bool:
        """Save a configuration section to file"""
        try:
            section = self._sections.get(section_name)
            if not section:
                logger.error(f"Configuration section {section_name} not found")
                return False
            
            if file_path is None:
                file_path = section.source_file
            
            # Determine file format
            if file_path.suffix in ['.yaml', '.yml']:
                content = {section_name: section.data}
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(content, f, default_flow_style=False, indent=2)
            else:
                content = {section_name: section.data}
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2)
            
            logger.info(f"Saved configuration section {section_name} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration section {section_name}: {e}")
            return False
    
    async def _handle_config_update(self, event: Event):
        """Handle configuration update event"""
        section_name = event.data.get("section")
        key = event.data.get("key")
        value = event.data.get("value")
        
        if section_name and key is not None:
            self.set_value(section_name, key, value)
    
    async def _handle_config_reload(self, event: Event):
        """Handle configuration reload event"""
        file_path = event.data.get("file_path")
        if file_path:
            await self._load_config_file(Path(file_path))
        else:
            await self._load_all_config_files()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get configuration manager statistics"""
        return {
            "total_sections": len(self._sections),
            "valid_sections": sum(1 for s in self._sections.values() if s.is_valid),
            "invalid_sections": sum(1 for s in self._sections.values() if not s.is_valid),
            "total_schemas": len(self.validator._schemas),
            "file_watching_enabled": self._observer is not None and self._observer.is_alive()
        }
    
    async def shutdown(self):
        """Shutdown the configuration manager"""
        try:
            if self._observer:
                self._observer.stop()
                self._observer.join()
            
            logger.info("Configuration manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during configuration manager shutdown: {e}")


# Global configuration manager instance
config_manager = ConfigManager()
