#!/usr/bin/env python3
"""
Modular Physical AI System - Main Entry Point
Orchestrates the modular system with event-driven architecture
"""

import asyncio
import argparse
import logging
import signal
import sys
from typing import Dict, Any, Optional

from core import (event_bus, plugin_manager, mission_broker, config_manager, 
                 Event, EventPriority, Mission, MissionPriority)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('modular_system.log')
    ]
)

logger = logging.getLogger(__name__)


class ModularPhysicalAI:
    """Main orchestrator for the modular Physical AI system"""
    
    def __init__(self):
        self.running = False
        self.health_check_task: Optional[asyncio.Task] = None
        self.system_stats = {
            "start_time": None,
            "uptime": 0,
            "missions_processed": 0,
            "events_processed": 0,
            "plugins_loaded": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the modular system"""
        try:
            logger.info("Initializing Modular Physical AI System...")
            
            # Initialize configuration manager
            logger.info("Initializing configuration manager...")
            if not await config_manager.initialize():
                logger.error("Failed to initialize configuration manager")
                return False
            
            # Initialize event bus
            logger.info("Starting event bus...")
            await event_bus.start()
            
            # Initialize plugin manager
            logger.info("Initializing plugin manager...")
            plugin_config = config_manager.get_section("plugin_manager") or {}
            if not await plugin_manager.initialize(plugin_config):
                logger.error("Failed to initialize plugin manager")
                return False
            
            # Initialize mission broker
            logger.info("Starting mission broker...")
            await mission_broker.start()
            
            # Load core plugins
            logger.info("Loading core plugins...")
            await self._load_core_plugins()
            
            # Start core plugins
            logger.info("Starting core plugins...")
            await self._start_core_plugins()
            
            # Register mission templates
            logger.info("Registering mission templates...")
            await self._register_mission_templates()
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
            
            self.system_stats["start_time"] = asyncio.get_event_loop().time()
            self.running = True
            
            logger.info("Modular Physical AI System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def _load_core_plugins(self):
        """Load the core plugins"""
        core_plugins = [
            ("PHI35FoundationPlugin", "foundation"),
            ("DevelopmentalLearningPlugin", "learning"),
            ("AgentExecutionPlugin", "execution"),
            ("HardwareAbstractionPlugin", "hal")
        ]
        
        for plugin_name, category in core_plugins:
            try:
                plugin_config = config_manager.get_section(category) or {}
                success = await plugin_manager.load_plugin(plugin_name, plugin_config)
                if success:
                    logger.info(f"Loaded plugin: {plugin_name}")
                    self.system_stats["plugins_loaded"] += 1
                else:
                    logger.warning(f"Failed to load plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_name}: {e}")
    
    async def _start_core_plugins(self):
        """Start the loaded core plugins"""
        loaded_plugins = plugin_manager.get_all_plugin_status()
        
        for plugin_name, status in loaded_plugins.items():
            if status and status.get("status") == "initialized":
                try:
                    success = await plugin_manager.start_plugin(plugin_name)
                    if success:
                        logger.info(f"Started plugin: {plugin_name}")
                    else:
                        logger.warning(f"Failed to start plugin: {plugin_name}")
                except Exception as e:
                    logger.error(f"Error starting plugin {plugin_name}: {e}")
    
    async def _register_mission_templates(self):
        """Register common mission templates"""
        mission_templates = [
            {
                "name": "text_generation",
                "description": "Generate text using PHI-3.5",
                "parameters": {
                    "prompt": "string",
                    "max_length": "integer",
                    "temperature": "float"
                }
            },
            {
                "name": "skill_acquisition",
                "description": "Acquire a new skill",
                "parameters": {
                    "skill_name": "string",
                    "experience_value": "float",
                    "context": "string"
                }
            },
            {
                "name": "motion_control",
                "description": "Control robot motion",
                "parameters": {
                    "motion_type": "string",
                    "parameters": "object"
                }
            },
            {
                "name": "sensor_management",
                "description": "Manage sensor operations",
                "parameters": {
                    "operation": "string",
                    "sensor_id": "string",
                    "config": "object"
                }
            }
        ]
        
        for template in mission_templates:
            logger.debug(f"Registered mission template: {template['name']}")
    
    async def start(self):
        """Start the modular system"""
        if not self.running:
            logger.error("System not initialized. Call initialize() first.")
            return False
        
        logger.info("Starting Modular Physical AI System...")
        
        try:
            # Main system loop
            while self.running:
                await asyncio.sleep(1.0)
                
                # Update uptime
                if self.system_stats["start_time"]:
                    self.system_stats["uptime"] = asyncio.get_event_loop().time() - self.system_stats["start_time"]
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the modular system gracefully"""
        logger.info("Shutting down Modular Physical AI System...")
        self.running = False
        
        try:
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop all plugins
            loaded_plugins = plugin_manager.get_all_plugin_status()
            for plugin_name in loaded_plugins.keys():
                try:
                    await plugin_manager.stop_plugin(plugin_name)
                    logger.info(f"Stopped plugin: {plugin_name}")
                except Exception as e:
                    logger.error(f"Error stopping plugin {plugin_name}: {e}")
            
            # Stop mission broker
            await mission_broker.stop()
            logger.info("Stopped mission broker")
            
            # Stop event bus
            await event_bus.stop()
            logger.info("Stopped event bus")
            
            # Shutdown configuration manager
            await config_manager.shutdown()
            logger.info("Shutdown configuration manager")
            
            logger.info("Modular Physical AI System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def submit_mission(self, mission_name: str, parameters: dict = None) -> str:
        """Submit a mission to the system"""
        try:
            mission = Mission(
                name=mission_name,
                description=f"Mission: {mission_name}",
                parameters=parameters or {},
                source="modular_main"
            )
            
            success = await mission_broker.submit_mission(mission)
            if success:
                logger.info(f"Submitted mission: {mission_name} (ID: {mission.mission_id})")
                self.system_stats["missions_processed"] += 1
                return mission.mission_id
            else:
                logger.error(f"Failed to submit mission: {mission_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting mission {mission_name}: {e}")
            return None
    
    async def get_mission_status(self, mission_id: str) -> Optional[Mission]:
        """Get the status of a mission"""
        try:
            return await mission_broker.get_mission_status(mission_id)
        except Exception as e:
            logger.error(f"Error getting mission status: {e}")
            return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            # Get component stats
            event_bus_stats = event_bus.get_stats()
            plugin_stats = plugin_manager.get_all_plugin_status()
            mission_stats = mission_broker.get_stats()
            config_stats = config_manager.get_stats()
            
            return {
                "system": {
                    "running": self.running,
                    "uptime": self.system_stats["uptime"],
                    "start_time": self.system_stats["start_time"]
                },
                "statistics": {
                    "missions_processed": self.system_stats["missions_processed"],
                    "events_processed": event_bus_stats.get("events_processed", 0),
                    "plugins_loaded": len(plugin_stats),
                    "active_missions": mission_stats.get("queue", {}).get("active_missions", 0)
                },
                "components": {
                    "event_bus": event_bus_stats,
                    "plugin_manager": {
                        "total_plugins": len(plugin_stats),
                        "active_plugins": sum(1 for s in plugin_stats.values() if s and s.get("status") == "started")
                    },
                    "mission_broker": mission_stats,
                    "config_manager": config_stats
                },
                "plugins": plugin_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop"""
        try:
            while self.running:
                try:
                    # Get system status
                    status = await self.get_system_status()
                    
                    # Check for critical issues
                    if not status.get("system", {}).get("running", False):
                        logger.warning("System health check failed - system not running")
                    
                    # Publish health status event
                    await event_bus.publish(Event(
                        event_type="system.health",
                        data=status,
                        source="modular_main",
                        priority=EventPriority.NORMAL
                    ))
                    
                    # Wait before next health check
                    await asyncio.sleep(30.0)  # Health check every 30 seconds
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {e}")
                    await asyncio.sleep(60.0)  # Wait longer on error
                    
        except asyncio.CancelledError:
            logger.info("Health monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Health monitoring loop failed: {e}")


async def run_demo():
    """Run a demonstration of the modular system"""
    logger.info("Running Modular System Demo...")
    
    # Create system instance
    system = ModularPhysicalAI()
    
    # Initialize system
    if not await system.initialize():
        logger.error("Failed to initialize system for demo")
        return
    
    try:
        # Wait a moment for system to stabilize
        await asyncio.sleep(2.0)
        
        # Demo 1: Text Generation
        logger.info("Demo 1: Text Generation")
        mission_id = await system.submit_mission("text_generation", {
            "prompt": "Explain the benefits of modular architecture in robotics",
            "max_length": 200,
            "temperature": 0.7
        })
        
        if mission_id:
            # Wait for completion
            await asyncio.sleep(3.0)
            status = await system.get_mission_status(mission_id)
            if status:
                logger.info(f"Text generation completed: {status.status}")
        
        # Demo 2: Skill Acquisition
        logger.info("Demo 2: Skill Acquisition")
        mission_id = await system.submit_mission("skill_acquisition", {
            "skill_name": "object_recognition",
            "experience_value": 5.0,
            "context": "Demo learning session"
        })
        
        if mission_id:
            await asyncio.sleep(2.0)
            status = await system.get_mission_status(mission_id)
            if status:
                logger.info(f"Skill acquisition completed: {status.status}")
        
        # Demo 3: Motion Control
        logger.info("Demo 3: Motion Control")
        mission_id = await system.submit_mission("motion_control", {
            "motion_type": "move_to_position",
            "parameters": {
                "target_position": [1.0, 2.0, 0.5],
                "velocity": 0.5
            }
        })
        
        if mission_id:
            await asyncio.sleep(2.0)
            status = await system.get_mission_status(mission_id)
            if status:
                logger.info(f"Motion control completed: {status.status}")
        
        # Demo 4: Sensor Management
        logger.info("Demo 4: Sensor Management")
        mission_id = await system.submit_mission("sensor_management", {
            "operation": "read",
            "sensor_id": "temperature_sensor_01"
        })
        
        if mission_id:
            await asyncio.sleep(2.0)
            status = await system.get_mission_status(mission_id)
            if status:
                logger.info(f"Sensor management completed: {status.status}")
        
        # Get final system status
        final_status = await system.get_system_status()
        logger.info("Final System Status:")
        logger.info(f"  Uptime: {final_status['system']['uptime']:.2f} seconds")
        logger.info(f"  Missions Processed: {final_status['statistics']['missions_processed']}")
        logger.info(f"  Events Processed: {final_status['statistics']['events_processed']}")
        logger.info(f"  Plugins Loaded: {final_status['statistics']['plugins_loaded']}")
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
    finally:
        await system.shutdown()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    sys.exit(0)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Modular Physical AI System")
    parser.add_argument("--demo", action="store_true", help="Run system demo")
    parser.add_argument("--config", default="configs/modular.yaml", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.demo:
        await run_demo()
    else:
        # Create and run the main system
        system = ModularPhysicalAI()
        
        if await system.initialize():
            await system.start()
        else:
            logger.error("Failed to initialize system")
            sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
