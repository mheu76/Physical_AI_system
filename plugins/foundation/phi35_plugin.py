# PLUGIN_NAME: PHI35FoundationPlugin
# PLUGIN_VERSION: 1.0.0
# PLUGIN_DESCRIPTION: Microsoft PHI-3.5 Foundation Model Plugin
# PLUGIN_AUTHOR: Physical AI Team
# PLUGIN_CATEGORY: foundation
# PLUGIN_DEPENDENCIES: 
# PLUGIN_ENTRY_POINT: PHI35FoundationPlugin

"""
PHI-3.5 Foundation Model Plugin
Provides text generation, reasoning, and AI capabilities using Microsoft PHI-3.5
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from core import PluginInterface, event_bus, config_manager, mission_broker

# Import the existing PHI-3.5 integration
try:
    from foundation_model.phi35_integration import PHI35ModelManager
except ImportError:
    # Fallback for when the original module is not available
    class PHI35ModelManager:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.initialized = False
        
        async def initialize(self) -> bool:
            self.initialized = True
            return True
        
        async def generate_text(self, prompt: str, max_length: int = 512) -> str:
            return f"PHI-3.5 Response: {prompt[:50]}..."
        
        async def reason(self, question: str, context: str = "") -> str:
            return f"PHI-3.5 Reasoning: {question[:50]}..."

logger = logging.getLogger(__name__)


class PHI35FoundationPlugin(PluginInterface):
    """PHI-3.5 Foundation Model Plugin"""
    
    def __init__(self):
        self.model_manager: Optional[PHI35ModelManager] = None
        self.config: Dict[str, Any] = {}
        self.capabilities = [
            "text_generation",
            "reasoning", 
            "code_generation",
            "mathematical_reasoning",
            "conversation"
        ]
        self._running = False
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the PHI-3.5 plugin"""
        try:
            self.config = config
            
            # Initialize the model manager
            self.model_manager = PHI35ModelManager(config)
            success = await self.model_manager.initialize()
            
            if success:
                logger.info("PHI-3.5 Foundation Plugin initialized successfully")
                return True
            else:
                logger.error("Failed to initialize PHI-3.5 model manager")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize PHI-3.5 plugin: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the PHI-3.5 plugin"""
        try:
            if not self.model_manager:
                logger.error("PHI-3.5 model manager not initialized")
                return False
            
            # Register capabilities with mission broker
            for capability in self.capabilities:
                await event_bus.publish(event_bus.Event(
                    event_type="plugin.capability.register",
                    data={
                        "plugin_name": "PHI35FoundationPlugin",
                        "capability": capability
                    },
                    source="PHI35FoundationPlugin"
                ))
            
            # Register mission execution handler
            event_bus.register_handler(
                "mission.execute", 
                self._handle_mission_execute, 
                "PHI35FoundationPlugin"
            )
            
            self._running = True
            logger.info("PHI-3.5 Foundation Plugin started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start PHI-3.5 plugin: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the PHI-3.5 plugin"""
        try:
            self._running = False
            
            # Unregister capabilities
            for capability in self.capabilities:
                await event_bus.publish(event_bus.Event(
                    event_type="plugin.capability.unregister",
                    data={
                        "plugin_name": "PHI35FoundationPlugin",
                        "capability": capability
                    },
                    source="PHI35FoundationPlugin"
                ))
            
            # Unregister event handler
            event_bus.unregister_handler("mission.execute", "PHI35FoundationPlugin")
            
            logger.info("PHI-3.5 Foundation Plugin stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop PHI-3.5 plugin: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        return {
            "name": "PHI-3.5 Foundation Plugin",
            "version": "1.0.0",
            "running": self._running,
            "initialized": self.model_manager is not None,
            "capabilities": self.capabilities,
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
                if mission_name == "text_generation":
                    result = await self._generate_text(parameters)
                elif mission_name == "reasoning":
                    result = await self._reason(parameters)
                elif mission_name == "code_generation":
                    result = await self._generate_code(parameters)
                elif mission_name == "mathematical_reasoning":
                    result = await self._mathematical_reasoning(parameters)
                elif mission_name == "conversation":
                    result = await self._conversation(parameters)
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
                source="PHI35FoundationPlugin"
            ))
            
        except Exception as e:
            logger.error(f"Error handling mission execution: {e}")
    
    async def _generate_text(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text using PHI-3.5"""
        try:
            prompt = parameters.get("prompt", "")
            max_length = parameters.get("max_length", 512)
            temperature = parameters.get("temperature", 0.7)
            
            if not prompt:
                raise ValueError("Prompt is required for text generation")
            
            response = await self.model_manager.generate_text(prompt, max_length)
            
            return {
                "generated_text": response,
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature
            }
            
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            raise
    
    async def _reason(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reasoning using PHI-3.5"""
        try:
            question = parameters.get("question", "")
            context = parameters.get("context", "")
            
            if not question:
                raise ValueError("Question is required for reasoning")
            
            response = await self.model_manager.reason(question, context)
            
            return {
                "reasoning": response,
                "question": question,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            raise
    
    async def _generate_code(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code using PHI-3.5"""
        try:
            prompt = parameters.get("prompt", "")
            language = parameters.get("language", "python")
            max_length = parameters.get("max_length", 1024)
            
            if not prompt:
                raise ValueError("Prompt is required for code generation")
            
            # Create a code generation prompt
            code_prompt = f"Generate {language} code for: {prompt}"
            response = await self.model_manager.generate_text(code_prompt, max_length)
            
            return {
                "generated_code": response,
                "language": language,
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            raise
    
    async def _mathematical_reasoning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mathematical reasoning using PHI-3.5"""
        try:
            problem = parameters.get("problem", "")
            
            if not problem:
                raise ValueError("Problem is required for mathematical reasoning")
            
            # Create a math reasoning prompt
            math_prompt = f"Solve this mathematical problem step by step: {problem}"
            response = await self.model_manager.generate_text(math_prompt, 1024)
            
            return {
                "solution": response,
                "problem": problem
            }
            
        except Exception as e:
            logger.error(f"Mathematical reasoning error: {e}")
            raise
    
    async def _conversation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conversation using PHI-3.5"""
        try:
            message = parameters.get("message", "")
            conversation_history = parameters.get("history", [])
            
            if not message:
                raise ValueError("Message is required for conversation")
            
            # Build conversation context
            context = ""
            if conversation_history:
                context = "\n".join([f"User: {h.get('user', '')}\nAssistant: {h.get('assistant', '')}" 
                                   for h in conversation_history[-5:]])  # Last 5 exchanges
            
            response = await self.model_manager.generate_text(f"{context}\nUser: {message}\nAssistant:", 512)
            
            return {
                "response": response,
                "message": message,
                "conversation_history": conversation_history
            }
            
        except Exception as e:
            logger.error(f"Conversation error: {e}")
            raise


# Plugin entry point
PHI35FoundationPlugin = PHI35FoundationPlugin
