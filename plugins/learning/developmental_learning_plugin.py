# PLUGIN_NAME: DevelopmentalLearningPlugin
# PLUGIN_VERSION: 1.0.0
# PLUGIN_DESCRIPTION: Developmental Learning Engine Plugin
# PLUGIN_AUTHOR: Physical AI Team
# PLUGIN_CATEGORY: learning
# PLUGIN_DEPENDENCIES: 
# PLUGIN_ENTRY_POINT: DevelopmentalLearningPlugin

"""
Developmental Learning Plugin
Provides skill acquisition, curriculum learning, and autonomous learning capabilities
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from core import PluginInterface, event_bus, config_manager, mission_broker

# Import the existing developmental learning engine
try:
    from developmental_learning.dev_engine import DevelopmentalEngine, Skill, Experience
except ImportError:
    # Fallback for when the original module is not available
    class Skill:
        def __init__(self, name: str, description: str = ""):
            self.name = name
            self.description = description
            self.level = 0.0
            self.experience = 0.0
    
    class Experience:
        def __init__(self, skill_name: str, value: float, context: str = ""):
            self.skill_name = skill_name
            self.value = value
            self.context = context
            self.timestamp = asyncio.get_event_loop().time()
    
    class DevelopmentalEngine:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.skills: Dict[str, Skill] = {}
            self.experiences: List[Experience] = []
            self.initialized = False
        
        async def initialize(self) -> bool:
            self.initialized = True
            return True
        
        async def acquire_skill(self, skill_name: str, experience_value: float, context: str = "") -> bool:
            if skill_name not in self.skills:
                self.skills[skill_name] = Skill(skill_name)
            
            skill = self.skills[skill_name]
            skill.experience += experience_value
            skill.level = min(100.0, skill.experience / 10.0)
            
            experience = Experience(skill_name, experience_value, context)
            self.experiences.append(experience)
            return True
        
        async def get_skill_level(self, skill_name: str) -> float:
            return self.skills.get(skill_name, Skill(skill_name)).level
        
        async def get_all_skills(self) -> Dict[str, Skill]:
            return self.skills.copy()

logger = logging.getLogger(__name__)


class DevelopmentalLearningPlugin(PluginInterface):
    """Developmental Learning Plugin"""
    
    def __init__(self):
        self.dev_engine: Optional[DevelopmentalEngine] = None
        self.config: Dict[str, Any] = {}
        self.capabilities = [
            "skill_acquisition",
            "curriculum_learning", 
            "autonomous_learning",
            "skill_assessment",
            "learning_progress"
        ]
        self._running = False
        self._autonomous_learning_task: Optional[asyncio.Task] = None
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the developmental learning plugin"""
        try:
            self.config = config
            
            # Initialize the developmental engine
            self.dev_engine = DevelopmentalEngine(config)
            success = await self.dev_engine.initialize()
            
            if success:
                logger.info("Developmental Learning Plugin initialized successfully")
                return True
            else:
                logger.error("Failed to initialize developmental engine")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize developmental learning plugin: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the developmental learning plugin"""
        try:
            if not self.dev_engine:
                logger.error("Developmental engine not initialized")
                return False
            
            # Register capabilities with mission broker
            for capability in self.capabilities:
                await event_bus.publish(event_bus.Event(
                    event_type="plugin.capability.register",
                    data={
                        "plugin_name": "DevelopmentalLearningPlugin",
                        "capability": capability
                    },
                    source="DevelopmentalLearningPlugin"
                ))
            
            # Register mission execution handler
            event_bus.register_handler(
                "mission.execute", 
                self._handle_mission_execute, 
                "DevelopmentalLearningPlugin"
            )
            
            # Start autonomous learning loop
            self._autonomous_learning_task = asyncio.create_task(self._autonomous_learning_loop())
            
            self._running = True
            logger.info("Developmental Learning Plugin started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start developmental learning plugin: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the developmental learning plugin"""
        try:
            self._running = False
            
            # Stop autonomous learning task
            if self._autonomous_learning_task:
                self._autonomous_learning_task.cancel()
                try:
                    await self._autonomous_learning_task
                except asyncio.CancelledError:
                    pass
            
            # Unregister capabilities
            for capability in self.capabilities:
                await event_bus.publish(event_bus.Event(
                    event_type="plugin.capability.unregister",
                    data={
                        "plugin_name": "DevelopmentalLearningPlugin",
                        "capability": capability
                    },
                    source="DevelopmentalLearningPlugin"
                ))
            
            # Unregister event handler
            event_bus.unregister_handler("mission.execute", "DevelopmentalLearningPlugin")
            
            logger.info("Developmental Learning Plugin stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop developmental learning plugin: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        skills = await self.dev_engine.get_all_skills() if self.dev_engine else {}
        return {
            "name": "Developmental Learning Plugin",
            "version": "1.0.0",
            "running": self._running,
            "initialized": self.dev_engine is not None,
            "capabilities": self.capabilities,
            "total_skills": len(skills),
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
                if mission_name == "skill_acquisition":
                    result = await self._acquire_skill(parameters)
                elif mission_name == "curriculum_learning":
                    result = await self._curriculum_learning(parameters)
                elif mission_name == "skill_assessment":
                    result = await self._assess_skill(parameters)
                elif mission_name == "learning_progress":
                    result = await self._get_learning_progress(parameters)
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
                source="DevelopmentalLearningPlugin"
            ))
            
        except Exception as e:
            logger.error(f"Error handling mission execution: {e}")
    
    async def _acquire_skill(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Acquire a skill through learning"""
        try:
            skill_name = parameters.get("skill_name", "")
            experience_value = parameters.get("experience_value", 1.0)
            context = parameters.get("context", "")
            
            if not skill_name:
                raise ValueError("Skill name is required for skill acquisition")
            
            success = await self.dev_engine.acquire_skill(skill_name, experience_value, context)
            
            if success:
                skill_level = await self.dev_engine.get_skill_level(skill_name)
                
                return {
                    "skill_name": skill_name,
                    "experience_gained": experience_value,
                    "current_level": skill_level,
                    "context": context,
                    "success": True
                }
            else:
                raise Exception("Failed to acquire skill")
            
        except Exception as e:
            logger.error(f"Skill acquisition error: {e}")
            raise
    
    async def _curriculum_learning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute curriculum-based learning"""
        try:
            curriculum_name = parameters.get("curriculum_name", "")
            skills_to_learn = parameters.get("skills", [])
            learning_rate = parameters.get("learning_rate", 1.0)
            
            if not curriculum_name or not skills_to_learn:
                raise ValueError("Curriculum name and skills list are required")
            
            results = []
            for skill_name in skills_to_learn:
                # Simulate learning progress
                experience_value = learning_rate * 2.0  # Base experience per skill
                success = await self.dev_engine.acquire_skill(skill_name, experience_value, f"Curriculum: {curriculum_name}")
                
                if success:
                    skill_level = await self.dev_engine.get_skill_level(skill_name)
                    results.append({
                        "skill_name": skill_name,
                        "level": skill_level,
                        "success": True
                    })
                else:
                    results.append({
                        "skill_name": skill_name,
                        "success": False
                    })
            
            return {
                "curriculum_name": curriculum_name,
                "skills_learned": len([r for r in results if r["success"]]),
                "total_skills": len(skills_to_learn),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Curriculum learning error: {e}")
            raise
    
    async def _assess_skill(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current skill level"""
        try:
            skill_name = parameters.get("skill_name", "")
            
            if not skill_name:
                raise ValueError("Skill name is required for assessment")
            
            skill_level = await self.dev_engine.get_skill_level(skill_name)
            
            # Determine skill category based on level
            if skill_level >= 80.0:
                category = "Expert"
            elif skill_level >= 60.0:
                category = "Advanced"
            elif skill_level >= 40.0:
                category = "Intermediate"
            elif skill_level >= 20.0:
                category = "Beginner"
            else:
                category = "Novice"
            
            return {
                "skill_name": skill_name,
                "level": skill_level,
                "category": category,
                "assessment_timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Skill assessment error: {e}")
            raise
    
    async def _get_learning_progress(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get overall learning progress"""
        try:
            skills = await self.dev_engine.get_all_skills()
            
            total_skills = len(skills)
            if total_skills == 0:
                return {
                    "total_skills": 0,
                    "average_level": 0.0,
                    "highest_level": 0.0,
                    "skills": []
                }
            
            skill_levels = [skill.level for skill in skills.values()]
            average_level = sum(skill_levels) / total_skills
            highest_level = max(skill_levels)
            
            # Get top skills
            sorted_skills = sorted(skills.items(), key=lambda x: x[1].level, reverse=True)
            top_skills = [
                {
                    "name": skill.name,
                    "level": skill.level,
                    "experience": skill.experience
                }
                for skill_name, skill in sorted_skills[:5]  # Top 5 skills
            ]
            
            return {
                "total_skills": total_skills,
                "average_level": average_level,
                "highest_level": highest_level,
                "top_skills": top_skills,
                "skills": [
                    {
                        "name": skill.name,
                        "level": skill.level,
                        "experience": skill.experience
                    }
                    for skill in skills.values()
                ]
            }
            
        except Exception as e:
            logger.error(f"Learning progress error: {e}")
            raise
    
    async def _autonomous_learning_loop(self):
        """Autonomous learning loop that continuously improves skills"""
        try:
            while self._running:
                try:
                    # Get current skills
                    skills = await self.dev_engine.get_all_skills()
                    
                    # Find skills that need improvement (below 50% level)
                    skills_to_improve = [
                        skill_name for skill_name, skill in skills.items()
                        if skill.level < 50.0
                    ]
                    
                    if skills_to_improve:
                        # Randomly select a skill to improve
                        import random
                        skill_to_improve = random.choice(skills_to_improve)
                        
                        # Autonomous learning experience
                        experience_value = random.uniform(0.5, 2.0)
                        await self.dev_engine.acquire_skill(
                            skill_to_improve, 
                            experience_value, 
                            "Autonomous learning"
                        )
                        
                        logger.debug(f"Autonomous learning: {skill_to_improve} +{experience_value:.2f} exp")
                        
                        # Publish learning event
                        await event_bus.publish(event_bus.Event(
                            event_type="learning.autonomous",
                            data={
                                "skill_name": skill_to_improve,
                                "experience_gained": experience_value,
                                "learning_type": "autonomous"
                            },
                            source="DevelopmentalLearningPlugin"
                        ))
                    
                    # Wait before next learning cycle
                    await asyncio.sleep(30.0)  # 30 seconds between learning cycles
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in autonomous learning loop: {e}")
                    await asyncio.sleep(60.0)  # Wait longer on error
                    
        except asyncio.CancelledError:
            logger.info("Autonomous learning loop cancelled")
        except Exception as e:
            logger.error(f"Autonomous learning loop failed: {e}")


# Plugin entry point
DevelopmentalLearningPlugin = DevelopmentalLearningPlugin
