"""
LLM Integration Module - 실제 언어모델 통합

Physical AI System을 위한 실제 LLM 모델 통합 모듈입니다.
다양한 LLM 백엔드를 지원합니다.
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """LLM 백엔드 추상 클래스"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """LLM 응답 생성"""
        pass
    
    @abstractmethod
    async def generate_structured_response(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """구조화된 응답 생성"""
        pass


class HuggingFaceBackend(LLMBackend):
    """Hugging Face Transformers 백엔드"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    async def initialize(self):
        """모델 초기화"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def generate_response(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """응답 생성"""
        if not self.model or not self.tokenizer:
            await self.initialize()
            
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 입력 프롬프트 제거
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"
    
    async def generate_structured_response(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """구조화된 응답 생성"""
        structured_prompt = f"""
{prompt}

Please respond in the following JSON format:
{json.dumps(schema, indent=2)}

Response:
"""
        
        response = await self.generate_response(structured_prompt, max_length=1024)
        
        try:
            # JSON 응답 추출
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # JSON 형식이 아닌 경우 기본값 반환
                return {"error": "Failed to parse structured response", "raw_response": response}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {"error": f"JSON parsing error: {str(e)}", "raw_response": response}


class OpenAIBackend(LLMBackend):
    """OpenAI API 백엔드"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = None
        
    async def initialize(self):
        """OpenAI 클라이언트 초기화"""
        try:
            import openai
            
            if self.api_key:
                openai.api_key = self.api_key
                
            self.client = openai
            logger.info("OpenAI client initialized")
            
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            raise
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """OpenAI API를 통한 응답 생성"""
        if not self.client:
            await self.initialize()
            
        try:
            response = await self.client.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return f"Error: OpenAI API call failed - {str(e)}"
    
    async def generate_structured_response(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """구조화된 응답 생성"""
        structured_prompt = f"""
{prompt}

Please respond in JSON format following this schema:
{json.dumps(schema, indent=2)}
"""
        
        response = await self.generate_response(structured_prompt)
        
        try:
            # JSON 추출 및 파싱
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "No JSON found in response", "raw_response": response}
                
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing error: {str(e)}", "raw_response": response}


class PhysicalAILLM:
    """Physical AI를 위한 특화된 LLM 인터페이스"""
    
    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.mission_templates = self._load_mission_templates()
        self.physics_knowledge = self._load_physics_knowledge()
        
    async def initialize(self):
        """LLM 초기화"""
        await self.backend.initialize()
        logger.info("Physical AI LLM initialized")
    
    def _load_mission_templates(self) -> Dict[str, str]:
        """미션 분해 템플릿 로드"""
        return {
            "task_decomposition": """
You are a robotics task planner. Decompose the following mission into executable subtasks.

Mission: {mission}

Consider these constraints:
- Robot capabilities: movement, grasping, object recognition, safe manipulation
- Physical limitations: max speed 2.0 m/s, max force 50N, safety margin 0.1m
- Environment: indoor, obstacles possible, human presence

Decompose into subtasks with this format:
- Type: navigation/manipulation/perception
- Action: specific action name  
- Target: what/where to act upon
- Priority: execution order
- Preconditions: what must be true before
- Postconditions: what will be true after

Provide practical, safe, and executable subtasks.
""",
            
            "safety_analysis": """
Analyze the safety implications of this mission: {mission}

Consider:
- Collision risks
- Human safety
- Object fragility  
- Environmental hazards
- Robot limitations

Provide safety constraints and precautions.
""",
            
            "skill_assessment": """
What skills are required for this mission: {mission}

Available skills:
- basic_movement: moving to positions
- object_recognition: identifying objects
- simple_grasp: picking up objects
- precise_manipulation: delicate handling
- collaborative_task: working with humans

Assess which skills are needed and their difficulty level (1-10).
"""
        }
    
    def _load_physics_knowledge(self) -> Dict[str, Any]:
        """물리 지식베이스 로드"""
        return {
            "gravity": 9.81,
            "common_materials": {
                "glass": {"fragile": True, "density": 2.5},
                "metal": {"fragile": False, "density": 7.8},
                "plastic": {"fragile": False, "density": 1.2},
                "wood": {"fragile": False, "density": 0.8}
            },
            "manipulation_forces": {
                "delicate": "< 5N",
                "normal": "5-20N", 
                "strong": "20-50N"
            }
        }
    
    async def decompose_mission(self, mission: str) -> List[Dict[str, Any]]:
        """미션을 서브태스크로 분해"""
        prompt = self.mission_templates["task_decomposition"].format(mission=mission)
        
        schema = {
            "subtasks": [
                {
                    "type": "navigation|manipulation|perception",
                    "action": "specific_action_name",
                    "target": "target_object_or_location", 
                    "priority": 1,
                    "preconditions": ["condition1", "condition2"],
                    "postconditions": ["result1", "result2"],
                    "estimated_duration": 10.0,
                    "difficulty": 3
                }
            ],
            "overall_complexity": 5,
            "success_probability": 0.8
        }
        
        try:
            response = await self.backend.generate_structured_response(prompt, schema)
            
            if "subtasks" in response:
                return response["subtasks"]
            else:
                # Fallback to simple parsing
                return await self._fallback_mission_parsing(mission)
                
        except Exception as e:
            logger.error(f"Mission decomposition failed: {e}")
            return await self._fallback_mission_parsing(mission)
    
    async def _fallback_mission_parsing(self, mission: str) -> List[Dict[str, Any]]:
        """LLM 실패 시 폴백 파싱"""
        mission_lower = mission.lower()
        subtasks = []
        
        # 기본적인 패턴 매칭
        if "pick" in mission_lower and "place" in mission_lower:
            subtasks = [
                {
                    "type": "navigation",
                    "action": "move_to",
                    "target": "object_location",
                    "priority": 1,
                    "preconditions": ["path_clear"],
                    "postconditions": ["at_object_location"],
                    "estimated_duration": 10.0,
                    "difficulty": 2
                },
                {
                    "type": "manipulation", 
                    "action": "grasp",
                    "target": "target_object",
                    "priority": 2,
                    "preconditions": ["object_reachable", "gripper_open"],
                    "postconditions": ["object_grasped"],
                    "estimated_duration": 5.0,
                    "difficulty": 4
                },
                {
                    "type": "navigation",
                    "action": "move_to", 
                    "target": "destination",
                    "priority": 3,
                    "preconditions": ["object_grasped"],
                    "postconditions": ["at_destination"],
                    "estimated_duration": 10.0,
                    "difficulty": 2
                },
                {
                    "type": "manipulation",
                    "action": "place",
                    "target": "destination_surface",
                    "priority": 4,
                    "preconditions": ["at_destination", "object_grasped"],
                    "postconditions": ["object_placed"],
                    "estimated_duration": 3.0,
                    "difficulty": 3
                }
            ]
        else:
            # 기본 탐색 태스크
            subtasks = [
                {
                    "type": "exploration",
                    "action": "explore_environment", 
                    "target": "unknown_area",
                    "priority": 1,
                    "preconditions": ["robot_ready"],
                    "postconditions": ["area_explored"],
                    "estimated_duration": 30.0,
                    "difficulty": 2
                }
            ]
        
        return subtasks
    
    async def analyze_safety_constraints(self, mission: str, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """안전 제약사항 분석"""
        prompt = self.mission_templates["safety_analysis"].format(mission=mission)
        
        try:
            response = await self.backend.generate_response(prompt, max_length=512)
            
            # 기본 안전 제약사항에 LLM 분석 추가
            constraints = {
                "max_velocity": 2.0,
                "max_force": 50.0,
                "safety_distance": 0.1,
                "human_detection": True,
                "collision_avoidance": True,
                "emergency_stop": True,
                "llm_analysis": response
            }
            
            return constraints
            
        except Exception as e:
            logger.error(f"Safety analysis failed: {e}")
            return {
                "max_velocity": 2.0,
                "max_force": 50.0, 
                "safety_distance": 0.1,
                "human_detection": True,
                "collision_avoidance": True,
                "emergency_stop": True,
                "llm_analysis": "Safety analysis unavailable"
            }
    
    async def assess_required_skills(self, mission: str) -> Dict[str, Dict[str, Any]]:
        """필요 스킬 평가"""
        prompt = self.mission_templates["skill_assessment"].format(mission=mission)
        
        try:
            response = await self.backend.generate_response(prompt, max_length=512)
            
            # 기본 스킬 분석
            skills = {
                "basic_movement": {"required": True, "difficulty": 2, "confidence": 0.9},
                "object_recognition": {"required": "object" in mission.lower(), "difficulty": 3, "confidence": 0.8},
                "simple_grasp": {"required": "pick" in mission.lower() or "grasp" in mission.lower(), "difficulty": 4, "confidence": 0.7},
                "precise_manipulation": {"required": "place" in mission.lower() or "delicate" in mission.lower(), "difficulty": 6, "confidence": 0.6},
                "collaborative_task": {"required": "human" in mission.lower() or "person" in mission.lower(), "difficulty": 8, "confidence": 0.5}
            }
            
            # LLM 분석 추가
            skills["llm_analysis"] = {"analysis": response}
            
            return skills
            
        except Exception as e:
            logger.error(f"Skill assessment failed: {e}")
            return {
                "basic_movement": {"required": True, "difficulty": 2, "confidence": 0.9},
                "llm_analysis": {"analysis": "Skill assessment unavailable"}
            }


# Factory 함수들
def create_huggingface_llm(model_name: str = "microsoft/DialoGPT-medium") -> PhysicalAILLM:
    """Hugging Face 기반 LLM 생성"""
    backend = HuggingFaceBackend(model_name)
    return PhysicalAILLM(backend)

def create_openai_llm(api_key: str = None, model: str = "gpt-3.5-turbo") -> PhysicalAILLM:
    """OpenAI 기반 LLM 생성"""
    backend = OpenAIBackend(api_key, model)
    return PhysicalAILLM(backend)


# 테스트 코드
if __name__ == "__main__":
    async def test_llm():
        # Hugging Face 모델 테스트
        print("Testing Hugging Face LLM...")
        hf_llm = create_huggingface_llm()
        await hf_llm.initialize()
        
        mission = "Pick up the red cup and place it on the table"
        
        print(f"Mission: {mission}")
        
        subtasks = await hf_llm.decompose_mission(mission)
        print(f"Subtasks: {len(subtasks)}")
        for i, task in enumerate(subtasks):
            print(f"  {i+1}. {task['action']} -> {task['target']}")
        
        skills = await hf_llm.assess_required_skills(mission)
        print(f"Required skills: {list(skills.keys())}")
        
        safety = await hf_llm.analyze_safety_constraints(mission, subtasks)
        print(f"Safety constraints: {len(safety)} items")
    
    # asyncio.run(test_llm())