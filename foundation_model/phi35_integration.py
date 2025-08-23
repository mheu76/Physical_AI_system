"""
PHI-3.5 Integration Module - Microsoft PHI-3.5 모델 내장

Physical AI System에 최적화된 PHI-3.5 소형 언어모델 통합
"""

import asyncio
import json
import re
import torch
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class PHI35ModelManager:
    """PHI-3.5 모델 관리자"""
    
    def __init__(self, 
                 model_name: str = "microsoft/Phi-3.5-mini-instruct",
                 device: str = "auto",
                 max_length: int = 2048,
                 cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # 모델 관련 변수
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
        # 물리 AI 특화 프롬프트 템플릿
        self.system_prompt = self._create_system_prompt()
        
    def _setup_device(self, device: str) -> str:
        """디바이스 설정 (양자화 최적화)"""
        if device == "auto":
            if torch.cuda.is_available():
                # GPU 메모리 확인
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    logger.info(f"GPU detected: {total_memory/1024**3:.1f}GB total memory")
                    # 3GB 이상이면 양자화와 함께 GPU 사용
                    if total_memory >= 3 * 1024**3:  # 3GB 이상
                        logger.info("Using GPU with 8-bit quantization")
                        return "cuda"
                    else:
                        logger.warning(f"GPU memory insufficient ({total_memory/1024**3:.1f}GB). Using CPU.")
                        return "cpu"
                except Exception:
                    return "cpu"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps" 
            else:
                return "cpu"
        return device
    
    def _create_system_prompt(self) -> str:
        """Physical AI 특화 시스템 프롬프트"""
        return """You are PHI-3.5, a specialized AI assistant for Physical AI robotics systems.

CORE CAPABILITIES:
- Mission planning and task decomposition for robotics
- Physics-aware motion reasoning
- Safety constraint analysis  
- Skill requirement assessment
- Real-time decision making for embodied AI

KNOWLEDGE DOMAINS:
- Robotics and automation
- Physics and kinematics
- Safety protocols
- Human-robot interaction
- Sensor fusion and control systems

RESPONSE GUIDELINES:
- Be precise and actionable
- Consider physical constraints
- Prioritize safety
- Provide structured outputs
- Use technical robotics terminology when appropriate

You are embedded in a Physical AI system that controls real robots. Your responses directly impact physical actions."""

    async def initialize(self) -> bool:
        """PHI-3.5 모델 초기화"""
        try:
            logger.info(f"Initializing PHI-3.5 model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Transformers 라이브러리 import
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # 토크나이저 로드
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드 (메모리 최적화)
            logger.info("Loading model...")
            
            # 양자화 및 메모리 최적화 설정
            if self.device == "cpu":
                model_kwargs = {
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,
                    "device_map": None,
                    "low_cpu_mem_usage": True,
                }
            else:
                # GPU 모드: 8bit 양자화 적용
                try:
                    from transformers import BitsAndBytesConfig
                    
                    # 8bit 양자화 설정
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                    
                    model_kwargs = {
                        "cache_dir": self.cache_dir,
                        "trust_remote_code": True,
                        "quantization_config": quantization_config,
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                    }
                    logger.info("Using 8-bit quantization for GPU optimization")
                    
                except ImportError:
                    logger.warning("BitsAndBytes not available, using standard FP16")
                    model_kwargs = {
                        "cache_dir": self.cache_dir,
                        "trust_remote_code": True,
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                        "max_memory": {0: "3GB"},  # GPU 메모리 제한
                    }
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU out of memory, falling back to CPU...")
                self.device = "cpu"
                model_kwargs = {
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,
                    "device_map": None,
                    "low_cpu_mem_usage": True,
                }
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            
            # CPU로 이동 (필요한 경우)
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            elif self.device == "mps":
                self.model = self.model.to("mps")
            
            # 평가 모드로 설정
            self.model.eval()
            
            self.is_initialized = True
            logger.info("PHI-3.5 model initialized successfully")
            
            # 초기 테스트
            test_result = await self.generate_response("Hello, are you ready?")
            logger.info(f"Initialization test: {test_result[:100]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PHI-3.5: {e}")
            self.is_initialized = False
            return False
    
    async def generate_response(self, 
                              prompt: str,
                              max_new_tokens: int = 512,
                              temperature: float = 0.7,
                              top_p: float = 0.9,
                              do_sample: bool = True) -> str:
        """PHI-3.5를 사용한 응답 생성"""
        if not self.is_initialized:
            await self.initialize()
            
        if not self.is_initialized:
            return "Error: PHI-3.5 model not initialized"
        
        try:
            # 전체 프롬프트 구성
            full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nAssistant:"
            
            # 토큰화
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - max_new_tokens
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성 파라미터 (DynamicCache 호환성 문제 해결)
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "attention_mask": inputs.get("attention_mask"),
                "use_cache": False,  # DynamicCache 문제 해결
            }
            
            # 응답 생성
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        **generation_kwargs
                    )
                except AttributeError as e:
                    if "seen_tokens" in str(e):
                        # DynamicCache 문제 해결을 위한 대안 방법
                        generation_kwargs["use_cache"] = False
                        generation_kwargs["past_key_values"] = None
                        outputs = self.model.generate(
                            inputs["input_ids"],
                            **generation_kwargs
                        )
                    else:
                        raise e
            
            # 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: Generation failed - {str(e)}"
    
    async def generate_structured_response(self, 
                                         prompt: str, 
                                         schema: Dict[str, Any],
                                         max_retries: int = 3) -> Dict[str, Any]:
        """구조화된 JSON 응답 생성"""
        
        structured_prompt = f"""
{prompt}

Please respond in JSON format following this exact schema:
{json.dumps(schema, indent=2)}

Important: 
- Return ONLY valid JSON
- Follow the schema exactly
- Include all required fields
- Use appropriate data types

JSON Response:"""

        for attempt in range(max_retries):
            try:
                response = await self.generate_response(
                    structured_prompt,
                    max_new_tokens=1024,
                    temperature=0.3  # 낮은 온도로 더 일관된 출력
                )
                
                # JSON 추출
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_response = json.loads(json_str)
                    
                    # 스키마 검증 (기본적)
                    if self._validate_schema(parsed_response, schema):
                        return parsed_response
                    else:
                        logger.warning(f"Schema validation failed (attempt {attempt + 1})")
                
                logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {response[:200]}...")
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Structured response error (attempt {attempt + 1}): {e}")
        
        # 모든 시도 실패 시 기본값 반환
        logger.error("All structured response attempts failed")
        return {
            "error": "Failed to generate structured response", 
            "raw_response": response if 'response' in locals() else "No response generated",
            "attempts": max_retries
        }
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """기본적인 스키마 검증"""
        try:
            # 필수 키 확인
            for key in schema:
                if key not in data:
                    return False
            return True
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "initialized": self.is_initialized,
            "max_length": self.max_length,
            "parameters": self.model.num_parameters() if self.model else None,
            "dtype": str(self.model.dtype) if self.model else None
        }


class PhysicalAI_PHI35:
    """Physical AI를 위한 PHI-3.5 특화 인터페이스"""
    
    def __init__(self, model_manager: PHI35ModelManager):
        self.model_manager = model_manager
        self.mission_templates = self._load_mission_templates()
        self.physics_knowledge = self._load_physics_knowledge()
    
    async def initialize(self):
        """PHI-3.5 초기화"""
        success = await self.model_manager.initialize()
        if success:
            logger.info("PhysicalAI PHI-3.5 interface initialized")
        else:
            logger.error("Failed to initialize PhysicalAI PHI-3.5 interface")
        return success
    
    def _load_mission_templates(self) -> Dict[str, str]:
        """미션 분해 템플릿"""
        return {
            "task_decomposition": """
ROBOTICS MISSION ANALYSIS

Mission: "{mission}"

ROBOT CAPABILITIES:
- 6-DOF manipulator arm
- Parallel gripper (max 50N force)
- RGB-D camera + tactile sensors
- Mobile base (max 2.0 m/s)
- Real-time safety monitoring

ENVIRONMENT:
- Indoor workspace
- Human presence possible
- Mixed objects (fragile/robust)
- Obstacles and clutter

TASK: Decompose this mission into executable subtasks.

For each subtask, specify:
- Type: navigation | manipulation | perception
- Action: specific robot action
- Target: what to act upon
- Preconditions: requirements before execution
- Postconditions: expected results
- Duration: estimated time (seconds)
- Difficulty: 1-10 scale
- Safety_risk: low | medium | high

Provide practical, safe, and physically realizable subtasks.
""",
            
            "safety_analysis": """
SAFETY ANALYSIS FOR ROBOTICS MISSION

Mission: "{mission}"

ANALYZE FOR:
1. Collision risks with objects/humans
2. Object fragility concerns  
3. Force/speed limitations needed
4. Human safety zones required
5. Emergency stop scenarios

CONSTRAINTS TO SET:
- Maximum velocity limits
- Force/torque restrictions  
- Safety distances
- Sensor monitoring requirements
- Human detection protocols

Provide specific safety constraints and precautions.
""",
            
            "skill_assessment": """
SKILL REQUIREMENT ANALYSIS

Mission: "{mission}"

AVAILABLE ROBOT SKILLS:
1. basic_movement (difficulty: 2) - Navigate to positions
2. object_recognition (difficulty: 4) - Identify and locate objects
3. simple_grasp (difficulty: 5) - Pick up regular objects  
4. precise_manipulation (difficulty: 7) - Delicate object handling
5. collaborative_task (difficulty: 9) - Human-robot cooperation
6. adaptive_control (difficulty: 8) - Dynamic response to changes

ASSESS:
- Which skills are required?
- Difficulty level for each skill (1-10)
- Success probability estimate
- Potential failure modes
- Alternative approaches

Provide detailed skill requirements analysis.
"""
        }
    
    def _load_physics_knowledge(self) -> Dict[str, Any]:
        """물리 지식베이스"""
        return {
            "kinematics": {
                "max_joint_velocity": "3.14 rad/s",
                "max_linear_velocity": "2.0 m/s", 
                "max_acceleration": "5.0 m/s²"
            },
            "dynamics": {
                "max_payload": "5.0 kg",
                "max_gripper_force": "50.0 N",
                "safety_margin": "0.1 m"
            },
            "materials": {
                "glass": {"fragile": True, "max_force": "5N"},
                "ceramic": {"fragile": True, "max_force": "8N"}, 
                "plastic": {"fragile": False, "max_force": "30N"},
                "metal": {"fragile": False, "max_force": "50N"},
                "fabric": {"deformable": True, "max_force": "10N"}
            },
            "environmental": {
                "gravity": 9.81,
                "air_resistance": "negligible",
                "friction_coefficients": {
                    "rubber_concrete": 0.7,
                    "metal_metal": 0.6,
                    "plastic_wood": 0.3
                }
            }
        }
    
    async def decompose_mission(self, mission: str) -> List[Dict[str, Any]]:
        """PHI-3.5를 사용한 미션 분해"""
        prompt = self.mission_templates["task_decomposition"].format(mission=mission)
        
        schema = {
            "mission_analysis": {
                "complexity": "low|medium|high",
                "estimated_duration": 60.0,
                "success_probability": 0.85
            },
            "subtasks": [
                {
                    "type": "navigation|manipulation|perception",
                    "action": "specific_action_name",
                    "target": "target_description",
                    "preconditions": ["condition1", "condition2"],
                    "postconditions": ["result1", "result2"],
                    "duration": 10.0,
                    "difficulty": 5,
                    "safety_risk": "low|medium|high",
                    "priority": 1
                }
            ]
        }
        
        try:
            response = await self.model_manager.generate_structured_response(prompt, schema)
            
            if "subtasks" in response and isinstance(response["subtasks"], list):
                return response["subtasks"]
            else:
                logger.warning("Invalid subtasks format, using fallback")
                return await self._fallback_mission_parsing(mission)
                
        except Exception as e:
            logger.error(f"PHI-3.5 mission decomposition failed: {e}")
            return await self._fallback_mission_parsing(mission)
    
    async def analyze_safety_constraints(self, mission: str) -> Dict[str, Any]:
        """PHI-3.5를 사용한 안전 분석"""
        prompt = self.mission_templates["safety_analysis"].format(mission=mission)
        
        try:
            response = await self.model_manager.generate_response(prompt, max_new_tokens=512)
            
            # 기본 안전 제약사항에 PHI-3.5 분석 추가
            constraints = {
                "max_velocity": 2.0,
                "max_force": 50.0,
                "safety_distance": 0.1,
                "human_detection": True,
                "collision_avoidance": True,
                "emergency_stop": True,
                "phi35_analysis": response,
                "risk_assessment": self._extract_risk_level(response)
            }
            
            return constraints
            
        except Exception as e:
            logger.error(f"PHI-3.5 safety analysis failed: {e}")
            return self._default_safety_constraints()
    
    async def assess_required_skills(self, mission: str) -> Dict[str, Dict[str, Any]]:
        """PHI-3.5를 사용한 스킬 요구사항 분석"""
        prompt = self.mission_templates["skill_assessment"].format(mission=mission)
        
        try:
            response = await self.model_manager.generate_response(prompt, max_new_tokens=512)
            
            # 기본 스킬 매핑
            skills = {
                "basic_movement": {"required": True, "difficulty": 2, "confidence": 0.95},
                "object_recognition": {"required": self._mission_needs_recognition(mission), "difficulty": 4, "confidence": 0.85},
                "simple_grasp": {"required": self._mission_needs_grasp(mission), "difficulty": 5, "confidence": 0.80},
                "precise_manipulation": {"required": self._mission_needs_precision(mission), "difficulty": 7, "confidence": 0.70},
                "collaborative_task": {"required": self._mission_needs_collaboration(mission), "difficulty": 9, "confidence": 0.60},
                "phi35_analysis": {"analysis": response}
            }
            
            return skills
            
        except Exception as e:
            logger.error(f"PHI-3.5 skill assessment failed: {e}")
            return self._default_skill_assessment()
    
    async def _fallback_mission_parsing(self, mission: str) -> List[Dict[str, Any]]:
        """PHI-3.5 실패시 폴백 파싱"""
        mission_lower = mission.lower()
        
        if any(word in mission_lower for word in ["pick", "grasp", "grab"]) and \
           any(word in mission_lower for word in ["place", "put", "drop"]):
            return [
                {
                    "type": "navigation",
                    "action": "move_to",
                    "target": "object_location",
                    "preconditions": ["path_clear", "robot_ready"],
                    "postconditions": ["at_object_location"],
                    "duration": 10.0,
                    "difficulty": 2,
                    "safety_risk": "low",
                    "priority": 1
                },
                {
                    "type": "perception",
                    "action": "locate_object",
                    "target": "target_object",
                    "preconditions": ["at_object_location", "camera_active"],
                    "postconditions": ["object_located"],
                    "duration": 5.0,
                    "difficulty": 4,
                    "safety_risk": "low", 
                    "priority": 2
                },
                {
                    "type": "manipulation",
                    "action": "grasp",
                    "target": "target_object",
                    "preconditions": ["object_located", "gripper_open"],
                    "postconditions": ["object_grasped"],
                    "duration": 8.0,
                    "difficulty": 5,
                    "safety_risk": "medium",
                    "priority": 3
                },
                {
                    "type": "navigation",
                    "action": "move_to",
                    "target": "destination",
                    "preconditions": ["object_grasped"],
                    "postconditions": ["at_destination"],
                    "duration": 12.0,
                    "difficulty": 3,
                    "safety_risk": "medium",
                    "priority": 4
                },
                {
                    "type": "manipulation",
                    "action": "place",
                    "target": "destination_surface",
                    "preconditions": ["at_destination", "object_grasped"],
                    "postconditions": ["object_placed"],
                    "duration": 6.0,
                    "difficulty": 4,
                    "safety_risk": "low",
                    "priority": 5
                }
            ]
        else:
            return [
                {
                    "type": "exploration",
                    "action": "explore_environment",
                    "target": "workspace",
                    "preconditions": ["robot_ready"],
                    "postconditions": ["area_explored"],
                    "duration": 30.0,
                    "difficulty": 3,
                    "safety_risk": "low",
                    "priority": 1
                }
            ]
    
    def _extract_risk_level(self, analysis: str) -> str:
        """분석에서 위험도 추출"""
        analysis_lower = analysis.lower()
        if any(word in analysis_lower for word in ["high risk", "dangerous", "hazard"]):
            return "high"
        elif any(word in analysis_lower for word in ["medium risk", "caution", "careful"]):
            return "medium"
        else:
            return "low"
    
    def _mission_needs_recognition(self, mission: str) -> bool:
        """객체 인식이 필요한 미션인지 판단"""
        keywords = ["object", "cup", "book", "bottle", "box", "item", "thing"]
        return any(keyword in mission.lower() for keyword in keywords)
    
    def _mission_needs_grasp(self, mission: str) -> bool:
        """그립이 필요한 미션인지 판단"""
        keywords = ["pick", "grab", "grasp", "take", "hold", "lift"]
        return any(keyword in mission.lower() for keyword in keywords)
    
    def _mission_needs_precision(self, mission: str) -> bool:
        """정밀 조작이 필요한 미션인지 판단"""
        keywords = ["delicate", "precise", "careful", "fragile", "gently", "slowly"]
        return any(keyword in mission.lower() for keyword in keywords)
    
    def _mission_needs_collaboration(self, mission: str) -> bool:
        """협업이 필요한 미션인지 판단"""
        keywords = ["human", "person", "user", "together", "help", "assist"]
        return any(keyword in mission.lower() for keyword in keywords)
    
    def _default_safety_constraints(self) -> Dict[str, Any]:
        """기본 안전 제약사항"""
        return {
            "max_velocity": 2.0,
            "max_force": 50.0,
            "safety_distance": 0.1,
            "human_detection": True,
            "collision_avoidance": True,
            "emergency_stop": True,
            "phi35_analysis": "Safety analysis unavailable",
            "risk_assessment": "medium"
        }
    
    def _default_skill_assessment(self) -> Dict[str, Dict[str, Any]]:
        """기본 스킬 평가"""
        return {
            "basic_movement": {"required": True, "difficulty": 2, "confidence": 0.95},
            "phi35_analysis": {"analysis": "Skill assessment unavailable"}
        }


# Factory 함수
def create_phi35_physical_ai(model_name: str = "microsoft/Phi-3.5-mini-instruct",
                            device: str = "auto",
                            cache_dir: Optional[str] = None) -> PhysicalAI_PHI35:
    """PHI-3.5 기반 Physical AI 생성"""
    model_manager = PHI35ModelManager(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir
    )
    return PhysicalAI_PHI35(model_manager)


# 테스트 코드
if __name__ == "__main__":
    async def test_phi35():
        print("Testing PHI-3.5 Physical AI Integration...")
        
        # PHI-3.5 생성 및 초기화
        phi35_ai = create_phi35_physical_ai()
        success = await phi35_ai.initialize()
        
        if not success:
            print("❌ PHI-3.5 initialization failed")
            return
        
        print("✅ PHI-3.5 initialized successfully")
        
        # 테스트 미션
        mission = "Pick up the red cup and place it gently on the wooden table"
        print(f"\nTesting mission: {mission}")
        
        # 미션 분해
        print("\n🎯 Mission decomposition...")
        subtasks = await phi35_ai.decompose_mission(mission)
        print(f"Generated {len(subtasks)} subtasks:")
        for i, task in enumerate(subtasks):
            print(f"  {i+1}. {task.get('action', 'unknown')} -> {task.get('target', 'unknown')} (difficulty: {task.get('difficulty', 'N/A')})")
        
        # 안전 분석
        print("\n🛡️ Safety analysis...")
        safety = await phi35_ai.analyze_safety_constraints(mission)
        print(f"Risk level: {safety.get('risk_assessment', 'unknown')}")
        
        # 스킬 평가
        print("\n🌱 Skill assessment...")
        skills = await phi35_ai.assess_required_skills(mission)
        required_skills = [name for name, info in skills.items() 
                          if isinstance(info, dict) and info.get('required', False)]
        print(f"Required skills: {required_skills}")
        
        print("\n🎉 PHI-3.5 integration test completed!")
    
    # asyncio.run(test_phi35())