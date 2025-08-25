"""
sLM Foundation Model - 미션 해석 및 동작 로직 추론

이 모듈은 자연어 미션을 받아서 구체적인 동작 계획으로 
변환하는 핵심 추론 엔진입니다.
LLM Foundation 학습 모듈과 훈련 모듈이 통합되어 지속적 개선이 가능합니다.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# 상대 임포트를 절대 임포트로 변경 (독립 실행을 위해)
try:
    from .phi35_integration import PHI35ModelManager
    from .llm_learning_module import LLMLearningModule
    from .slm_training_module import SLMTrainingModule, TrainingConfig, TrainingExample
except ImportError:
    # 독립 실행 시 절대 임포트 사용
    from phi35_integration import PHI35ModelManager
    from llm_learning_module import LLMLearningModule
    from slm_training_module import SLMTrainingModule, TrainingConfig, TrainingExample

logger = logging.getLogger(__name__)

@dataclass
class TaskPlan:
    """태스크 계획 데이터 구조"""
    mission: str
    subtasks: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    expected_duration: float
    success_criteria: List[str]

@dataclass
class MotionPrimitive:
    """기본 동작 단위"""
    name: str
    parameters: Dict[str, Any]
    preconditions: List[str]
    postconditions: List[str]
    energy_cost: float

class TaskPlanningModule:
    """태스크 계획 수립 모듈"""
    
    def __init__(self):
        self.motion_primitives = self._load_motion_primitives()
        self.physics_rules = self._load_physics_knowledge()
        self.phi35_ai = None  # PHI-3.5 AI 인터페이스
        self.performance_metrics = {
            "missions_processed": 0,
            "successful_decompositions": 0,
            "average_response_time": 0.0
        }
    
    def _load_motion_primitives(self) -> List[MotionPrimitive]:
        """기본 동작 단위들을 로드"""
        return [
            MotionPrimitive(
                name="grasp",
                parameters={"target": "object", "force": "adaptive"},
                preconditions=["object_reachable", "gripper_open"],
                postconditions=["object_grasped"],
                energy_cost=0.5
            ),
            MotionPrimitive(
                name="move_to",
                parameters={"position": "3d_coordinates", "speed": "optimal"},
                preconditions=["path_clear"],
                postconditions=["at_position"],
                energy_cost=1.0
            ),
            MotionPrimitive(
                name="place",
                parameters={"location": "surface", "orientation": "stable"},
                preconditions=["object_grasped", "location_clear"],
                postconditions=["object_placed"],
                energy_cost=0.3
            )
        ]
    
    def _load_physics_knowledge(self) -> Dict[str, Any]:
        """물리 법칙 지식베이스 로드"""
        return {
            "gravity": 9.81,
            "friction_coefficients": {
                "metal_on_metal": 0.6,
                "rubber_on_concrete": 0.7,
                "plastic_on_wood": 0.3
            },
            "material_properties": {
                "fragile": ["glass", "ceramic", "egg"],
                "heavy": ["metal", "stone"],
                "flexible": ["fabric", "rubber", "paper"]
            }
        }
    
    async def decompose_mission(self, mission: str) -> List[Dict[str, Any]]:
        """미션을 서브태스크로 분해 - PHI-3.5 사용"""
        try:
            # PHI-3.5를 통한 실제 미션 분해
            if hasattr(self, 'phi35_ai') and self.phi35_ai:
                import time
                start_time = time.time()
                
                subtasks = await self.phi35_ai.decompose_mission(mission)
                
                # 성능 메트릭 업데이트
                elapsed_time = time.time() - start_time
                self.performance_metrics["missions_processed"] += 1
                self.performance_metrics["successful_decompositions"] += 1 if subtasks else 0
                
                # 평균 응답 시간 계산
                total_time = self.performance_metrics["average_response_time"] * (self.performance_metrics["missions_processed"] - 1)
                self.performance_metrics["average_response_time"] = (total_time + elapsed_time) / self.performance_metrics["missions_processed"]
                
                logger.info(f"🎯 PHI-3.5 미션 분해 완료: {len(subtasks)}개 서브태스크 ({elapsed_time:.2f}초)")
                return subtasks
            else:
                # PHI-3.5가 없는 경우 폴백 구현
                logger.warning("⚠️  PHI-3.5 없음, 폴백 모드 사용")
                return await self._fallback_mission_decomposition(mission)
                
        except Exception as e:
            logger.error(f"❌ PHI-3.5 미션 분해 실패: {e}")
            return await self._fallback_mission_decomposition(mission)
    
    async def _fallback_mission_decomposition(self, mission: str) -> List[Dict[str, Any]]:
        """폴백 미션 분해 (PHI-3.5 없을 때)"""
        mission_lower = mission.lower()
        
        # 간단한 키워드 기반 분해
        if "pick" in mission_lower and "place" in mission_lower:
            return [
                {
                    "type": "navigation",
                    "action": "move_to",
                    "target": "object_location",
                    "priority": 1,
                    "preconditions": ["robot_ready"],
                    "postconditions": ["at_object"],
                    "estimated_duration": 10.0,
                    "difficulty": 2
                },
                {
                    "type": "manipulation",
                    "action": "grasp",
                    "target": "object",
                    "priority": 2,
                    "preconditions": ["at_object", "object_visible"],
                    "postconditions": ["object_grasped"],
                    "estimated_duration": 5.0,
                    "difficulty": 3
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
                    "target": "surface",
                    "priority": 4,
                    "preconditions": ["at_destination", "object_grasped"],
                    "postconditions": ["object_placed"],
                    "estimated_duration": 3.0,
                    "difficulty": 2
                }
            ]
        
        # 기본 탐색 태스크
        return [
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

class MotionReasoningModule:
    """동작 추론 모듈"""
    
    def __init__(self):
        self.motion_patterns = self._load_motion_patterns()
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_motion_patterns(self) -> Dict[str, Any]:
        """동작 패턴 로드"""
        return {
            "pick_and_place": {
                "sequence": ["approach", "grasp", "lift", "move", "place"],
                "energy_optimization": True,
                "safety_checks": ["collision_detection", "force_monitoring"]
            },
            "exploration": {
                "sequence": ["scan", "move", "scan", "move"],
                "energy_optimization": False,
                "safety_checks": ["obstacle_detection"]
            }
        }
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """최적화 규칙 로드"""
        return {
            "energy_efficiency": {
                "minimize_distance": True,
                "smooth_trajectories": True,
                "optimal_speed": True
            },
            "safety": {
                "maintain_distance": 0.1,
                "slow_approach": True,
                "emergency_stop": True
            }
        }
    
    async def optimize_motion_sequence(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """동작 시퀀스 최적화"""
        optimized_tasks = []
        
        for i, task in enumerate(subtasks):
            optimized_task = task.copy()
            
            # 에너지 최적화
            if self._should_optimize_energy(task):
                optimized_task["energy_efficient"] = True
                optimized_task["speed_factor"] = 0.8  # 안전을 위해 속도 조절
            
            # 안전 최적화
            if self._should_apply_safety(task):
                optimized_task["safety_checks"] = self._get_safety_checks(task)
            
            # 순서 최적화
            optimized_task["priority"] = i + 1
            
            optimized_tasks.append(optimized_task)
        
        return optimized_tasks
    
    def _should_optimize_energy(self, task: Dict[str, Any]) -> bool:
        """에너지 최적화 여부 판단"""
        action = task.get("action", "")
        return action in ["move_to", "grasp", "place"]
    
    def _should_apply_safety(self, task: Dict[str, Any]) -> bool:
        """안전 적용 여부 판단"""
        return True  # 모든 태스크에 안전 적용
    
    def _get_safety_checks(self, task: Dict[str, Any]) -> List[str]:
        """안전 검사 항목 반환"""
        action = task.get("action", "")
        
        if action == "grasp":
            return ["force_monitoring", "collision_detection", "object_stability"]
        elif action == "move_to":
            return ["path_clearance", "obstacle_detection", "speed_monitoring"]
        else:
            return ["general_safety"]

class SLMFoundation:
    """sLM Foundation Model 메인 클래스 - PHI-3.5 내장 + LLM 학습 모듈 + 훈련 모듈"""
    
    def __init__(self, model_type: str = "phi35", **model_config):
        self.task_planner = TaskPlanningModule()
        self.motion_reasoner = MotionReasoningModule()
        self.context_memory = {}
        
        # PHI-3.5 모델 초기화
        self.model_type = model_type
        self.model_config = model_config
        self.phi35_ai = None
        self.llm_learning = None
        self.training_module = None  # 훈련 모듈 추가
        
        # 성능 메트릭
        self.performance_metrics = {
            "missions_processed": 0,
            "successful_decompositions": 0,
            "average_response_time": 0.0,
            "model_info": {},
            "learning_metrics": {},
            "training_metrics": {}  # 훈련 메트릭 추가
        }
    
    async def initialize(self):
        """Foundation Model 초기화 - PHI-3.5 내장 + 훈련 모듈"""
        logger.info("🧠 PHI-3.5 Foundation Model 초기화 중...")
        
        try:
            if self.model_type == "phi35":
                from .phi35_integration import create_phi35_physical_ai
                
                # PHI-3.5 설정
                model_name = self.model_config.get("model_name", "microsoft/Phi-3.5-mini-instruct")
                device = self.model_config.get("device", "auto")
                cache_dir = self.model_config.get("cache_dir", None)
                
                logger.info(f"🔧 PHI-3.5 설정: {model_name} on {device}")
                
                # PHI-3.5 생성 및 초기화
                self.phi35_ai = create_phi35_physical_ai(
                    model_name=model_name,
                    device=device,
                    cache_dir=cache_dir
                )
                
                # 초기화 실행
                success = await self.phi35_ai.initialize()
                if success:
                    logger.info("✅ PHI-3.5 초기화 완료")
                    
                    # TaskPlanningModule에 PHI-3.5 연결
                    self.task_planner.phi35_ai = self.phi35_ai
                    
                    # LLM 학습 모듈 초기화
                    learning_config = self.model_config.get("learning_config", {})
                    self.llm_learning = LLMLearningModule(self.phi35_ai.model_manager, learning_config)
                    await self.llm_learning.initialize()
                    
                    # 훈련 모듈 초기화
                    training_config = TrainingConfig(
                        model_name=model_name,
                        output_dir=self.model_config.get("training_output_dir", "models/slm_foundation"),
                        num_epochs=self.model_config.get("num_epochs", 3),
                        batch_size=self.model_config.get("batch_size", 4),
                        learning_rate=self.model_config.get("learning_rate", 5e-5)
                    )
                    self.training_module = SLMTrainingModule(self.phi35_ai.model_manager, training_config)
                    await self.training_module.initialize()
                    
                    # 모델 정보 저장
                    self.performance_metrics["model_info"] = self.phi35_ai.model_manager.get_model_info()
                    
                else:
                    logger.error("❌ PHI-3.5 초기화 실패")
                    raise Exception("PHI-3.5 initialization failed")
                    
            else:
                logger.error(f"❌ 지원하지 않는 모델 타입: {self.model_type}")
                raise Exception(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"❌ Foundation Model 초기화 실패: {e}")
            logger.warning("⚠️  폴백 모드로 계속 진행합니다 (PHI-3.5 없이)")
            self.phi35_ai = None
        
        logger.info("🎯 sLM Foundation Model 초기화 완료")
        return True
    
    async def interpret_mission(self, mission: str) -> TaskPlan:
        """미션 해석 및 계획 수립"""
        logger.info(f"미션 해석 중: {mission}")
        
        # 1. 미션을 서브태스크로 분해
        subtasks = await self.task_planner.decompose_mission(mission)
        
        # 2. 동작 시퀀스 최적화
        optimized_tasks = await self.motion_reasoner.optimize_motion_sequence(subtasks)
        
        # 3. 제약 조건 분석
        constraints = self._analyze_constraints(mission, optimized_tasks)
        
        # 4. 성공 기준 정의
        success_criteria = self._define_success_criteria(mission)
        
        task_plan = TaskPlan(
            mission=mission,
            subtasks=optimized_tasks,
            constraints=constraints,
            expected_duration=self._estimate_duration(optimized_tasks),
            success_criteria=success_criteria
        )
        
        logger.info(f"태스크 계획 생성 완료: {len(optimized_tasks)}개 서브태스크")
        return task_plan
    
    def _analyze_constraints(self, mission: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """제약 조건 분석"""
        return {
            "max_force": 50.0,  # Newton
            "max_velocity": 2.0,  # m/s
            "safety_distance": 0.1,  # m
            "timeout": 300.0,  # seconds
            "energy_limit": 100.0  # Joules
        }
    
    def _define_success_criteria(self, mission: str) -> List[str]:
        """성공 기준 정의"""
        mission_lower = mission.lower()
        
        if "pick" in mission_lower and "place" in mission_lower:
            return ["object_picked", "object_placed", "mission_completed"]
        elif "organize" in mission_lower:
            return ["items_organized", "space_neat", "mission_completed"]
        elif "clean" in mission_lower:
            return ["area_cleaned", "items_sorted", "mission_completed"]
        else:
            return ["mission_completed"]
    
    def _estimate_duration(self, tasks: List[Dict[str, Any]]) -> float:
        """예상 소요 시간 계산"""
        total_duration = 0.0
        
        for task in tasks:
            duration = task.get("estimated_duration", 10.0)
            total_duration += duration
        
        # 안전 마진 추가 (20%)
        return total_duration * 1.2
    
    async def process_mission_with_learning(self, 
                                          mission: str, 
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """학습이 포함된 미션 처리"""
        if context is None:
            context = {}
        
        try:
            # 1. 미션 분해
            subtasks = await self.task_planner.decompose_mission(mission)
            
            # 2. 동작 최적화
            optimized_plan = await self.motion_reasoner.optimize_motion_sequence(subtasks)
            
            # 3. 실행 결과 시뮬레이션 (실제로는 Agent Executor에서 실행)
            execution_result = await self._simulate_execution(optimized_plan, context)
            
            # 4. 학습 수행
            learning_value = 0.0
            if self.llm_learning:
                learning_value = await self.llm_learning.learn_from_experience(
                    mission=mission,
                    context=context,
                    generated_plan={"subtasks": optimized_plan},
                    execution_result=execution_result
                )
                
                # 학습 메트릭 업데이트
                self.performance_metrics["learning_metrics"] = await self.llm_learning.get_learning_insights()
            
            # 5. 훈련 예제 추가
            if self.training_module:
                training_example = TrainingExample(
                    mission=mission,
                    context=context,
                    subtasks=optimized_plan,
                    constraints=self._analyze_constraints(mission, optimized_plan),
                    success_criteria=self._define_success_criteria(mission),
                    execution_result=execution_result,
                    learning_value=learning_value
                )
                await self.training_module.add_training_example(training_example)
            
            return {
                "success": True,
                "mission": mission,
                "subtasks": subtasks,
                "optimized_plan": optimized_plan,
                "execution_result": execution_result,
                "learning_value": learning_value,
                "performance_metrics": self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ 미션 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "mission": mission
            }
    
    async def _simulate_execution(self, plan: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """실행 결과 시뮬레이션"""
        import random
        
        # 시뮬레이션된 실행 결과
        success = random.random() > 0.1  # 90% 성공률
        execution_time = sum(task.get("estimated_duration", 10.0) for task in plan)
        
        # 성능 메트릭 시뮬레이션
        performance_metrics = {
            "efficiency": random.uniform(0.6, 0.9),
            "accuracy": random.uniform(0.7, 0.95),
            "safety_score": random.uniform(0.8, 1.0),
            "energy_consumption": execution_time * random.uniform(0.5, 1.5)
        }
        
        return {
            "success": success,
            "execution_time": execution_time,
            "performance_metrics": performance_metrics,
            "errors": [] if success else ["simulated_error"]
        }
    
    # 훈련 관련 메서드들 추가
    async def train_model(self, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """모델 훈련 실행"""
        if not self.training_module:
            return {"success": False, "error": "훈련 모듈이 초기화되지 않았습니다."}
        
        logger.info("🚀 sLM Foundation Model 훈련 시작")
        result = await self.training_module.train_model(resume_from_checkpoint)
        
        if result["success"]:
            # 훈련 메트릭 업데이트
            self.performance_metrics["training_metrics"] = await self.training_module.get_training_status()
            logger.info("✅ 모델 훈련 완료")
        else:
            logger.error(f"❌ 모델 훈련 실패: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def evaluate_model(self, test_examples: List[TrainingExample] = None) -> Dict[str, Any]:
        """모델 성능 평가"""
        if not self.training_module:
            return {"success": False, "error": "훈련 모듈이 초기화되지 않았습니다."}
        
        logger.info("🔍 모델 성능 평가 시작")
        result = await self.training_module.evaluate_model(test_examples)
        
        if result["success"]:
            logger.info(f"📊 평가 결과: 정확도 {result['accuracy']:.3f}")
        else:
            logger.error(f"❌ 평가 실패: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def get_training_status(self) -> Dict[str, Any]:
        """훈련 상태 조회"""
        if not self.training_module:
            return {"error": "훈련 모듈이 초기화되지 않았습니다."}
        
        return await self.training_module.get_training_status()
    
    async def export_trained_model(self, export_path: str = None) -> Dict[str, Any]:
        """훈련된 모델 내보내기"""
        if not self.training_module:
            return {"success": False, "error": "훈련 모듈이 초기화되지 않았습니다."}
        
        logger.info("💾 훈련된 모델 내보내기 시작")
        result = await self.training_module.export_model(export_path)
        
        if result["success"]:
            logger.info(f"✅ 모델 내보내기 완료: {result['export_path']}")
        else:
            logger.error(f"❌ 모델 내보내기 실패: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """학습 인사이트 제공"""
        if not self.llm_learning:
            return {"error": "LLM 학습 모듈이 초기화되지 않았습니다."}
        
        return await self.llm_learning.get_learning_insights()
    
    async def optimize_learning_strategy(self) -> Dict[str, Any]:
        """학습 전략 최적화"""
        if not self.llm_learning:
            return {"error": "LLM 학습 모듈이 초기화되지 않았습니다."}
        
        return await self.llm_learning.optimize_learning_strategy()
    
    async def get_knowledge_patterns(self) -> Dict[str, Any]:
        """지식 패턴 조회"""
        if not self.llm_learning:
            return {"error": "LLM 학습 모듈이 초기화되지 않았습니다."}
        
        # 지식 패턴 정보 반환
        patterns = self.llm_learning.knowledge_patterns
        total_patterns = len(patterns)
        
        pattern_list = []
        for pattern_id, pattern in patterns.items():
            pattern_list.append({
                "id": pattern_id,
                "type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "usage_count": pattern.usage_count,
                "description": pattern.pattern_data.get("description", "No description")
            })
        
        return {
            "total_patterns": total_patterns,
            "patterns": pattern_list
        }
    
    async def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """사용자 입력에 대한 응답 생성"""
        try:
            if self.phi35_ai and hasattr(self.phi35_ai, 'generate_response'):
                return await self.phi35_ai.generate_response(user_input, context or {})
            else:
                # 폴백: 간단한 규칙 기반 응답
                if "안녕" in user_input or "hello" in user_input.lower():
                    return "안녕하세요! Physical AI Code입니다. 무엇을 도와드릴까요?"
                elif "상태" in user_input or "status" in user_input.lower():
                    return "시스템이 정상적으로 작동 중입니다."
                elif "도구" in user_input or "tool" in user_input.lower():
                    return "사용 가능한 도구들을 확인하겠습니다."
                elif "미션" in user_input or "mission" in user_input.lower():
                    return f"미션을 처리하겠습니다: {user_input}"
                else:
                    return f"입력을 받았습니다: {user_input}. 처리 중입니다."
        except Exception as e:
            logger.error(f"응답 생성 중 오류: {e}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

# 테스트 코드
if __name__ == "__main__":
    async def test_slm_foundation():
        """sLM Foundation Model 테스트"""
        logger.info("🧠 sLM Foundation Model 테스트")
        
        # Foundation Model 초기화
        foundation = SLMFoundation(
            model_type="phi35",
            model_name="microsoft/Phi-3.5-mini-instruct",
            device="auto",
            learning_config={"enabled": True, "learning_rate": 0.01},
            training_output_dir="models/slm_foundation",
            num_epochs=2,
            batch_size=2,
            learning_rate=1e-4
        )
        
        try:
            await foundation.initialize()
            
            # 미션 처리 테스트
            test_missions = [
                "Pick up the red cup and place it on the table",
                "Organize the books on the shelf by size"
            ]
            
            for mission in test_missions:
                logger.info(f"\n📋 미션 처리: {mission}")
                result = await foundation.process_mission_with_learning(
                    mission=mission,
                    context={"environment": "simple", "safety_level": "normal"}
                )
                
                if result['success']:
                    logger.info(f"✅ 처리 완료: {len(result['subtasks'])}개 서브태스크")
                    logger.info(f"📊 학습 가치: {result['learning_value']:.3f}")
                else:
                    logger.error(f"❌ 처리 실패: {result.get('error', 'Unknown error')}")
            
            # 훈련 상태 확인
            training_status = await foundation.get_training_status()
            logger.info(f"\n📊 훈련 상태: {training_status}")
            
            # 학습 인사이트 확인
            insights = await foundation.get_learning_insights()
            logger.info(f"\n🧠 학습 인사이트: {insights}")
            
        except Exception as e:
            logger.error(f"❌ 테스트 실패: {e}")
        
        logger.info("✅ sLM Foundation Model 테스트 완료")
    
    asyncio.run(test_slm_foundation())
