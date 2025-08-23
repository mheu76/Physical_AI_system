"""
sLM Foundation Model - 미션 해석 및 동작 로직 추론

이 모듈은 자연어 미션을 받아서 구체적인 동작 계획으로 
변환하는 핵심 추론 엔진입니다.
"""

import asyncio
import json
from typing import Dict, List, Any
from dataclasses import dataclass

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
                
                print(f"🎯 PHI-3.5 미션 분해 완료: {len(subtasks)}개 서브태스크 ({elapsed_time:.2f}초)")
                return subtasks
            else:
                # PHI-3.5가 없는 경우 폴백 구현
                print("⚠️  PHI-3.5 없음, 폴백 모드 사용")
                return await self._fallback_mission_decomposition(mission)
                
        except Exception as e:
            print(f"❌ PHI-3.5 미션 분해 실패: {e}")
            return await self._fallback_mission_decomposition(mission)
    
    async def _fallback_mission_decomposition(self, mission: str) -> List[Dict[str, Any]]:
        """LLM 실패시 폴백 미션 분해"""
        if "pick up" in mission.lower() and "place" in mission.lower():
            return [
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
        self.energy_optimizer = EnergyOptimizer()
        self.physics_simulator = PhysicsSimulator()
    
    async def optimize_motion_sequence(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """동작 시퀀스 최적화"""
        optimized_sequence = []
        
        for task in subtasks:
            # 에너지 효율성 고려
            energy_cost = await self.energy_optimizer.calculate_cost(task)
            
            # 물리적 제약 검증
            feasible = await self.physics_simulator.validate_motion(task)
            
            if feasible:
                task["energy_cost"] = energy_cost
                task["validated"] = True
                optimized_sequence.append(task)
            else:
                # 대안 동작 생성
                alternative = await self._generate_alternative(task)
                optimized_sequence.append(alternative)
        
        return optimized_sequence
    
    async def _generate_alternative(self, failed_task: Dict[str, Any]) -> Dict[str, Any]:
        """실패한 태스크에 대한 대안 생성"""
        # 더 안전하고 확실한 대안 동작 생성
        return {
            **failed_task,
            "modified": True,
            "safety_margin": 1.5,
            "speed_reduction": 0.7
        }

class EnergyOptimizer:
    """에너지 효율성 최적화"""
    
    async def calculate_cost(self, task: Dict[str, Any]) -> float:
        """태스크의 에너지 비용 계산"""
        base_cost = {
            "move_to": 1.0,
            "grasp": 0.5,
            "place": 0.3,
            "explore": 2.0
        }.get(task.get("action", "unknown"), 1.0)
        
        # 거리, 속도, 부하 등을 고려한 동적 계산
        # 여기서는 단순화된 버전
        return base_cost

class PhysicsSimulator:
    """물리 법칙 기반 동작 검증"""
    
    async def validate_motion(self, task: Dict[str, Any]) -> bool:
        """물리적 실현 가능성 검증"""
        # 실제로는 물리 엔진을 사용한 시뮬레이션
        # 관절 한계, 충돌, 안정성 등 검증
        
        # 여기서는 기본적인 검증만 수행
        action = task.get("action", "")
        
        # 기본 동작들은 대부분 실현 가능하다고 가정
        return action in ["move_to", "grasp", "place", "explore"]

class SLMFoundation:
    """sLM Foundation Model 메인 클래스 - PHI-3.5 내장"""
    
    def __init__(self, model_type: str = "phi35", **model_config):
        self.task_planner = TaskPlanningModule()
        self.motion_reasoner = MotionReasoningModule()
        self.context_memory = {}
        
        # PHI-3.5 모델 초기화
        self.model_type = model_type
        self.model_config = model_config
        self.phi35_ai = None
        
        # 성능 메트릭
        self.performance_metrics = {
            "missions_processed": 0,
            "successful_decompositions": 0,
            "average_response_time": 0.0,
            "model_info": {}
        }
    
    async def initialize(self):
        """Foundation Model 초기화 - PHI-3.5 내장"""
        print("🧠 PHI-3.5 Foundation Model 초기화 중...")
        
        try:
            if self.model_type == "phi35":
                from .phi35_integration import create_phi35_physical_ai
                
                # PHI-3.5 설정
                model_name = self.model_config.get("model_name", "microsoft/Phi-3.5-mini-instruct")
                device = self.model_config.get("device", "auto")
                cache_dir = self.model_config.get("cache_dir", None)
                
                print(f"🔧 PHI-3.5 설정: {model_name} on {device}")
                
                # PHI-3.5 생성 및 초기화
                self.phi35_ai = create_phi35_physical_ai(
                    model_name=model_name,
                    device=device,
                    cache_dir=cache_dir
                )
                
                # 초기화 실행
                success = await self.phi35_ai.initialize()
                if success:
                    print("✅ PHI-3.5 초기화 완료")
                    
                    # TaskPlanningModule에 PHI-3.5 연결
                    self.task_planner.phi35_ai = self.phi35_ai
                    
                    # 모델 정보 저장
                    self.performance_metrics["model_info"] = self.phi35_ai.model_manager.get_model_info()
                    
                else:
                    print("❌ PHI-3.5 초기화 실패")
                    raise Exception("PHI-3.5 initialization failed")
                    
            else:
                print(f"❌ 지원하지 않는 모델 타입: {self.model_type}")
                raise Exception(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            print(f"❌ Foundation Model 초기화 실패: {e}")
            print("⚠️  폴백 모드로 계속 진행합니다 (PHI-3.5 없이)")
            self.phi35_ai = None
        
        print("🎯 sLM Foundation Model 초기화 완료")
    
    async def interpret_mission(self, mission: str) -> TaskPlan:
        """미션 해석 및 계획 수립"""
        print(f"미션 해석 중: {mission}")
        
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
        
        print(f"태스크 계획 생성 완료: {len(optimized_tasks)}개 서브태스크")
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
        return [
            "task_completion",
            "no_collisions",
            "within_time_limit",
            "energy_efficient"
        ]
    
    def _estimate_duration(self, tasks: List[Dict[str, Any]]) -> float:
        """실행 시간 추정"""
        base_durations = {
            "move_to": 10.0,
            "grasp": 5.0,
            "place": 3.0,
            "explore": 30.0
        }
        
        total_duration = sum(
            base_durations.get(task.get("action", "unknown"), 15.0)
            for task in tasks
        )
        
        return total_duration

# 테스트 코드
if __name__ == "__main__":
    async def test():
        foundation = SLMFoundation()
        await foundation.initialize()
        
        mission = "Pick up the red cup and place it on the table"
        plan = await foundation.interpret_mission(mission)
        
        print(f"Mission: {plan.mission}")
        print(f"Subtasks: {len(plan.subtasks)}")
        for i, task in enumerate(plan.subtasks):
            print(f"  {i+1}. {task['action']} -> {task['target']}")
        print(f"Expected duration: {plan.expected_duration} seconds")
    
    asyncio.run(test())
