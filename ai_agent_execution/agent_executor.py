"""
AI Agent Executor - 실시간 물리적 실행 레이어

Foundation Model의 계획과 Developmental Engine의 스킬을 받아서
실제 물리적 동작을 안전하게 실행합니다.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """실행 결과"""
    success: bool
    execution_time: float
    actions_performed: List[Dict[str, Any]]
    errors: List[str]
    performance_metrics: Dict[str, float]
    learning_value: float

class MotionController:
    """동작 제어 모듈"""
    
    def __init__(self):
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_velocity = np.array([0.0, 0.0, 0.0])
        self.max_velocity = 2.0  # m/s
        self.max_acceleration = 5.0  # m/s²
        self.safety_margin = 0.1  # m
        
    async def execute_motion(self, target_position: np.ndarray, 
                           speed_factor: float = 1.0) -> bool:
        """동작 실행"""
        logger.info(f"동작 실행: {self.current_position} -> {target_position}")
        
        # 경로 계획
        path = self._plan_trajectory(target_position)
        
        # 경로를 따라 이동 (시뮬레이션)
        for waypoint in path:
            await self._move_to_waypoint(waypoint, speed_factor)
            await asyncio.sleep(0.1)  # 물리적 시간 지연
            
        logger.info(f"동작 완료: 현재 위치 {self.current_position}")
        return True
    
    def _plan_trajectory(self, target: np.ndarray, num_waypoints: int = 10) -> List[np.ndarray]:
        """궤적 계획"""
        # 단순한 직선 보간
        waypoints = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            waypoint = self.current_position + t * (target - self.current_position)
            waypoints.append(waypoint)
        return waypoints
    
    async def _move_to_waypoint(self, waypoint: np.ndarray, speed_factor: float):
        """웨이포인트로 이동"""
        # 속도 제한 적용
        direction = waypoint - self.current_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            velocity = direction / distance * min(self.max_velocity * speed_factor, distance * 10)
            self.current_velocity = velocity
            self.current_position = waypoint

class SafetyMonitor:
    """실시간 안전 모니터링"""
    
    def __init__(self):
        self.emergency_stop = False
        self.collision_zones = []
        self.human_nearby = False
        self.safety_violations = []
        
    async def monitor_safety(self) -> Dict[str, Any]:
        """안전 상태 모니터링"""
        safety_status = {
            "safe": True,
            "warnings": [],
            "violations": []
        }
        
        # 충돌 감지 (시뮬레이션)
        if await self._check_collision_risk():
            safety_status["safe"] = False
            safety_status["violations"].append("collision_risk")
            
        # 인간 근접 감지
        if await self._check_human_proximity():
            safety_status["warnings"].append("human_nearby")
            
        # 에너지 한계 확인
        if await self._check_energy_limits():
            safety_status["warnings"].append("energy_limit_approaching")
            
        return safety_status
    
    async def _check_collision_risk(self) -> bool:
        """충돌 위험 확인"""
        # 실제로는 센서 데이터를 분석
        # 여기서는 랜덤한 시뮬레이션
        import random
        return random.random() < 0.05  # 5% 확률로 충돌 위험
    
    async def _check_human_proximity(self) -> bool:
        """인간 근접 확인"""
        import random
        return random.random() < 0.1  # 10% 확률로 인간 근접
    
    async def _check_energy_limits(self) -> bool:
        """에너지 한계 확인"""
        import random
        return random.random() < 0.15  # 15% 확률로 에너지 경고
    
    def trigger_emergency_stop(self):
        """비상 정지 트리거"""
        self.emergency_stop = True
        logger.warning("⚠️ 비상 정지 활성화!")

class AgentExecutor:
    """AI Agent 실행기 메인 클래스"""
    
    def __init__(self):
        self.motion_controller = MotionController()
        self.safety_monitor = SafetyMonitor()
        self.hw_manager = None
        self.execution_active = False
        
    async def initialize(self, hardware_manager):
        """Agent Executor 초기화"""
        logger.info("AI Agent Executor 초기화 중...")
        self.hw_manager = hardware_manager
        
        # 안전 모니터링 백그라운드 태스크 시작
        asyncio.create_task(self._safety_monitoring_loop())
        
        logger.info("AI Agent Executor 초기화 완료")
    
    async def execute(self, task_plan, skill_states: Dict[str, Any]) -> ExecutionResult:
        """태스크 계획 실행"""
        logger.info(f"태스크 실행 시작: {task_plan.mission}")
        
        start_time = time.time()
        actions_performed = []
        errors = []
        
        self.execution_active = True
        
        try:
            # 각 서브태스크 순차 실행
            for i, subtask in enumerate(task_plan.subtasks):
                logger.info(f"서브태스크 {i+1}/{len(task_plan.subtasks)}: {subtask['action']}")
                
                # 안전 확인
                safety_status = await self.safety_monitor.monitor_safety()
                if not safety_status["safe"]:
                    errors.append(f"안전 위반: {safety_status['violations']}")
                    break
                
                # 서브태스크 실행
                success = await self._execute_subtask(subtask, skill_states)
                
                actions_performed.append({
                    "subtask": subtask,
                    "success": success,
                    "timestamp": time.time()
                })
                
                if not success:
                    errors.append(f"서브태스크 실패: {subtask['action']}")
                    break
                
                # 태스크 간 휴식
                await asyncio.sleep(0.5)
                
        except Exception as e:
            errors.append(f"실행 중 예외 발생: {str(e)}")
            
        finally:
            self.execution_active = False
        
        execution_time = time.time() - start_time
        success = len(errors) == 0
        
        # 성능 지표 계산
        performance_metrics = self._calculate_performance_metrics(
            actions_performed, execution_time
        )
        
        # 학습 가치 계산
        learning_value = self._calculate_learning_value(
            success, actions_performed, errors
        )
        
        result = ExecutionResult(
            success=success,
            execution_time=execution_time,
            actions_performed=actions_performed,
            errors=errors,
            performance_metrics=performance_metrics,
            learning_value=learning_value
        )
        
        logger.info(f"태스크 실행 완료: {'성공' if success else '실패'} "
              f"({execution_time:.2f}초)")
        
        return result
    
    async def _execute_subtask(self, subtask: Dict[str, Any], 
                             skill_states: Dict[str, Any]) -> bool:
        """개별 서브태스크 실행"""
        action = subtask.get("action", "")
        target = subtask.get("target", "")
        
        try:
            if action == "move_to":
                # 이동 동작
                target_pos = self._resolve_target_position(target)
                return await self.motion_controller.execute_motion(target_pos)
                
            elif action == "grasp":
                # 잡기 동작
                logger.info(f"객체 '{target}' 잡기 시도")
                # 하드웨어 그리퍼 제어
                if self.hw_manager:
                    return await self.hw_manager.control_gripper("close", target)
                return True  # 시뮬레이션에서는 항상 성공
                
            elif action == "place":
                # 놓기 동작
                logger.info(f"객체를 '{target}'에 놓기")
                if self.hw_manager:
                    return await self.hw_manager.control_gripper("open", target)
                return True
                
            elif action == "explore" or action == "explore_environment":
                # 탐색 동작
                logger.info(f"'{target}' 탐색 중")
                # 랜덤 위치들을 방문하며 탐색
                for _ in range(3):
                    random_pos = np.random.uniform(-1, 1, 3)
                    await self.motion_controller.execute_motion(random_pos)
                return True
                
            else:
                logger.warning(f"알 수 없는 동작: {action}")
                return False
                
        except Exception as e:
            logger.error(f"서브태스크 실행 오류: {e}")
            return False
    
    def _resolve_target_position(self, target: str) -> np.ndarray:
        """타겟 이름을 3D 위치로 변환"""
        # 실제로는 환경 맵이나 객체 감지를 통해 위치 결정
        position_map = {
            "object_location": np.array([1.0, 0.0, 0.5]),
            "destination": np.array([2.0, 1.0, 0.5]),
            "table": np.array([1.5, 0.5, 0.8]),
            "unknown_area": np.random.uniform(-2, 2, 3)
        }
        
        return position_map.get(target, np.array([0.0, 0.0, 0.0]))
    
    def _calculate_performance_metrics(self, actions: List[Dict[str, Any]], 
                                     execution_time: float) -> Dict[str, float]:
        """성능 지표 계산"""
        total_actions = len(actions)
        successful_actions = sum(1 for action in actions if action["success"])
        
        return {
            "success_rate": successful_actions / max(total_actions, 1),
            "execution_efficiency": total_actions / max(execution_time, 0.1),
            "average_action_time": execution_time / max(total_actions, 1),
            "error_rate": (total_actions - successful_actions) / max(total_actions, 1)
        }
    
    def _calculate_learning_value(self, success: bool, 
                                actions: List[Dict[str, Any]], 
                                errors: List[str]) -> float:
        """학습 가치 계산"""
        base_value = 0.5
        
        # 성공/실패에 따른 가중치
        success_weight = 0.8 if success else 0.3
        
        # 복잡성에 따른 가중치
        complexity_weight = min(1.0, len(actions) / 5.0)
        
        # 오류의 유익성 (일부 오류는 학습에 도움)
        error_weight = min(0.3, len(errors) * 0.1)
        
        learning_value = base_value * success_weight * complexity_weight + error_weight
        return min(1.0, learning_value)
    
    async def _safety_monitoring_loop(self):
        """백그라운드 안전 모니터링 루프"""
        while True:
            if self.execution_active:
                safety_status = await self.safety_monitor.monitor_safety()
                
                if not safety_status["safe"]:
                    logger.warning(f"⚠️ 안전 경고: {safety_status['violations']}")
                    self.safety_monitor.trigger_emergency_stop()
                    self.execution_active = False
                    
            await asyncio.sleep(0.1)  # 10Hz 모니터링

# 테스트 코드
if __name__ == "__main__":
    async def test():
        from foundation_model.slm_foundation import SLMFoundation, TaskPlan
        
        executor = AgentExecutor()
        await executor.initialize(None)  # 하드웨어 없이 테스트
        
        # 가짜 태스크 계획
        fake_plan = TaskPlan(
            mission="Test mission",
            subtasks=[
                {"action": "move_to", "target": "object_location"},
                {"action": "grasp", "target": "test_object"},
                {"action": "move_to", "target": "destination"},
                {"action": "place", "target": "table"}
            ],
            constraints={},
            expected_duration=30.0,
            success_criteria=["completion"]
        )
        
        skill_states = {
            "basic_movement": {"success_rate": 0.8},
            "simple_grasp": {"success_rate": 0.7}
        }
        
        result = await executor.execute(fake_plan, skill_states)
        
        logger.info(f"\n=== 실행 결과 ===")
        logger.info(f"성공: {result.success}")
        logger.info(f"실행 시간: {result.execution_time:.2f}초")
        logger.info(f"수행 동작: {len(result.actions_performed)}개")
        logger.info(f"오류: {len(result.errors)}개")
        logger.info(f"학습 가치: {result.learning_value:.2f}")
    
    asyncio.run(test())
