"""
Distributed Mission Orchestrator

Physical AI System의 분산 미션 처리 오케스트레이터
"""

import ray
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

@dataclass
class MissionTask:
    """미션 태스크 정의"""
    task_id: str
    mission_id: str
    task_type: str  # foundation, learning, execution, hardware
    priority: int
    dependencies: List[str]
    parameters: Dict[str, Any]
    created_at: datetime
    status: str = "pending"  # pending, running, completed, failed

@dataclass
class MissionResult:
    """미션 결과"""
    mission_id: str
    success: bool
    execution_time: float
    results: Dict[str, Any]
    errors: List[str]
    worker_info: Dict[str, Any]

@ray.remote
class MissionScheduler:
    """미션 스케줄러"""
    
    def __init__(self):
        self.pending_tasks = []
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.worker_pool = {}
        
    def add_task(self, task: MissionTask):
        """태스크 추가"""
        self.pending_tasks.append(task)
        logger.info(f"태스크 추가: {task.task_id} ({task.task_type})")
        
    def get_next_task(self, worker_id: str, worker_type: str) -> Optional[MissionTask]:
        """다음 실행할 태스크 반환"""
        for task in self.pending_tasks:
            if (task.task_type == worker_type and 
                task.status == "pending" and
                self._check_dependencies(task)):
                
                task.status = "running"
                self.running_tasks[task.task_id] = {
                    "worker_id": worker_id,
                    "start_time": datetime.now()
                }
                self.pending_tasks.remove(task)
                return task
        
        return None
    
    def _check_dependencies(self, task: MissionTask) -> bool:
        """의존성 확인"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """태스크 완료 처리"""
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            self.completed_tasks[task_id] = {
                "result": result,
                "worker_id": task_info["worker_id"],
                "start_time": task_info["start_time"],
                "end_time": datetime.now()
            }
            logger.info(f"태스크 완료: {task_id}")
    
    def fail_task(self, task_id: str, error: str):
        """태스크 실패 처리"""
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            self.failed_tasks[task_id] = {
                "error": error,
                "worker_id": task_info["worker_id"],
                "start_time": task_info["start_time"],
                "end_time": datetime.now()
            }
            logger.error(f"태스크 실패: {task_id} - {error}")

@ray.remote
class TaskDistributor:
    """태스크 분배기"""
    
    def __init__(self):
        self.worker_registry = {}
        self.load_balancer = LoadBalancer()
        
    def register_worker(self, worker_id: str, worker_type: str, capabilities: Dict[str, Any]):
        """워커 등록"""
        self.worker_registry[worker_id] = {
            "type": worker_type,
            "capabilities": capabilities,
            "status": "available",
            "current_load": 0,
            "registered_at": datetime.now()
        }
        logger.info(f"워커 등록: {worker_id} ({worker_type})")
        
    def get_optimal_worker(self, task_type: str, requirements: Dict[str, Any]) -> Optional[str]:
        """최적 워커 선택"""
        available_workers = [
            wid for wid, info in self.worker_registry.items()
            if (info["type"] == task_type and 
                info["status"] == "available" and
                self._check_capabilities(info["capabilities"], requirements))
        ]
        
        if not available_workers:
            return None
            
        # 로드 밸런싱 적용
        return self.load_balancer.select_worker(available_workers, self.worker_registry)
    
    def _check_capabilities(self, capabilities: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
        """능력 확인"""
        for req_key, req_value in requirements.items():
            if req_key not in capabilities:
                return False
            if capabilities[req_key] < req_value:
                return False
        return True
    
    def update_worker_status(self, worker_id: str, status: str, load: int = None):
        """워커 상태 업데이트"""
        if worker_id in self.worker_registry:
            self.worker_registry[worker_id]["status"] = status
            if load is not None:
                self.worker_registry[worker_id]["current_load"] = load

class LoadBalancer:
    """로드 밸런서"""
    
    def select_worker(self, available_workers: List[str], worker_registry: Dict[str, Any]) -> str:
        """워커 선택 (라운드 로빈 + 로드 기반)"""
        if not available_workers:
            return None
            
        # 가장 낮은 로드를 가진 워커 선택
        min_load = float('inf')
        selected_worker = available_workers[0]
        
        for worker_id in available_workers:
            load = worker_registry[worker_id]["current_load"]
            if load < min_load:
                min_load = load
                selected_worker = worker_id
                
        return selected_worker

@ray.remote
class ResourceManager:
    """리소스 관리자"""
    
    def __init__(self):
        self.resource_usage = {}
        self.resource_limits = {}
        self.monitoring_data = []
        
    def allocate_resources(self, worker_id: str, resources: Dict[str, Any]) -> bool:
        """리소스 할당"""
        # 리소스 가용성 확인
        if not self._check_resource_availability(resources):
            return False
            
        # 리소스 할당
        self.resource_usage[worker_id] = resources
        self._update_resource_limits(resources, allocate=True)
        
        logger.info(f"리소스 할당: {worker_id} - {resources}")
        return True
    
    def release_resources(self, worker_id: str):
        """리소스 해제"""
        if worker_id in self.resource_usage:
            resources = self.resource_usage.pop(worker_id)
            self._update_resource_limits(resources, allocate=False)
            logger.info(f"리소스 해제: {worker_id}")
    
    def _check_resource_availability(self, resources: Dict[str, Any]) -> bool:
        """리소스 가용성 확인"""
        for resource_type, amount in resources.items():
            if resource_type in self.resource_limits:
                if self.resource_limits[resource_type] < amount:
                    return False
        return True
    
    def _update_resource_limits(self, resources: Dict[str, Any], allocate: bool):
        """리소스 한계 업데이트"""
        for resource_type, amount in resources.items():
            if resource_type not in self.resource_limits:
                self.resource_limits[resource_type] = 0
                
            if allocate:
                self.resource_limits[resource_type] -= amount
            else:
                self.resource_limits[resource_type] += amount

class DistributedMissionOrchestrator:
    """분산 미션 오케스트레이터"""
    
    def __init__(self):
        self.scheduler = MissionScheduler.remote()
        self.distributor = TaskDistributor.remote()
        self.resource_manager = ResourceManager.remote()
        self.active_missions = {}
        
    async def submit_mission(self, mission: str, mission_type: str = "complex") -> str:
        """미션 제출"""
        mission_id = str(uuid.uuid4())
        
        # 미션을 태스크로 분해
        tasks = await self._decompose_mission(mission, mission_id, mission_type)
        
        # 태스크를 스케줄러에 추가
        for task in tasks:
            await self.scheduler.add_task.remote(task)
        
        self.active_missions[mission_id] = {
            "mission": mission,
            "tasks": [task.task_id for task in tasks],
            "status": "submitted",
            "submitted_at": datetime.now()
        }
        
        logger.info(f"미션 제출: {mission_id} - {len(tasks)}개 태스크")
        return mission_id
    
    async def _decompose_mission(self, mission: str, mission_id: str, mission_type: str) -> List[MissionTask]:
        """미션을 태스크로 분해"""
        tasks = []
        
        if mission_type == "complex":
            # 복합 미션: Foundation → Learning → Execution → Hardware
            foundation_task = MissionTask(
                task_id=f"{mission_id}_foundation",
                mission_id=mission_id,
                task_type="foundation",
                priority=1,
                dependencies=[],
                parameters={"mission": mission},
                created_at=datetime.now()
            )
            tasks.append(foundation_task)
            
            learning_task = MissionTask(
                task_id=f"{mission_id}_learning",
                mission_id=mission_id,
                task_type="learning",
                priority=2,
                dependencies=[foundation_task.task_id],
                parameters={},
                created_at=datetime.now()
            )
            tasks.append(learning_task)
            
            execution_task = MissionTask(
                task_id=f"{mission_id}_execution",
                mission_id=mission_id,
                task_type="execution",
                priority=3,
                dependencies=[learning_task.task_id],
                parameters={},
                created_at=datetime.now()
            )
            tasks.append(execution_task)
            
            hardware_task = MissionTask(
                task_id=f"{mission_id}_hardware",
                mission_id=mission_id,
                task_type="hardware",
                priority=4,
                dependencies=[execution_task.task_id],
                parameters={},
                created_at=datetime.now()
            )
            tasks.append(hardware_task)
        
        return tasks
    
    async def get_mission_status(self, mission_id: str) -> Dict[str, Any]:
        """미션 상태 조회"""
        if mission_id not in self.active_missions:
            return {"status": "not_found"}
        
        mission_info = self.active_missions[mission_id]
        task_ids = mission_info["tasks"]
        
        # 각 태스크의 상태 확인
        task_statuses = {}
        for task_id in task_ids:
            # Ray remote call로 상태 확인
            status = await self.scheduler._get_task_status.remote(task_id)
            task_statuses[task_id] = status
        
        # 전체 미션 상태 결정
        all_completed = all(status == "completed" for status in task_statuses.values())
        any_failed = any(status == "failed" for status in task_statuses.values())
        
        if any_failed:
            mission_status = "failed"
        elif all_completed:
            mission_status = "completed"
        else:
            mission_status = "running"
        
        return {
            "mission_id": mission_id,
            "status": mission_status,
            "task_statuses": task_statuses,
            "submitted_at": mission_info["submitted_at"].isoformat()
        }
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """클러스터 통계"""
        # Ray 클러스터 정보
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        
        # 워커 정보
        worker_registry = await self.distributor.get_worker_registry.remote()
        
        return {
            "cluster_resources": cluster_resources,
            "available_resources": available_resources,
            "active_workers": len(worker_registry),
            "active_missions": len(self.active_missions),
            "worker_types": {
                worker_type: len([w for w in workers if w["type"] == worker_type])
                for worker_type in set(w["type"] for w in worker_registry.values())
            }
        }

# 테스트 함수
async def test_orchestrator():
    """오케스트레이터 테스트"""
    orchestrator = DistributedMissionOrchestrator()
    
    # 테스트 미션 제출
    mission_id = await orchestrator.submit_mission(
        "Pick up the red cup and place it on the table",
        mission_type="complex"
    )
    
    print(f"제출된 미션 ID: {mission_id}")
    
    # 미션 상태 확인
    status = await orchestrator.get_mission_status(mission_id)
    print(f"미션 상태: {status}")
    
    # 클러스터 통계
    stats = await orchestrator.get_cluster_stats()
    print(f"클러스터 통계: {stats}")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
