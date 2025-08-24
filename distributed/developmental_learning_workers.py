"""
Distributed Developmental Learning Workers

발달적 학습의 분산 처리를 위한 Ray Workers
"""

import ray
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import uuid
import json

# 기존 모듈 import
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from developmental_learning.dev_engine import DevelopmentalEngine, Skill, Experience

logger = logging.getLogger(__name__)

@dataclass
class LearningTask:
    """학습 태스크"""
    task_id: str
    skill_name: str
    environment_config: Dict[str, Any]
    learning_params: Dict[str, Any]
    priority: int = 1
    max_iterations: int = 1000
    timeout: float = 300.0

@dataclass
class LearningResult:
    """학습 결과"""
    task_id: str
    skill_name: str
    success: bool
    final_performance: float
    iterations_completed: int
    learning_curve: List[float]
    experience_data: List[Dict[str, Any]]
    processing_time: float
    worker_info: Dict[str, Any]
    errors: List[str] = None

@ray.remote(num_cpus=2)  # CPU 리소스 할당
class DevelopmentalLearningWorker:
    """발달적 학습 분산 워커"""
    
    def __init__(self, worker_id: str, learning_config: Dict[str, Any]):
        self.worker_id = worker_id
        self.learning_config = learning_config
        self.dev_engine = None
        self.is_initialized = False
        self.current_load = 0
        self.max_environments = learning_config.get("max_environments", 2)
        self.active_environments = {}
        self.learning_history = []
        
        logger.info(f"Developmental Learning Worker 초기화: {worker_id}")
    
    async def initialize(self) -> bool:
        """워커 초기화"""
        try:
            logger.info(f"Developmental Learning Worker {self.worker_id} 초기화 시작...")
            
            # 발달적 학습 엔진 초기화
            self.dev_engine = DevelopmentalEngine()
            await self.dev_engine.initialize()
            
            self.is_initialized = True
            logger.info(f"Developmental Learning Worker {self.worker_id} 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"Developmental Learning Worker {self.worker_id} 초기화 실패: {e}")
            return False
    
    async def process_learning_task(self, task: LearningTask) -> LearningResult:
        """학습 태스크 처리"""
        if not self.is_initialized:
            return LearningResult(
                task_id=task.task_id,
                skill_name=task.skill_name,
                success=False,
                final_performance=0.0,
                iterations_completed=0,
                learning_curve=[],
                experience_data=[],
                processing_time=0.0,
                worker_info={"worker_id": self.worker_id},
                errors=["Worker not initialized"]
            )
        
        start_time = datetime.now()
        learning_curve = []
        experience_data = []
        
        try:
            # 로드 증가
            self.current_load += 1
            self.active_environments[task.task_id] = task
            
            logger.info(f"학습 태스크 시작: {task.task_id} - {task.skill_name}")
            
            # 환경 설정
            environment_id = f"{self.worker_id}_{task.task_id}"
            
            # 스킬 학습 실행
            success = False
            final_performance = 0.0
            iterations_completed = 0
            
            for iteration in range(task.max_iterations):
                # 학습 단계 실행
                learning_step = await self.dev_engine.learning_step(
                    skill_name=task.skill_name,
                    environment_config=task.environment_config,
                    learning_params=task.learning_params
                )
                
                if learning_step:
                    # 성능 기록
                    performance = learning_step.get("performance", 0.0)
                    learning_curve.append(performance)
                    
                    # 경험 데이터 수집
                    experience = learning_step.get("experience", {})
                    experience_data.append(experience)
                    
                    iterations_completed = iteration + 1
                    
                    # 성공 조건 확인
                    if performance >= task.learning_params.get("success_threshold", 0.8):
                        success = True
                        final_performance = performance
                        break
                    
                    # 시간 초과 확인
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    if elapsed_time > task.timeout:
                        logger.warning(f"학습 태스크 시간 초과: {task.task_id}")
                        break
                    
                    # 짧은 대기 (시뮬레이션 환경 고려)
                    await asyncio.sleep(0.01)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 로드 감소
            self.current_load -= 1
            self.active_environments.pop(task.task_id, None)
            
            # 학습 히스토리 기록
            self.learning_history.append({
                "task_id": task.task_id,
                "skill_name": task.skill_name,
                "success": success,
                "final_performance": final_performance,
                "iterations": iterations_completed,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            })
            
            return LearningResult(
                task_id=task.task_id,
                skill_name=task.skill_name,
                success=success,
                final_performance=final_performance,
                iterations_completed=iterations_completed,
                learning_curve=learning_curve,
                experience_data=experience_data,
                processing_time=processing_time,
                worker_info={
                    "worker_id": self.worker_id,
                    "environment_id": environment_id
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.current_load -= 1
            self.active_environments.pop(task.task_id, None)
            
            logger.error(f"학습 태스크 처리 실패: {task.task_id} - {e}")
            
            return LearningResult(
                task_id=task.task_id,
                skill_name=task.skill_name,
                success=False,
                final_performance=0.0,
                iterations_completed=0,
                learning_curve=[],
                experience_data=[],
                processing_time=processing_time,
                worker_info={"worker_id": self.worker_id},
                errors=[str(e)]
            )
    
    async def process_parallel_learning(self, tasks: List[LearningTask]) -> List[LearningResult]:
        """병렬 학습 처리"""
        if not self.is_initialized:
            return [
                LearningResult(
                    task_id=task.task_id,
                    skill_name=task.skill_name,
                    success=False,
                    final_performance=0.0,
                    iterations_completed=0,
                    learning_curve=[],
                    experience_data=[],
                    processing_time=0.0,
                    worker_info={"worker_id": self.worker_id},
                    errors=["Worker not initialized"]
                )
                for task in tasks
            ]
        
        # 병렬 처리 가능한 태스크 수 제한
        max_parallel = min(self.max_environments, len(tasks))
        parallel_tasks = tasks[:max_parallel]
        
        logger.info(f"병렬 학습 시작: {len(parallel_tasks)}개 태스크")
        
        # 병렬 처리
        learning_tasks = [
            self.process_learning_task(task) 
            for task in parallel_tasks
        ]
        
        results = await asyncio.gather(*learning_tasks, return_exceptions=True)
        
        # 결과 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(LearningResult(
                    task_id=parallel_tasks[i].task_id,
                    skill_name=parallel_tasks[i].skill_name,
                    success=False,
                    final_performance=0.0,
                    iterations_completed=0,
                    learning_curve=[],
                    experience_data=[],
                    processing_time=0.0,
                    worker_info={"worker_id": self.worker_id},
                    errors=[str(result)]
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def share_experience(self, experience_data: List[Dict[str, Any]]) -> bool:
        """경험 공유"""
        try:
            if not self.is_initialized:
                return False
            
            # 중앙 메모리에 경험 데이터 저장
            for experience in experience_data:
                await self.dev_engine.store_experience(experience)
            
            logger.info(f"경험 공유 완료: {len(experience_data)}개")
            return True
            
        except Exception as e:
            logger.error(f"경험 공유 실패: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """워커 상태 반환"""
        return {
            "worker_id": self.worker_id,
            "initialized": self.is_initialized,
            "current_load": self.current_load,
            "max_environments": self.max_environments,
            "active_environments": len(self.active_environments),
            "learning_history_length": len(self.learning_history)
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """워커 능력 반환"""
        return {
            "worker_type": "developmental_learning",
            "max_environments": self.max_environments,
            "supported_skills": ["basic_movement", "object_recognition", "simple_grasp", "precise_manipulation"],
            "cpu_required": True,
            "memory_usage": "medium"
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """학습 통계"""
        if not self.learning_history:
            return {"total_tasks": 0, "success_rate": 0.0, "avg_performance": 0.0}
        
        total_tasks = len(self.learning_history)
        successful_tasks = sum(1 for record in self.learning_history if record["success"])
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        performances = [record["final_performance"] for record in self.learning_history]
        avg_performance = sum(performances) / len(performances) if performances else 0.0
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "avg_performance": avg_performance,
            "avg_processing_time": sum(record["processing_time"] for record in self.learning_history) / total_tasks
        }

@ray.remote
class DevelopmentalLearningOrchestrator:
    """발달적 학습 오케스트레이터"""
    
    def __init__(self, num_workers: int = 4, learning_config: Dict[str, Any] = None):
        self.num_workers = num_workers
        self.learning_config = learning_config or {}
        self.workers = []
        self.task_queue = []
        self.completed_tasks = {}
        self.shared_experiences = []
        
        logger.info(f"Developmental Learning Orchestrator 초기화: {num_workers}개 워커")
    
    async def initialize_workers(self) -> bool:
        """워커들 초기화"""
        try:
            logger.info("Developmental Learning Workers 초기화 시작...")
            
            # 워커 생성 및 초기화
            for i in range(self.num_workers):
                worker_id = f"learning_worker_{i}"
                worker = DevelopmentalLearningWorker.remote(worker_id, self.learning_config)
                
                # 워커 초기화
                success = await worker.initialize.remote()
                if success:
                    self.workers.append(worker)
                    logger.info(f"Developmental Learning Worker {worker_id} 초기화 완료")
                else:
                    logger.error(f"Developmental Learning Worker {worker_id} 초기화 실패")
            
            logger.info(f"Developmental Learning Workers 초기화 완료: {len(self.workers)}개")
            return len(self.workers) > 0
            
        except Exception as e:
            logger.error(f"Developmental Learning Workers 초기화 실패: {e}")
            return False
    
    async def submit_learning_task(self, skill_name: str, environment_config: Dict[str, Any], 
                                 learning_params: Dict[str, Any] = None) -> str:
        """학습 태스크 제출"""
        task_id = str(uuid.uuid4())
        
        task = LearningTask(
            task_id=task_id,
            skill_name=skill_name,
            environment_config=environment_config,
            learning_params=learning_params or {},
            priority=1
        )
        
        self.task_queue.append(task)
        logger.info(f"학습 태스크 제출: {task_id} - {skill_name}")
        
        return task_id
    
    async def process_tasks(self) -> Dict[str, LearningResult]:
        """태스크 처리"""
        if not self.workers:
            logger.warning("처리할 워커가 없습니다")
            return {}
        
        results = {}
        
        try:
            # 사용 가능한 워커 찾기
            available_workers = []
            for worker in self.workers:
                status = await worker.get_status.remote()
                if status["current_load"] < status["max_environments"]:
                    available_workers.append(worker)
            
            if not available_workers:
                logger.warning("사용 가능한 워커가 없습니다")
                return results
            
            # 태스크를 워커에 분배
            while self.task_queue and available_workers:
                # 가장 적은 로드를 가진 워커 선택
                selected_worker = min(available_workers, 
                                   key=lambda w: ray.get(w.get_status.remote())["current_load"])
                
                # 워커당 처리할 태스크 수 결정
                worker_status = await selected_worker.get_status.remote()
                max_tasks = worker_status["max_environments"] - worker_status["current_load"]
                
                batch_tasks = []
                for _ in range(min(max_tasks, len(self.task_queue))):
                    if self.task_queue:
                        batch_tasks.append(self.task_queue.pop(0))
                
                if batch_tasks:
                    # 병렬 학습 처리
                    batch_results = await selected_worker.process_parallel_learning.remote(batch_tasks)
                    
                    # 결과 저장 및 경험 공유
                    for result in batch_results:
                        results[result.task_id] = result
                        self.completed_tasks[result.task_id] = result
                        
                        # 경험 데이터 수집
                        if result.experience_data:
                            self.shared_experiences.extend(result.experience_data)
                
                # 워커 상태 업데이트
                worker_status = await selected_worker.get_status.remote()
                if worker_status["current_load"] >= worker_status["max_environments"]:
                    available_workers.remove(selected_worker)
            
        except Exception as e:
            logger.error(f"태스크 처리 실패: {e}")
        
        return results
    
    async def share_experiences_across_workers(self) -> bool:
        """워커 간 경험 공유"""
        if not self.shared_experiences:
            return True
        
        try:
            logger.info(f"워커 간 경험 공유 시작: {len(self.shared_experiences)}개")
            
            # 모든 워커에 경험 공유
            share_tasks = []
            for worker in self.workers:
                task = worker.share_experience.remote(self.shared_experiences)
                share_tasks.append(task)
            
            # 병렬로 경험 공유
            share_results = await asyncio.gather(*share_tasks, return_exceptions=True)
            
            # 결과 확인
            successful_shares = sum(1 for result in share_results if result is True)
            logger.info(f"경험 공유 완료: {successful_shares}/{len(self.workers)}개 워커")
            
            # 공유된 경험 초기화
            self.shared_experiences = []
            
            return successful_shares > 0
            
        except Exception as e:
            logger.error(f"경험 공유 실패: {e}")
            return False
    
    async def get_task_result(self, task_id: str) -> Optional[LearningResult]:
        """태스크 결과 조회"""
        return self.completed_tasks.get(task_id)
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """오케스트레이터 상태"""
        worker_statuses = []
        worker_stats = []
        
        for worker in self.workers:
            status = await worker.get_status.remote()
            stats = await worker.get_learning_stats.remote()
            worker_statuses.append(status)
            worker_stats.append(stats)
        
        return {
            "num_workers": len(self.workers),
            "worker_statuses": worker_statuses,
            "worker_stats": worker_stats,
            "queue_length": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "shared_experiences": len(self.shared_experiences),
            "total_load": sum(status["current_load"] for status in worker_statuses)
        }

# 테스트 함수
async def test_developmental_learning_workers():
    """Developmental Learning Workers 테스트"""
    # 오케스트레이터 생성
    orchestrator = DevelopmentalLearningOrchestrator(
        num_workers=2,
        learning_config={
            "max_environments": 2,
            "success_threshold": 0.8,
            "max_iterations": 100
        }
    )
    
    # 워커 초기화
    success = await orchestrator.initialize_workers()
    if not success:
        print("워커 초기화 실패")
        return
    
    # 테스트 태스크 제출
    task_ids = []
    test_skills = [
        "basic_movement",
        "object_recognition", 
        "simple_grasp",
        "precise_manipulation"
    ]
    
    for skill in test_skills:
        task_id = await orchestrator.submit_learning_task(
            skill_name=skill,
            environment_config={"simulation": True, "difficulty": "medium"},
            learning_params={"learning_rate": 0.01, "success_threshold": 0.8}
        )
        task_ids.append(task_id)
    
    print(f"제출된 태스크: {len(task_ids)}개")
    
    # 태스크 처리
    results = await orchestrator.process_tasks()
    print(f"처리된 태스크: {len(results)}개")
    
    # 결과 확인
    for task_id in task_ids:
        result = await orchestrator.get_task_result(task_id)
        if result:
            print(f"태스크 {task_id}: {'성공' if result.success else '실패'} (성능: {result.final_performance:.3f})")
    
    # 경험 공유
    await orchestrator.share_experiences_across_workers()
    
    # 상태 확인
    status = await orchestrator.get_orchestrator_status()
    print(f"오케스트레이터 상태: {status}")

if __name__ == "__main__":
    asyncio.run(test_developmental_learning_workers())
