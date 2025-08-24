"""
Distributed Foundation Model Workers

PHI-3.5 모델의 분산 추론을 위한 Ray Workers
"""

import ray
import torch
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import uuid

# 기존 모듈 import
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from foundation_model.phi35_integration import PHI35ModelManager

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """추론 요청"""
    request_id: str
    mission: str
    context: Dict[str, Any]
    priority: int = 1
    batch_size: int = 1
    timeout: float = 30.0

@dataclass
class InferenceResult:
    """추론 결과"""
    request_id: str
    success: bool
    result: Dict[str, Any]
    processing_time: float
    model_info: Dict[str, Any]
    errors: List[str] = None

@ray.remote(num_gpus=0.5)  # GPU 리소스 할당
class FoundationModelWorker:
    """Foundation Model 분산 워커"""
    
    def __init__(self, worker_id: str, model_config: Dict[str, Any]):
        self.worker_id = worker_id
        self.model_config = model_config
        self.model_manager = None
        self.is_initialized = False
        self.current_load = 0
        self.max_batch_size = model_config.get("max_batch_size", 8)
        self.request_queue = []
        self.processing_requests = {}
        
        logger.info(f"Foundation Model Worker 초기화: {worker_id}")
    
    async def initialize(self) -> bool:
        """워커 초기화"""
        try:
            logger.info(f"Foundation Model Worker {self.worker_id} 초기화 시작...")
            
            # PHI-3.5 모델 매니저 초기화
            self.model_manager = PHI35ModelManager(
                model_name=self.model_config.get("model_name", "microsoft/Phi-3.5-mini-instruct"),
                device=self.model_config.get("device", "auto"),
                max_length=self.model_config.get("max_length", 2048)
            )
            
            await self.model_manager.initialize()
            self.is_initialized = True
            
            logger.info(f"Foundation Model Worker {self.worker_id} 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"Foundation Model Worker {self.worker_id} 초기화 실패: {e}")
            return False
    
    async def process_inference_request(self, request: InferenceRequest) -> InferenceResult:
        """추론 요청 처리"""
        if not self.is_initialized:
            return InferenceResult(
                request_id=request.request_id,
                success=False,
                result={},
                processing_time=0.0,
                model_info={},
                errors=["Worker not initialized"]
            )
        
        start_time = datetime.now()
        
        try:
            # 로드 증가
            self.current_load += 1
            self.processing_requests[request.request_id] = request
            
            logger.info(f"추론 요청 처리 시작: {request.request_id}")
            
            # PHI-3.5 모델로 미션 처리
            result = await self.model_manager.process_mission_with_learning(
                mission=request.mission,
                context=request.context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 로드 감소
            self.current_load -= 1
            self.processing_requests.pop(request.request_id, None)
            
            return InferenceResult(
                request_id=request.request_id,
                success=result.get("success", False),
                result=result,
                processing_time=processing_time,
                model_info={
                    "worker_id": self.worker_id,
                    "model_name": self.model_manager.model_name,
                    "device": self.model_manager.device
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.current_load -= 1
            self.processing_requests.pop(request.request_id, None)
            
            logger.error(f"추론 요청 처리 실패: {request.request_id} - {e}")
            
            return InferenceResult(
                request_id=request.request_id,
                success=False,
                result={},
                processing_time=processing_time,
                model_info={"worker_id": self.worker_id},
                errors=[str(e)]
            )
    
    async def process_batch_requests(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """배치 추론 요청 처리"""
        if not self.is_initialized:
            return [
                InferenceResult(
                    request_id=req.request_id,
                    success=False,
                    result={},
                    processing_time=0.0,
                    model_info={},
                    errors=["Worker not initialized"]
                )
                for req in requests
            ]
        
        start_time = datetime.now()
        results = []
        
        try:
            # 배치 크기 제한
            batch_requests = requests[:self.max_batch_size]
            self.current_load += len(batch_requests)
            
            logger.info(f"배치 추론 처리: {len(batch_requests)}개 요청")
            
            # 병렬 처리
            tasks = [
                self.process_inference_request(req) 
                for req in batch_requests
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append(InferenceResult(
                        request_id=batch_requests[i].request_id,
                        success=False,
                        result={},
                        processing_time=0.0,
                        model_info={"worker_id": self.worker_id},
                        errors=[str(result)]
                    ))
                else:
                    results.append(result)
            
            self.current_load -= len(batch_requests)
            
        except Exception as e:
            logger.error(f"배치 추론 처리 실패: {e}")
            self.current_load -= len(requests)
            
            # 에러 결과 생성
            for req in requests:
                results.append(InferenceResult(
                    request_id=req.request_id,
                    success=False,
                    result={},
                    processing_time=0.0,
                    model_info={"worker_id": self.worker_id},
                    errors=[str(e)]
                ))
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """워커 상태 반환"""
        return {
            "worker_id": self.worker_id,
            "initialized": self.is_initialized,
            "current_load": self.current_load,
            "max_batch_size": self.max_batch_size,
            "processing_requests": len(self.processing_requests),
            "model_info": {
                "model_name": self.model_manager.model_name if self.model_manager else None,
                "device": self.model_manager.device if self.model_manager else None
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """워커 능력 반환"""
        return {
            "worker_type": "foundation_model",
            "max_batch_size": self.max_batch_size,
            "supported_models": ["phi35"],
            "gpu_required": True,
            "memory_usage": "high"
        }

@ray.remote
class FoundationModelOrchestrator:
    """Foundation Model 오케스트레이터"""
    
    def __init__(self, num_workers: int = 2, model_config: Dict[str, Any] = None):
        self.num_workers = num_workers
        self.model_config = model_config or {}
        self.workers = []
        self.request_queue = []
        self.completed_requests = {}
        
        logger.info(f"Foundation Model Orchestrator 초기화: {num_workers}개 워커")
    
    async def initialize_workers(self) -> bool:
        """워커들 초기화"""
        try:
            logger.info("Foundation Model Workers 초기화 시작...")
            
            # 워커 생성 및 초기화
            for i in range(self.num_workers):
                worker_id = f"foundation_worker_{i}"
                worker = FoundationModelWorker.remote(worker_id, self.model_config)
                
                # 워커 초기화
                success = await worker.initialize.remote()
                if success:
                    self.workers.append(worker)
                    logger.info(f"Foundation Model Worker {worker_id} 초기화 완료")
                else:
                    logger.error(f"Foundation Model Worker {worker_id} 초기화 실패")
            
            logger.info(f"Foundation Model Workers 초기화 완료: {len(self.workers)}개")
            return len(self.workers) > 0
            
        except Exception as e:
            logger.error(f"Foundation Model Workers 초기화 실패: {e}")
            return False
    
    async def submit_inference_request(self, mission: str, context: Dict[str, Any] = None) -> str:
        """추론 요청 제출"""
        request_id = str(uuid.uuid4())
        
        request = InferenceRequest(
            request_id=request_id,
            mission=mission,
            context=context or {},
            priority=1
        )
        
        self.request_queue.append(request)
        logger.info(f"추론 요청 제출: {request_id}")
        
        return request_id
    
    async def process_requests(self) -> Dict[str, InferenceResult]:
        """요청 처리"""
        if not self.workers:
            logger.warning("처리할 워커가 없습니다")
            return {}
        
        results = {}
        
        try:
            # 사용 가능한 워커 찾기
            available_workers = []
            for worker in self.workers:
                status = await worker.get_status.remote()
                if status["current_load"] < status["max_batch_size"]:
                    available_workers.append(worker)
            
            if not available_workers:
                logger.warning("사용 가능한 워커가 없습니다")
                return results
            
            # 요청을 워커에 분배
            while self.request_queue and available_workers:
                # 가장 적은 로드를 가진 워커 선택
                selected_worker = min(available_workers, 
                                   key=lambda w: ray.get(w.get_status.remote())["current_load"])
                
                # 배치 크기만큼 요청 가져오기
                batch_requests = []
                worker_status = await selected_worker.get_status.remote()
                max_batch = worker_status["max_batch_size"] - worker_status["current_load"]
                
                for _ in range(min(max_batch, len(self.request_queue))):
                    if self.request_queue:
                        batch_requests.append(self.request_queue.pop(0))
                
                if batch_requests:
                    # 배치 처리
                    batch_results = await selected_worker.process_batch_requests.remote(batch_requests)
                    
                    # 결과 저장
                    for result in batch_results:
                        results[result.request_id] = result
                        self.completed_requests[result.request_id] = result
                
                # 워커 상태 업데이트
                worker_status = await selected_worker.get_status.remote()
                if worker_status["current_load"] >= worker_status["max_batch_size"]:
                    available_workers.remove(selected_worker)
            
        except Exception as e:
            logger.error(f"요청 처리 실패: {e}")
        
        return results
    
    async def get_request_result(self, request_id: str) -> Optional[InferenceResult]:
        """요청 결과 조회"""
        return self.completed_requests.get(request_id)
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """오케스트레이터 상태"""
        worker_statuses = []
        for worker in self.workers:
            status = await worker.get_status.remote()
            worker_statuses.append(status)
        
        return {
            "num_workers": len(self.workers),
            "worker_statuses": worker_statuses,
            "queue_length": len(self.request_queue),
            "completed_requests": len(self.completed_requests),
            "total_load": sum(status["current_load"] for status in worker_statuses)
        }

# 테스트 함수
async def test_foundation_workers():
    """Foundation Model Workers 테스트"""
    # 오케스트레이터 생성
    orchestrator = FoundationModelOrchestrator(
        num_workers=2,
        model_config={
            "model_name": "microsoft/Phi-3.5-mini-instruct",
            "device": "auto",
            "max_batch_size": 4
        }
    )
    
    # 워커 초기화
    success = await orchestrator.initialize_workers()
    if not success:
        print("워커 초기화 실패")
        return
    
    # 테스트 요청 제출
    request_ids = []
    test_missions = [
        "Pick up the red cup",
        "Place the object on the table",
        "Move to position [1, 0, 0.5]",
        "Explore the environment"
    ]
    
    for mission in test_missions:
        request_id = await orchestrator.submit_inference_request(mission)
        request_ids.append(request_id)
    
    print(f"제출된 요청: {len(request_ids)}개")
    
    # 요청 처리
    results = await orchestrator.process_requests()
    print(f"처리된 요청: {len(results)}개")
    
    # 결과 확인
    for request_id in request_ids:
        result = await orchestrator.get_request_result(request_id)
        if result:
            print(f"요청 {request_id}: {'성공' if result.success else '실패'}")
    
    # 상태 확인
    status = await orchestrator.get_orchestrator_status()
    print(f"오케스트레이터 상태: {status}")

if __name__ == "__main__":
    asyncio.run(test_foundation_workers())
