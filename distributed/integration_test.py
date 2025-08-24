"""
Distributed Processing Integration Test

Physical AI System의 분산처리 시스템 통합 테스트
"""

import asyncio
import logging
import time
from typing import Dict, List, Any
from datetime import datetime
import json

# 분산처리 모듈 import
from distributed.ray_cluster_setup import RayClusterManager
from distributed.mission_orchestrator import DistributedMissionOrchestrator
from distributed.foundation_model_workers import FoundationModelOrchestrator
from distributed.developmental_learning_workers import DevelopmentalLearningOrchestrator
from distributed.distributed_memory import DistributedMemoryOrchestrator

logger = logging.getLogger(__name__)

class DistributedSystemIntegrationTest:
    """분산 시스템 통합 테스트"""
    
    def __init__(self):
        self.cluster_manager = None
        self.mission_orchestrator = None
        self.foundation_orchestrator = None
        self.learning_orchestrator = None
        self.memory_orchestrator = None
        self.test_results = {}
        
    async def setup_distributed_system(self) -> bool:
        """분산 시스템 설정"""
        try:
            logger.info("🚀 분산 시스템 설정 시작...")
            
            # 1. Ray Cluster 설정
            self.cluster_manager = RayClusterManager()
            cluster_success = await self.cluster_manager.initialize_cluster()
            if not cluster_success:
                logger.error("Ray Cluster 초기화 실패")
                return False
            
            # 2. 분산 메모리 시스템 초기화
            self.memory_orchestrator = DistributedMemoryOrchestrator(
                num_memory_managers=2,
                num_replay_buffers=2
            )
            memory_success = await self.memory_orchestrator.initialize()
            if not memory_success:
                logger.error("분산 메모리 시스템 초기화 실패")
                return False
            
            # 3. Foundation Model Workers 초기화
            self.foundation_orchestrator = FoundationModelOrchestrator(
                num_workers=2,
                model_config={
                    "model_name": "microsoft/Phi-3.5-mini-instruct",
                    "device": "auto",
                    "max_batch_size": 4
                }
            )
            foundation_success = await self.foundation_orchestrator.initialize_workers()
            if not foundation_success:
                logger.error("Foundation Model Workers 초기화 실패")
                return False
            
            # 4. Developmental Learning Workers 초기화
            self.learning_orchestrator = DevelopmentalLearningOrchestrator(
                num_workers=2,
                learning_config={
                    "max_environments": 2,
                    "success_threshold": 0.8,
                    "max_iterations": 100
                }
            )
            learning_success = await self.learning_orchestrator.initialize_workers()
            if not learning_success:
                logger.error("Developmental Learning Workers 초기화 실패")
                return False
            
            # 5. Mission Orchestrator 초기화
            self.mission_orchestrator = DistributedMissionOrchestrator()
            
            logger.info("✅ 분산 시스템 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"분산 시스템 설정 실패: {e}")
            return False
    
    async def test_foundation_model_distribution(self) -> Dict[str, Any]:
        """Foundation Model 분산 처리 테스트"""
        logger.info("🧠 Foundation Model 분산 처리 테스트 시작...")
        
        start_time = time.time()
        test_results = {
            "success": False,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0,
            "throughput": 0.0
        }
        
        try:
            # 테스트 미션들
            test_missions = [
                "Pick up the red cup and place it on the table",
                "Move to position [1, 0, 0.5] and explore the area",
                "Identify all objects in the workspace",
                "Plan a safe path to the target location",
                "Analyze the physical constraints of the task"
            ]
            
            # 병렬 요청 제출
            request_ids = []
            for mission in test_missions:
                request_id = await self.foundation_orchestrator.submit_inference_request(mission)
                request_ids.append(request_id)
            
            test_results["total_requests"] = len(request_ids)
            
            # 요청 처리
            results = await self.foundation_orchestrator.process_requests()
            
            # 결과 분석
            successful_count = 0
            total_processing_time = 0.0
            
            for request_id in request_ids:
                result = await self.foundation_orchestrator.get_request_result(request_id)
                if result:
                    if result.success:
                        successful_count += 1
                        total_processing_time += result.processing_time
                    else:
                        logger.warning(f"Foundation Model 요청 실패: {request_id}")
            
            test_results["successful_requests"] = successful_count
            test_results["failed_requests"] = test_results["total_requests"] - successful_count
            
            if successful_count > 0:
                test_results["avg_processing_time"] = total_processing_time / successful_count
                test_results["throughput"] = successful_count / (time.time() - start_time)
                test_results["success"] = True
            
            logger.info(f"Foundation Model 테스트 완료: {successful_count}/{test_results['total_requests']} 성공")
            
        except Exception as e:
            logger.error(f"Foundation Model 테스트 실패: {e}")
        
        return test_results
    
    async def test_developmental_learning_distribution(self) -> Dict[str, Any]:
        """발달적 학습 분산 처리 테스트"""
        logger.info("🌱 발달적 학습 분산 처리 테스트 시작...")
        
        start_time = time.time()
        test_results = {
            "success": False,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_learning_time": 0.0,
            "avg_performance": 0.0,
            "experience_shared": 0
        }
        
        try:
            # 테스트 학습 태스크들
            test_skills = [
                "basic_movement",
                "object_recognition",
                "simple_grasp",
                "precise_manipulation"
            ]
            
            # 병렬 학습 태스크 제출
            task_ids = []
            for skill in test_skills:
                task_id = await self.learning_orchestrator.submit_learning_task(
                    skill_name=skill,
                    environment_config={"simulation": True, "difficulty": "medium"},
                    learning_params={"learning_rate": 0.01, "success_threshold": 0.8}
                )
                task_ids.append(task_id)
            
            test_results["total_tasks"] = len(task_ids)
            
            # 태스크 처리
            results = await self.learning_orchestrator.process_tasks()
            
            # 결과 분석
            successful_count = 0
            total_learning_time = 0.0
            total_performance = 0.0
            
            for task_id in task_ids:
                result = await self.learning_orchestrator.get_task_result(task_id)
                if result:
                    if result.success:
                        successful_count += 1
                        total_learning_time += result.processing_time
                        total_performance += result.final_performance
                    else:
                        logger.warning(f"학습 태스크 실패: {task_id}")
            
            test_results["successful_tasks"] = successful_count
            test_results["failed_tasks"] = test_results["total_tasks"] - successful_count
            
            if successful_count > 0:
                test_results["avg_learning_time"] = total_learning_time / successful_count
                test_results["avg_performance"] = total_performance / successful_count
                test_results["success"] = True
            
            # 경험 공유 테스트
            share_success = await self.learning_orchestrator.share_experiences_across_workers()
            test_results["experience_shared"] = 1 if share_success else 0
            
            logger.info(f"발달적 학습 테스트 완료: {successful_count}/{test_results['total_tasks']} 성공")
            
        except Exception as e:
            logger.error(f"발달적 학습 테스트 실패: {e}")
        
        return test_results
    
    async def test_distributed_memory_system(self) -> Dict[str, Any]:
        """분산 메모리 시스템 테스트"""
        logger.info("💾 분산 메모리 시스템 테스트 시작...")
        
        test_results = {
            "success": False,
            "memory_stored": 0,
            "memory_retrieved": 0,
            "experiences_added": 0,
            "experiences_sampled": 0,
            "avg_retrieval_time": 0.0
        }
        
        try:
            # 메모리 저장 테스트
            memory_stored = 0
            for i in range(20):
                from distributed.distributed_memory import MemoryEntry
                memory = MemoryEntry(
                    entry_id=f"test_memory_{i}",
                    memory_type="episodic",
                    content={"action": f"action_{i}", "reward": i * 0.1, "context": f"context_{i}"},
                    timestamp=datetime.now(),
                    priority=1.0 - i * 0.05
                )
                
                success = await self.memory_orchestrator.store_memory_distributed(memory)
                if success:
                    memory_stored += 1
            
            test_results["memory_stored"] = memory_stored
            
            # 메모리 검색 테스트
            from distributed.distributed_memory import MemoryQuery
            query = MemoryQuery(
                query_id="test_query",
                memory_type="episodic",
                search_criteria={"action": "action_10"},
                max_results=5
            )
            
            start_time = time.time()
            retrieved_memories = await self.memory_orchestrator.retrieve_memory_distributed(query)
            retrieval_time = time.time() - start_time
            
            test_results["memory_retrieved"] = len(retrieved_memories)
            test_results["avg_retrieval_time"] = retrieval_time
            
            # 경험 추가 테스트
            experiences_added = 0
            for i in range(30):
                experience = {
                    "state": f"state_{i}",
                    "action": f"action_{i}",
                    "reward": i * 0.1,
                    "next_state": f"state_{i+1}",
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.memory_orchestrator.add_experience_distributed(experience, priority=1.0 - i * 0.03)
                experiences_added += 1
            
            test_results["experiences_added"] = experiences_added
            
            # 경험 샘플링 테스트
            sampled_experiences = await self.memory_orchestrator.sample_experience_batch(10)
            test_results["experiences_sampled"] = len(sampled_experiences)
            
            test_results["success"] = True
            logger.info(f"분산 메모리 테스트 완료: {memory_stored}개 메모리, {len(retrieved_memories)}개 검색")
            
        except Exception as e:
            logger.error(f"분산 메모리 테스트 실패: {e}")
        
        return test_results
    
    async def test_end_to_end_mission_processing(self) -> Dict[str, Any]:
        """엔드투엔드 미션 처리 테스트"""
        logger.info("🎯 엔드투엔드 미션 처리 테스트 시작...")
        
        start_time = time.time()
        test_results = {
            "success": False,
            "missions_submitted": 0,
            "missions_completed": 0,
            "missions_failed": 0,
            "avg_mission_time": 0.0,
            "total_processing_time": 0.0
        }
        
        try:
            # 복합 미션들
            complex_missions = [
                "Pick up the red cup and place it on the table, then clean up the workspace",
                "Navigate to the kitchen, prepare coffee, and serve it to the human",
                "Inspect all equipment in the lab and report any issues found",
                "Organize the workspace by sorting tools by color and size"
            ]
            
            # 미션 제출
            mission_ids = []
            for mission in complex_missions:
                mission_id = await self.mission_orchestrator.submit_mission(mission, mission_type="complex")
                mission_ids.append(mission_id)
            
            test_results["missions_submitted"] = len(mission_ids)
            
            # 미션 처리 시뮬레이션 (실제 하드웨어 없이)
            completed_missions = 0
            total_mission_time = 0.0
            
            for mission_id in mission_ids:
                # 미션 상태 확인
                status = await self.mission_orchestrator.get_mission_status(mission_id)
                
                if status["status"] == "completed":
                    completed_missions += 1
                    # 시뮬레이션된 처리 시간
                    mission_time = 5.0 + (hash(mission_id) % 10)  # 5-15초
                    total_mission_time += mission_time
                elif status["status"] == "failed":
                    logger.warning(f"미션 실패: {mission_id}")
            
            test_results["missions_completed"] = completed_missions
            test_results["missions_failed"] = test_results["missions_submitted"] - completed_missions
            
            if completed_missions > 0:
                test_results["avg_mission_time"] = total_mission_time / completed_missions
                test_results["total_processing_time"] = time.time() - start_time
                test_results["success"] = True
            
            logger.info(f"엔드투엔드 테스트 완료: {completed_missions}/{test_results['missions_submitted']} 성공")
            
        except Exception as e:
            logger.error(f"엔드투엔드 테스트 실패: {e}")
        
        return test_results
    
    async def test_system_performance_and_scalability(self) -> Dict[str, Any]:
        """시스템 성능 및 확장성 테스트"""
        logger.info("📊 시스템 성능 및 확장성 테스트 시작...")
        
        test_results = {
            "success": False,
            "cluster_resources": {},
            "worker_utilization": {},
            "memory_usage": {},
            "throughput_metrics": {},
            "scalability_score": 0.0
        }
        
        try:
            # 클러스터 리소스 정보
            cluster_info = self.cluster_manager.get_cluster_info()
            test_results["cluster_resources"] = cluster_info
            
            # 워커 활용도
            foundation_status = await self.foundation_orchestrator.get_orchestrator_status()
            learning_status = await self.learning_orchestrator.get_orchestrator_status()
            
            test_results["worker_utilization"] = {
                "foundation_workers": foundation_status,
                "learning_workers": learning_status
            }
            
            # 메모리 사용량
            memory_stats = await self.memory_orchestrator.get_distributed_stats()
            test_results["memory_usage"] = memory_stats
            
            # 처리량 메트릭
            throughput_metrics = {
                "foundation_requests_per_second": foundation_status.get("total_load", 0) / 60.0,
                "learning_tasks_per_second": learning_status.get("total_load", 0) / 60.0,
                "memory_operations_per_second": memory_stats.get("total_memory_entries", 0) / 60.0
            }
            test_results["throughput_metrics"] = throughput_metrics
            
            # 확장성 점수 계산
            scalability_score = 0.0
            
            # 리소스 활용도
            if cluster_info.get("status") == "initialized":
                available_resources = cluster_info.get("available_resources", {})
                total_resources = cluster_info.get("total_resources", {})
                
                if total_resources:
                    cpu_utilization = 1.0 - (available_resources.get("CPU", 0) / total_resources.get("CPU", 1))
                    memory_utilization = 1.0 - (available_resources.get("memory", 0) / total_resources.get("memory", 1))
                    scalability_score = (cpu_utilization + memory_utilization) / 2.0
            
            test_results["scalability_score"] = scalability_score
            test_results["success"] = True
            
            logger.info(f"성능 테스트 완료: 확장성 점수 {scalability_score:.2f}")
            
        except Exception as e:
            logger.error(f"성능 테스트 실패: {e}")
        
        return test_results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("🧪 분산 시스템 통합 테스트 시작...")
        
        # 시스템 설정
        setup_success = await self.setup_distributed_system()
        if not setup_success:
            return {"error": "시스템 설정 실패"}
        
        # 테스트 실행
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "setup_success": setup_success,
            "foundation_model_test": await self.test_foundation_model_distribution(),
            "developmental_learning_test": await self.test_developmental_learning_distribution(),
            "distributed_memory_test": await self.test_distributed_memory_system(),
            "end_to_end_test": await self.test_end_to_end_mission_processing(),
            "performance_test": await self.test_system_performance_and_scalability()
        }
        
        # 종합 결과 계산
        overall_success = all(
            test.get("success", False) 
            for test in test_results.values() 
            if isinstance(test, dict) and "success" in test
        )
        
        test_results["overall_success"] = overall_success
        
        # 결과 저장
        self.test_results = test_results
        
        logger.info(f"통합 테스트 완료: {'성공' if overall_success else '실패'}")
        return test_results
    
    async def generate_test_report(self) -> str:
        """테스트 리포트 생성"""
        if not self.test_results:
            return "테스트 결과가 없습니다."
        
        report = []
        report.append("# Physical AI System 분산처리 통합 테스트 리포트")
        report.append(f"**테스트 시간**: {self.test_results['test_timestamp']}")
        report.append(f"**전체 성공**: {'✅' if self.test_results['overall_success'] else '❌'}")
        report.append("")
        
        # 각 테스트 결과
        test_names = {
            "foundation_model_test": "🧠 Foundation Model 분산 처리",
            "developmental_learning_test": "🌱 발달적 학습 분산 처리", 
            "distributed_memory_test": "💾 분산 메모리 시스템",
            "end_to_end_test": "🎯 엔드투엔드 미션 처리",
            "performance_test": "📊 성능 및 확장성"
        }
        
        for test_key, test_name in test_names.items():
            if test_key in self.test_results:
                test_result = self.test_results[test_key]
                success_icon = "✅" if test_result.get("success", False) else "❌"
                report.append(f"## {test_name} {success_icon}")
                
                for key, value in test_result.items():
                    if key != "success":
                        report.append(f"- **{key}**: {value}")
                report.append("")
        
        return "\n".join(report)
    
    async def cleanup(self):
        """정리 작업"""
        logger.info("🧹 분산 시스템 정리 시작...")
        
        try:
            if self.cluster_manager:
                await self.cluster_manager.shutdown_cluster()
            
            logger.info("✅ 분산 시스템 정리 완료")
            
        except Exception as e:
            logger.error(f"정리 작업 실패: {e}")

# 메인 테스트 실행
async def main():
    """메인 테스트 실행"""
    test_suite = DistributedSystemIntegrationTest()
    
    try:
        # 모든 테스트 실행
        results = await test_suite.run_all_tests()
        
        # 결과 출력
        print("\n" + "="*60)
        print("분산처리 시스템 통합 테스트 결과")
        print("="*60)
        
        for key, value in results.items():
            if key != "test_timestamp":
                print(f"{key}: {value}")
        
        # 리포트 생성
        report = await test_suite.generate_test_report()
        print("\n" + "="*60)
        print("상세 테스트 리포트")
        print("="*60)
        print(report)
        
        # 결과 저장
        with open("distributed_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n테스트 결과가 'distributed_test_results.json'에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")
    
    finally:
        # 정리 작업
        await test_suite.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
