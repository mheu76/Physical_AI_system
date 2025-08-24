"""
Distributed Memory System

Physical AI System의 분산 메모리 관리 시스템
"""

import ray
import redis
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """메모리 엔트리"""
    entry_id: str
    memory_type: str  # episodic, semantic, working
    content: Dict[str, Any]
    timestamp: datetime
    priority: float
    access_count: int = 0
    last_accessed: datetime = None

@dataclass
class MemoryQuery:
    """메모리 쿼리"""
    query_id: str
    memory_type: str
    search_criteria: Dict[str, Any]
    max_results: int = 10
    similarity_threshold: float = 0.7

@ray.remote
class DistributedMemoryManager:
    """분산 메모리 관리자"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            "host": "localhost",
            "port": 6379,
            "db": 0
        }
        self.redis_client = None
        self.memory_cache = {}
        self.access_patterns = {}
        
        logger.info("Distributed Memory Manager 초기화")
    
    async def initialize(self) -> bool:
        """메모리 시스템 초기화"""
        try:
            # Redis 연결
            self.redis_client = redis.Redis(**self.redis_config)
            await self.redis_client.ping()
            
            # 메모리 타입별 초기화
            memory_types = ["episodic", "semantic", "working"]
            for mem_type in memory_types:
                await self._initialize_memory_type(mem_type)
            
            logger.info("Distributed Memory Manager 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"Distributed Memory Manager 초기화 실패: {e}")
            return False
    
    async def _initialize_memory_type(self, memory_type: str):
        """메모리 타입별 초기화"""
        # Redis에 메모리 타입별 키 생성
        key = f"memory:{memory_type}:index"
        if not await self.redis_client.exists(key):
            await self.redis_client.set(key, json.dumps([]))
    
    async def store_memory(self, memory_entry: MemoryEntry) -> bool:
        """메모리 저장"""
        try:
            # 메모리 엔트리 직렬화
            entry_data = asdict(memory_entry)
            entry_data["timestamp"] = memory_entry.timestamp.isoformat()
            if memory_entry.last_accessed:
                entry_data["last_accessed"] = memory_entry.last_accessed.isoformat()
            
            # Redis에 저장
            key = f"memory:{memory_entry.memory_type}:{memory_entry.entry_id}"
            await self.redis_client.set(key, json.dumps(entry_data))
            
            # 인덱스 업데이트
            await self._update_memory_index(memory_entry.memory_type, memory_entry.entry_id)
            
            # 로컬 캐시 업데이트
            if memory_entry.memory_type not in self.memory_cache:
                self.memory_cache[memory_entry.memory_type] = {}
            self.memory_cache[memory_entry.memory_type][memory_entry.entry_id] = memory_entry
            
            logger.info(f"메모리 저장 완료: {memory_entry.entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"메모리 저장 실패: {e}")
            return False
    
    async def _update_memory_index(self, memory_type: str, entry_id: str):
        """메모리 인덱스 업데이트"""
        index_key = f"memory:{memory_type}:index"
        index_data = await self.redis_client.get(index_key)
        
        if index_data:
            index_list = json.loads(index_data)
            if entry_id not in index_list:
                index_list.append(entry_id)
                await self.redis_client.set(index_key, json.dumps(index_list))
    
    async def retrieve_memory(self, query: MemoryQuery) -> List[MemoryEntry]:
        """메모리 검색"""
        try:
            results = []
            
            # 인덱스에서 엔트리 ID 목록 가져오기
            index_key = f"memory:{query.memory_type}:index"
            index_data = await self.redis_client.get(index_key)
            
            if not index_data:
                return results
            
            entry_ids = json.loads(index_data)
            
            # 각 엔트리 검사
            for entry_id in entry_ids:
                entry = await self._get_memory_entry(query.memory_type, entry_id)
                if entry and self._matches_query(entry, query):
                    results.append(entry)
                    
                    # 접근 패턴 기록
                    await self._record_access_pattern(query.memory_type, entry_id)
            
            # 우선순위 및 접근 빈도로 정렬
            results.sort(key=lambda x: (x.priority, x.access_count), reverse=True)
            
            return results[:query.max_results]
            
        except Exception as e:
            logger.error(f"메모리 검색 실패: {e}")
            return []
    
    async def _get_memory_entry(self, memory_type: str, entry_id: str) -> Optional[MemoryEntry]:
        """메모리 엔트리 가져오기"""
        # 로컬 캐시 확인
        if (memory_type in self.memory_cache and 
            entry_id in self.memory_cache[memory_type]):
            return self.memory_cache[memory_type][entry_id]
        
        # Redis에서 가져오기
        key = f"memory:{memory_type}:{entry_id}"
        entry_data = await self.redis_client.get(key)
        
        if entry_data:
            data = json.loads(entry_data)
            entry = MemoryEntry(
                entry_id=data["entry_id"],
                memory_type=data["memory_type"],
                content=data["content"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                priority=data["priority"],
                access_count=data["access_count"],
                last_accessed=datetime.fromisoformat(data["last_accessed"]) if data["last_accessed"] else None
            )
            
            # 로컬 캐시에 저장
            if memory_type not in self.memory_cache:
                self.memory_cache[memory_type] = {}
            self.memory_cache[memory_type][entry_id] = entry
            
            return entry
        
        return None
    
    def _matches_query(self, entry: MemoryEntry, query: MemoryQuery) -> bool:
        """쿼리 매칭 확인"""
        # 기본 타입 매칭
        if entry.memory_type != query.memory_type:
            return False
        
        # 검색 기준 확인
        for key, value in query.search_criteria.items():
            if key not in entry.content:
                return False
            
            # 유사도 검사 (간단한 구현)
            if isinstance(value, (int, float)) and isinstance(entry.content[key], (int, float)):
                if abs(entry.content[key] - value) > 0.1:  # 임계값
                    return False
            elif entry.content[key] != value:
                return False
        
        return True
    
    async def _record_access_pattern(self, memory_type: str, entry_id: str):
        """접근 패턴 기록"""
        pattern_key = f"access_pattern:{memory_type}:{entry_id}"
        current_count = await self.redis_client.get(pattern_key)
        
        if current_count:
            new_count = int(current_count) + 1
        else:
            new_count = 1
        
        await self.redis_client.set(pattern_key, new_count)
        
        # 엔트리 업데이트
        entry = await self._get_memory_entry(memory_type, entry_id)
        if entry:
            entry.access_count = new_count
            entry.last_accessed = datetime.now()
            await self.store_memory(entry)
    
    async def update_memory_priority(self, entry_id: str, memory_type: str, new_priority: float) -> bool:
        """메모리 우선순위 업데이트"""
        try:
            entry = await self._get_memory_entry(memory_type, entry_id)
            if entry:
                entry.priority = new_priority
                await self.store_memory(entry)
                return True
            return False
        except Exception as e:
            logger.error(f"메모리 우선순위 업데이트 실패: {e}")
            return False
    
    async def cleanup_old_memories(self, memory_type: str, max_age_hours: int = 24) -> int:
        """오래된 메모리 정리"""
        try:
            cleaned_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            index_key = f"memory:{memory_type}:index"
            index_data = await self.redis_client.get(index_key)
            
            if not index_data:
                return 0
            
            entry_ids = json.loads(index_data)
            remaining_ids = []
            
            for entry_id in entry_ids:
                entry = await self._get_memory_entry(memory_type, entry_id)
                if entry and entry.timestamp > cutoff_time:
                    remaining_ids.append(entry_id)
                else:
                    # 오래된 메모리 삭제
                    key = f"memory:{memory_type}:{entry_id}"
                    await self.redis_client.delete(key)
                    cleaned_count += 1
                    
                    # 로컬 캐시에서도 제거
                    if (memory_type in self.memory_cache and 
                        entry_id in self.memory_cache[memory_type]):
                        del self.memory_cache[memory_type][entry_id]
            
            # 인덱스 업데이트
            await self.redis_client.set(index_key, json.dumps(remaining_ids))
            
            logger.info(f"메모리 정리 완료: {cleaned_count}개 삭제")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"메모리 정리 실패: {e}")
            return 0
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        try:
            stats = {}
            
            for memory_type in ["episodic", "semantic", "working"]:
                index_key = f"memory:{memory_type}:index"
                index_data = await self.redis_client.get(index_key)
                
                if index_data:
                    entry_ids = json.loads(index_data)
                    stats[memory_type] = {
                        "total_entries": len(entry_ids),
                        "cached_entries": len(self.memory_cache.get(memory_type, {}))
                    }
                else:
                    stats[memory_type] = {"total_entries": 0, "cached_entries": 0}
            
            return stats
            
        except Exception as e:
            logger.error(f"메모리 통계 수집 실패: {e}")
            return {}

@ray.remote
class ExperienceReplayBuffer:
    """경험 재생 버퍼"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = []
        self.priorities = []
        self.position = 0
        
        logger.info(f"Experience Replay Buffer 초기화: 최대 크기 {max_size}")
    
    async def add_experience(self, experience: Dict[str, Any], priority: float = 1.0):
        """경험 추가"""
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # 버퍼가 가득 찬 경우, 가장 낮은 우선순위의 경험 교체
            min_priority_idx = min(range(len(self.priorities)), key=lambda i: self.priorities[i])
            self.buffer[min_priority_idx] = experience
            self.priorities[min_priority_idx] = priority
    
    async def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """배치 샘플링"""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        # 우선순위 기반 샘플링
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)
        return [self.buffer[i] for i in indices]
    
    async def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """우선순위 업데이트"""
        for idx, priority in zip(indices, new_priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """버퍼 통계"""
        return {
            "current_size": len(self.buffer),
            "max_size": self.max_size,
            "avg_priority": np.mean(self.priorities) if self.priorities else 0.0,
            "min_priority": np.min(self.priorities) if self.priorities else 0.0,
            "max_priority": np.max(self.priorities) if self.priorities else 0.0
        }

@ray.remote
class DistributedMemoryOrchestrator:
    """분산 메모리 오케스트레이터"""
    
    def __init__(self, num_memory_managers: int = 2, num_replay_buffers: int = 2):
        self.num_memory_managers = num_memory_managers
        self.num_replay_buffers = num_replay_buffers
        self.memory_managers = []
        self.replay_buffers = []
        
        logger.info(f"Distributed Memory Orchestrator 초기화")
    
    async def initialize(self) -> bool:
        """오케스트레이터 초기화"""
        try:
            # 메모리 매니저 초기화
            for i in range(self.num_memory_managers):
                manager = DistributedMemoryManager.remote()
                success = await manager.initialize.remote()
                if success:
                    self.memory_managers.append(manager)
                    logger.info(f"Memory Manager {i} 초기화 완료")
            
            # 경험 재생 버퍼 초기화
            for i in range(self.num_replay_buffers):
                buffer = ExperienceReplayBuffer.remote(max_size=50000)
                self.replay_buffers.append(buffer)
                logger.info(f"Experience Replay Buffer {i} 초기화 완료")
            
            logger.info(f"Distributed Memory Orchestrator 초기화 완료")
            return len(self.memory_managers) > 0
            
        except Exception as e:
            logger.error(f"Distributed Memory Orchestrator 초기화 실패: {e}")
            return False
    
    async def store_memory_distributed(self, memory_entry: MemoryEntry) -> bool:
        """분산 메모리 저장"""
        if not self.memory_managers:
            return False
        
        # 라운드 로빈으로 메모리 매니저 선택
        manager_idx = hash(memory_entry.entry_id) % len(self.memory_managers)
        manager = self.memory_managers[manager_idx]
        
        return await manager.store_memory.remote(memory_entry)
    
    async def retrieve_memory_distributed(self, query: MemoryQuery) -> List[MemoryEntry]:
        """분산 메모리 검색"""
        if not self.memory_managers:
            return []
        
        # 모든 메모리 매니저에서 검색
        search_tasks = []
        for manager in self.memory_managers:
            task = manager.retrieve_memory.remote(query)
            search_tasks.append(task)
        
        # 병렬 검색
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # 결과 통합
        all_results = []
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
        
        # 중복 제거 및 정렬
        unique_results = {}
        for entry in all_results:
            if entry.entry_id not in unique_results:
                unique_results[entry.entry_id] = entry
            else:
                # 더 높은 우선순위의 결과 선택
                if entry.priority > unique_results[entry.entry_id].priority:
                    unique_results[entry.entry_id] = entry
        
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: (x.priority, x.access_count), reverse=True)
        
        return final_results[:query.max_results]
    
    async def add_experience_distributed(self, experience: Dict[str, Any], priority: float = 1.0):
        """분산 경험 추가"""
        if not self.replay_buffers:
            return
        
        # 라운드 로빈으로 버퍼 선택
        buffer_idx = hash(str(experience)) % len(self.replay_buffers)
        buffer = self.replay_buffers[buffer_idx]
        
        await buffer.add_experience.remote(experience, priority)
    
    async def sample_experience_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """경험 배치 샘플링"""
        if not self.replay_buffers:
            return []
        
        # 모든 버퍼에서 샘플링
        sample_tasks = []
        samples_per_buffer = batch_size // len(self.replay_buffers)
        
        for buffer in self.replay_buffers:
            task = buffer.sample_batch.remote(samples_per_buffer)
            sample_tasks.append(task)
        
        # 병렬 샘플링
        results = await asyncio.gather(*sample_tasks, return_exceptions=True)
        
        # 결과 통합
        all_samples = []
        for result in results:
            if isinstance(result, list):
                all_samples.extend(result)
        
        return all_samples[:batch_size]
    
    async def get_distributed_stats(self) -> Dict[str, Any]:
        """분산 통계"""
        stats = {
            "memory_managers": [],
            "replay_buffers": [],
            "total_memory_entries": 0,
            "total_experiences": 0
        }
        
        # 메모리 매니저 통계
        for i, manager in enumerate(self.memory_managers):
            manager_stats = await manager.get_memory_stats.remote()
            stats["memory_managers"].append({
                "manager_id": i,
                "stats": manager_stats
            })
            
            # 총 메모리 엔트리 수 계산
            for mem_type, mem_stats in manager_stats.items():
                stats["total_memory_entries"] += mem_stats["total_entries"]
        
        # 경험 재생 버퍼 통계
        for i, buffer in enumerate(self.replay_buffers):
            buffer_stats = await buffer.get_buffer_stats.remote()
            stats["replay_buffers"].append({
                "buffer_id": i,
                "stats": buffer_stats
            })
            
            stats["total_experiences"] += buffer_stats["current_size"]
        
        return stats

# 테스트 함수
async def test_distributed_memory():
    """분산 메모리 시스템 테스트"""
    # 오케스트레이터 생성
    orchestrator = DistributedMemoryOrchestrator(
        num_memory_managers=2,
        num_replay_buffers=2
    )
    
    # 초기화
    success = await orchestrator.initialize()
    if not success:
        print("오케스트레이터 초기화 실패")
        return
    
    # 테스트 메모리 저장
    test_memories = []
    for i in range(10):
        memory = MemoryEntry(
            entry_id=f"test_memory_{i}",
            memory_type="episodic",
            content={"action": f"action_{i}", "reward": i * 0.1},
            timestamp=datetime.now(),
            priority=1.0 - i * 0.1
        )
        test_memories.append(memory)
        
        success = await orchestrator.store_memory_distributed(memory)
        print(f"메모리 저장 {i}: {'성공' if success else '실패'}")
    
    # 테스트 메모리 검색
    query = MemoryQuery(
        query_id="test_query",
        memory_type="episodic",
        search_criteria={"action": "action_5"},
        max_results=5
    )
    
    results = await orchestrator.retrieve_memory_distributed(query)
    print(f"검색 결과: {len(results)}개")
    
    # 테스트 경험 추가
    for i in range(20):
        experience = {
            "state": f"state_{i}",
            "action": f"action_{i}",
            "reward": i * 0.1,
            "next_state": f"state_{i+1}"
        }
        await orchestrator.add_experience_distributed(experience, priority=1.0 - i * 0.05)
    
    # 테스트 경험 샘플링
    samples = await orchestrator.sample_experience_batch(10)
    print(f"샘플링된 경험: {len(samples)}개")
    
    # 통계 확인
    stats = await orchestrator.get_distributed_stats()
    print(f"분산 메모리 통계: {stats}")

if __name__ == "__main__":
    asyncio.run(test_distributed_memory())
