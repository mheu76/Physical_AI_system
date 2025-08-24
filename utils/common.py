"""
공통 유틸리티 함수들

코드 중복을 줄이기 위한 공통 함수들을 제공합니다.
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """성능 메트릭 관리 클래스"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """타이머 시작"""
        self.start_times[name] = datetime.now()
    
    def end_timer(self, name: str) -> float:
        """타이머 종료 및 경과 시간 반환"""
        if name not in self.start_times:
            return 0.0
        
        elapsed = (datetime.now() - self.start_times[name]).total_seconds()
        self.metrics[name] = elapsed
        return elapsed
    
    def get_metrics(self) -> Dict[str, Any]:
        """모든 메트릭 반환"""
        return self.metrics.copy()

class ConfigManager:
    """설정 관리 클래스"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ 설정 파일 로드 성공: {config_path}")
                return config
        except Exception as e:
            logger.warning(f"⚠️  설정 파일 로드 실패: {e}")
            logger.info("기본 설정 사용")
            return ConfigManager.get_default_config()
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "foundation_model": {
                "model_type": "phi35",
                "phi35": {
                    "model_name": "microsoft/Phi-3.5-mini-instruct",
                    "device": "auto"
                }
            },
            "system": {
                "name": "Physical AI System",
                "version": "2.0.0-phi35"
            }
        }

class DataValidator:
    """데이터 검증 클래스"""
    
    @staticmethod
    def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """스키마 검증"""
        try:
            # 필수 키 확인
            for key in schema:
                if key not in data:
                    return False
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: float, max_val: float) -> bool:
        """숫자 범위 검증"""
        return min_val <= value <= max_val

class AsyncHelper:
    """비동기 헬퍼 클래스"""
    
    @staticmethod
    async def retry_async(func, max_retries: int = 3, delay: float = 1.0):
        """비동기 함수 재시도"""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"재시도 {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(delay)
    
    @staticmethod
    async def timeout_async(func, timeout: float = 30.0):
        """비동기 함수 타임아웃"""
        try:
            return await asyncio.wait_for(func(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"함수 실행 타임아웃: {timeout}초")
            raise

class MathUtils:
    """수학 유틸리티 클래스"""
    
    @staticmethod
    def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """두 점 간의 거리 계산"""
        return np.linalg.norm(pos1 - pos2)
    
    @staticmethod
    def interpolate_linear(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """선형 보간"""
        return start + t * (end - start)
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """값을 범위 내로 제한"""
        return max(min_val, min(value, max_val))

class FileUtils:
    """파일 유틸리티 클래스"""
    
    @staticmethod
    def ensure_directory(path: str) -> bool:
        """디렉토리 존재 확인 및 생성"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"디렉토리 생성 실패: {e}")
            return False
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str) -> bool:
        """JSON 파일 저장"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            logger.error(f"JSON 저장 실패: {e}")
            return False
    
    @staticmethod
    def load_json(filepath: str) -> Optional[Dict[str, Any]]:
        """JSON 파일 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"JSON 로드 실패: {e}")
            return None

class SafetyUtils:
    """안전 관련 유틸리티 클래스"""
    
    @staticmethod
    def check_collision_risk(position: np.ndarray, obstacles: List[np.ndarray], 
                           safety_margin: float = 0.1) -> bool:
        """충돌 위험 확인"""
        for obstacle in obstacles:
            distance = MathUtils.calculate_distance(position, obstacle)
            if distance < safety_margin:
                return True
        return False
    
    @staticmethod
    def validate_joint_limits(position: float, min_limit: float, max_limit: float) -> bool:
        """관절 한계 검증"""
        return min_limit <= position <= max_limit
    
    @staticmethod
    def calculate_safety_score(velocity: float, distance_to_obstacle: float, 
                             max_velocity: float = 2.0) -> float:
        """안전 점수 계산"""
        velocity_score = 1.0 - (velocity / max_velocity)
        distance_score = min(distance_to_obstacle / 1.0, 1.0)  # 1m 이상이면 최대 점수
        return (velocity_score + distance_score) / 2.0
