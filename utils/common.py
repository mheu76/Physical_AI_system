"""
유틸리티 함수들

Physical AI 시스템에서 공통으로 사용하는 유틸리티 함수들
"""

import yaml
import logging
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import asyncio
import time
from dataclasses import dataclass
import json

@dataclass
class Vector3D:
    """3D 벡터 클래스"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector3D(self.x/mag, self.y/mag, self.z/mag)
        return Vector3D(0, 0, 0)
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        return cls(arr[0], arr[1], arr[2])

def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"설정 파일 로드 실패: {e}")
        return {}

def setup_logging(config: Dict[str, Any]):
    """로깅 설정"""
    log_config = config.get('monitoring', {}).get('logging', {})
    
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 로그 디렉토리 생성
    log_file = log_config.get('file_path', 'logs/physical_ai.log')
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def calculate_distance(pos1: Vector3D, pos2: Vector3D) -> float:
    """두 점 간 거리 계산"""
    return (pos2 - pos1).magnitude()

def interpolate_positions(start: Vector3D, end: Vector3D, t: float) -> Vector3D:
    """두 위치 간 선형 보간"""
    t = max(0.0, min(1.0, t))  # 0~1 사이로 클램프
    return start + (end - start) * t

def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """쿼터니언을 오일러 각으로 변환"""
    x, y, z, w = quat
    
    # Roll (x축 회전)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y축 회전)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z축 회전)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """오일러 각을 쿼터니언으로 변환"""
    roll, pitch, yaw = euler
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([x, y, z, w])

def clamp(value: float, min_val: float, max_val: float) -> float:
    """값을 범위로 제한"""
    return max(min_val, min(max_val, value))

def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """부드러운 스텝 함수"""
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3 - 2 * x)

def exponential_moving_average(new_value: float, old_average: float, alpha: float) -> float:
    """지수 이동 평균"""
    return alpha * new_value + (1 - alpha) * old_average

class Timer:
    """시간 측정 유틸리티"""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        logging.info(f"{self.name}: {elapsed:.4f}초")

class AsyncTimer:
    """비동기 시간 측정"""
    
    def __init__(self, name: str = "AsyncTimer"):
        self.name = name
        self.start_time = None
        
    async def __aenter__(self):
        self.start_time = asyncio.get_event_loop().time()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed = asyncio.get_event_loop().time() - self.start_time
        logging.info(f"{self.name}: {elapsed:.4f}초")

def save_json(data: Any, file_path: str, indent: int = 2):
    """JSON 파일 저장"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        logging.error(f"JSON 저장 실패: {e}")

def load_json(file_path: str) -> Any:
    """JSON 파일 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"JSON 로드 실패: {e}")
        return None

def create_safety_zone(center: Vector3D, radius: float) -> Dict[str, Any]:
    """안전 구역 생성"""
    return {
        "center": center,
        "radius": radius,
        "type": "sphere"
    }

def is_point_in_safety_zone(point: Vector3D, safety_zone: Dict[str, Any]) -> bool:
    """점이 안전 구역 내부에 있는지 확인"""
    center = safety_zone["center"]
    radius = safety_zone["radius"]
    
    distance = calculate_distance(point, center)
    return distance <= radius

def calculate_trajectory_length(waypoints: List[Vector3D]) -> float:
    """궤적의 총 길이 계산"""
    total_length = 0.0
    for i in range(1, len(waypoints)):
        total_length += calculate_distance(waypoints[i-1], waypoints[i])
    return total_length

def validate_joint_limits(joint_angles: np.ndarray, 
                         joint_limits: List[tuple]) -> np.ndarray:
    """관절 한계 내로 각도 제한"""
    validated_angles = joint_angles.copy()
    
    for i, (min_angle, max_angle) in enumerate(joint_limits):
        if i < len(validated_angles):
            validated_angles[i] = clamp(validated_angles[i], min_angle, max_angle)
    
    return validated_angles

def calculate_energy_consumption(velocity: np.ndarray, 
                               acceleration: np.ndarray, 
                               mass: float = 1.0,
                               dt: float = 0.01) -> float:
    """에너지 소모량 계산 (간단한 모델)"""
    kinetic_energy = 0.5 * mass * np.sum(velocity**2)
    work_done = mass * np.sum(np.abs(acceleration * velocity)) * dt
    return kinetic_energy + work_done

class RateLimiter:
    """비율 제한기"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def is_allowed(self) -> bool:
        now = time.time()
        
        # 시간 윈도우 밖의 호출 제거
        self.calls = [call_time for call_time in self.calls 
                      if now - call_time < self.time_window]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False

def format_duration(seconds: float) -> str:
    """초를 사람이 읽기 쉬운 형태로 변환"""
    if seconds < 60:
        return f"{seconds:.2f}초"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}분 {remaining_seconds:.1f}초"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}시간 {remaining_minutes}분"

# 물리 상수들
PHYSICS_CONSTANTS = {
    "gravity": 9.81,           # m/s²
    "air_density": 1.225,      # kg/m³
    "water_density": 1000.0,   # kg/m³
    "speed_of_sound": 343.0,   # m/s
    "pi": np.pi
}
