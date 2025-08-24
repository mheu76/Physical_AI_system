"""
Hardware Abstraction Layer - 하드웨어 추상화 레이어

다양한 로봇 하드웨어 플랫폼을 추상화하여 
상위 레이어가 하드웨어에 독립적으로 동작할 수 있게 합니다.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager
import threading
import weakref

# Import our enhanced error handling and performance monitoring
try:
    from utils.error_handling import PhysicalAIException, HardwareError, safe_async_call, require_initialization, validate_input
    from utils.performance_monitor import profile_function, performance_context, performance_monitor
except ImportError:
    # Fallback if utils not available
    logger.warning("Enhanced error handling and performance monitoring not available, using basic fallbacks")
    
    class PhysicalAIException(Exception):
        pass
    
    class HardwareError(Exception):
        pass
    
    def safe_async_call(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def require_initialization(func):
        return func
    
    def validate_input(value, expected_type, **kwargs):
        return value
    
    def profile_function(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

class SensorStatus(Enum):
    """센서 상태"""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    CALIBRATING = "calibrating"

class ActuatorStatus(Enum):
    """액추에이터 상태"""
    OFFLINE = "offline"
    READY = "ready"
    EXECUTING = "executing"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class SensorData:
    """강화된 센서 데이터 구조"""
    timestamp: datetime = field(default_factory=datetime.now)
    sensor_type: str = ""
    sensor_id: str = ""
    data: Any = None
    confidence: float = 1.0
    status: SensorStatus = SensorStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    data_quality: float = 1.0  # 0.0 (poor) to 1.0 (excellent)
    
    def __post_init__(self):
        """Validate sensor data"""
        self.confidence = validate_input(self.confidence, (int, float), "confidence", min_value=0.0, max_value=1.0)
        self.data_quality = validate_input(self.data_quality, (int, float), "data_quality", min_value=0.0, max_value=1.0)
        
    def is_valid(self) -> bool:
        """Check if sensor data is valid and usable"""
        return (self.status == SensorStatus.ACTIVE and 
                self.confidence >= 0.5 and 
                self.data_quality >= 0.5 and
                self.data is not None)

@dataclass
class ActuatorCommand:
    """강화된 액추에이터 명령 구조"""
    actuator_id: str
    command_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    timeout: float = 5.0  # seconds
    safety_checks: bool = True
    command_id: str = field(default_factory=lambda: f"cmd_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    expected_duration: Optional[float] = None
    
    def __post_init__(self):
        """Validate command parameters"""
        self.priority = validate_input(self.priority, int, "priority", min_value=0, max_value=10)
        self.timeout = validate_input(self.timeout, (int, float), "timeout", min_value=0.1, max_value=300.0)
        
        if not self.actuator_id:
            raise ValueError("actuator_id cannot be empty")
        if not self.command_type:
            raise ValueError("command_type cannot be empty")

@dataclass
class HardwareHealth:
    """하드웨어 상태 정보"""
    device_id: str
    device_type: str
    status: Union[SensorStatus, ActuatorStatus]
    last_update: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    uptime: float = 0.0
    temperature: Optional[float] = None
    power_consumption: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if hardware is healthy"""
        return (self.status not in [SensorStatus.ERROR, ActuatorStatus.ERROR, ActuatorStatus.EMERGENCY_STOP] and
                self.error_count < 10)

class SensorInterface(ABC):
    """강화된 센서 인터페이스 추상 클래스"""
    
    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self._initialized = False
        self._status = SensorStatus.OFFLINE
        self._health = HardwareHealth(sensor_id, self.__class__.__name__, SensorStatus.OFFLINE)
        self._lock = asyncio.Lock()
        
    @abstractmethod
    async def initialize(self) -> bool:
        """센서 초기화"""
        pass
    
    @abstractmethod
    async def read_data(self) -> SensorData:
        """센서 데이터 읽기"""
        pass
    
    @abstractmethod
    async def calibrate(self) -> bool:
        """센서 캘리브레이션"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """센서 종료"""
        pass
    
    @property
    def status(self) -> SensorStatus:
        return self._status
    
    @property
    def health(self) -> HardwareHealth:
        self._health.last_update = datetime.now()
        return self._health
    
    async def get_diagnostics(self) -> Dict[str, Any]:
        """센서 진단 정보"""
        return {
            "sensor_id": self.sensor_id,
            "status": self._status.value,
            "initialized": self._initialized,
            "health": self._health.__dict__
        }

class ActuatorInterface(ABC):
    """강화된 액추에이터 인터페이스 추상 클래스"""
    
    def __init__(self, actuator_id: str):
        self.actuator_id = actuator_id
        self._initialized = False
        self._status = ActuatorStatus.OFFLINE
        self._health = HardwareHealth(actuator_id, self.__class__.__name__, ActuatorStatus.OFFLINE)
        self._lock = asyncio.Lock()
        self._emergency_stop = asyncio.Event()
        
    @abstractmethod
    async def initialize(self) -> bool:
        """액추에이터 초기화"""
        pass
    
    @abstractmethod
    async def execute_command(self, command: ActuatorCommand) -> bool:
        """명령 실행"""
        pass
    
    @abstractmethod
    async def emergency_stop(self) -> bool:
        """긴급 정지"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """액추에이터 종료"""
        pass
    
    @property
    def status(self) -> ActuatorStatus:
        return self._status
    
    @property
    def health(self) -> HardwareHealth:
        self._health.last_update = datetime.now()
        return self._health
    
    async def get_diagnostics(self) -> Dict[str, Any]:
        """액추에이터 진단 정보"""
        return {
            "actuator_id": self.actuator_id,
            "status": self._status.value,
            "initialized": self._initialized,
            "emergency_stop": self._emergency_stop.is_set(),
            "health": self._health.__dict__
        }

class VisionSensor(SensorInterface):
    """강화된 비전 센서 (RGB-D 카메라)"""
    
    def __init__(self, sensor_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(sensor_id)
        self._config = config or {}
        self.resolution = self._config.get('resolution', (640, 480))
        self.fps = self._config.get('fps', 30)
        self.auto_calibrate = self._config.get('auto_calibrate', True)
        
        # Mock camera data for simulation
        self._simulation_mode = self._config.get('simulation_mode', True)
        self._frame_count = 0
        
    @require_initialization
    @safe_async_call(fallback_value=False, max_retries=3, component="VisionSensor", operation="initialize")
    async def initialize(self) -> bool:
        """비전 센서 초기화"""
        try:
            self._status = SensorStatus.INITIALIZING
            
            if self._simulation_mode:
                logger.info(f"Initializing vision sensor {self.sensor_id} in simulation mode")
                # Simulate initialization time
                await asyncio.sleep(0.5)
            else:
                # Real camera initialization would go here
                logger.info(f"Initializing real camera {self.sensor_id}")
            
            # Auto-calibrate if enabled
            if self.auto_calibrate:
                await self.calibrate()
            
            self._status = SensorStatus.ACTIVE
            self._initialized = True
            self._health.status = SensorStatus.ACTIVE
            
            logger.info(f"Vision sensor {self.sensor_id} initialized successfully")
            return True
            
        except Exception as e:
            self._status = SensorStatus.ERROR
            self._health.status = SensorStatus.ERROR
            self._health.error_count += 1
            logger.error(f"Failed to initialize vision sensor {self.sensor_id}: {e}")
            return False
    
    @require_initialization
    @profile_function(include_memory=True, category="sensor")
    async def read_data(self) -> SensorData:
        """비전 데이터 읽기"""
        async with self._lock:
            try:
                if self._simulation_mode:
                    # Generate mock RGB-D data
                    rgb_data = np.random.randint(0, 255, (*self.resolution, 3), dtype=np.uint8)
                    depth_data = np.random.uniform(0.1, 5.0, self.resolution)
                    
                    data = {
                        'rgb': rgb_data,
                        'depth': depth_data,
                        'frame_id': self._frame_count,
                        'resolution': self.resolution,
                        'fps': self.fps
                    }
                    
                    confidence = 0.95  # High confidence for simulation
                    quality = 0.9
                else:
                    # Real camera data acquisition would go here
                    data = {}
                    confidence = 0.8
                    quality = 0.85
                
                self._frame_count += 1
                
                return SensorData(
                    sensor_type="vision",
                    sensor_id=self.sensor_id,
                    data=data,
                    confidence=confidence,
                    status=self._status,
                    data_quality=quality,
                    metadata={
                        'frame_count': self._frame_count,
                        'simulation_mode': self._simulation_mode
                    }
                )
                
            except Exception as e:
                self._health.error_count += 1
                logger.error(f"Failed to read vision data from {self.sensor_id}: {e}")
                raise HardwareError(f"Vision sensor read failed: {e}", self.sensor_id, "camera")
    
    @safe_async_call(fallback_value=False, max_retries=2, component="VisionSensor", operation="calibrate")
    async def calibrate(self) -> bool:
        """비전 센서 캘리브레이션"""
        try:
            self._status = SensorStatus.CALIBRATING
            
            # Simulate calibration process
            logger.info(f"Calibrating vision sensor {self.sensor_id}")
            await asyncio.sleep(1.0)  # Simulate calibration time
            
            self._status = SensorStatus.ACTIVE
            logger.info(f"Vision sensor {self.sensor_id} calibrated successfully")
            return True
            
        except Exception as e:
            self._status = SensorStatus.ERROR
            self._health.error_count += 1
            logger.error(f"Failed to calibrate vision sensor {self.sensor_id}: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """비전 센서 종료"""
        try:
            self._status = SensorStatus.OFFLINE
            self._initialized = False
            logger.info(f"Vision sensor {self.sensor_id} shutdown complete")
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown vision sensor {self.sensor_id}: {e}")
            return False
        self.sensor_id = sensor_id
        self.resolution = (640, 480)
        self.fps = 30
        self.calibrated = False
        
    async def read_data(self) -> SensorData:
        """이미지 데이터 읽기"""
        # 시뮬레이션: 랜덤한 이미지 데이터
        rgb_data = np.random.randint(0, 255, (*self.resolution, 3), dtype=np.uint8)
        depth_data = np.random.uniform(0.1, 5.0, self.resolution)
        
        return SensorData(
            timestamp=asyncio.get_event_loop().time(),
            sensor_type="vision",
            data={
                "rgb": rgb_data,
                "depth": depth_data,
                "objects_detected": self._detect_objects(rgb_data)
            },
            confidence=0.85
        )
    
    def _detect_objects(self, rgb_data: np.ndarray) -> List[Dict[str, Any]]:
        """간단한 객체 감지 시뮬레이션"""
        # 실제로는 YOLO, RCNN 등의 모델 사용
        objects = [
            {
                "class": "cup",
                "bbox": [100, 150, 50, 80],
                "confidence": 0.9,
                "position_3d": [1.2, 0.3, 0.8]
            },
            {
                "class": "table",
                "bbox": [200, 300, 200, 100],
                "confidence": 0.95,
                "position_3d": [1.5, 0.0, 0.0]
            }
        ]
        return objects
    
    async def calibrate(self) -> bool:
        """카메라 캘리브레이션"""
        logger.info(f"비전 센서 {self.sensor_id} 캘리브레이션 중...")
        await asyncio.sleep(2)  # 캘리브레이션 시간
        self.calibrated = True
        logger.info(f"비전 센서 {self.sensor_id} 캘리브레이션 완료")
        return True

class TactileSensor(SensorInterface):
    """촉각 센서"""
    
    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.sensitivity = 0.1  # Newton
        self.calibrated = False
        
    async def read_data(self) -> SensorData:
        """촉각 데이터 읽기"""
        # 시뮬레이션: 압력, 온도, 질감 데이터
        pressure = np.random.uniform(0, 10)  # Newton
        temperature = np.random.uniform(15, 35)  # Celsius
        texture_roughness = np.random.uniform(0, 1)
        
        return SensorData(
            timestamp=asyncio.get_event_loop().time(),
            sensor_type="tactile",
            data={
                "pressure": pressure,
                "temperature": temperature,
                "texture": texture_roughness,
                "contact_detected": pressure > self.sensitivity
            },
            confidence=0.92
        )
    
    async def calibrate(self) -> bool:
        """촉각 센서 캘리브레이션"""
        logger.info(f"촉각 센서 {self.sensor_id} 캘리브레이션 중...")
        await asyncio.sleep(1)
        self.calibrated = True
        logger.info(f"촉각 센서 {self.sensor_id} 캘리브레이션 완료")
        return True

class IMUSensor(SensorInterface):
    """관성 측정 장치"""
    
    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.calibrated = False
        
    async def read_data(self) -> SensorData:
        """IMU 데이터 읽기"""
        # 가속도, 각속도, 자기장 데이터
        acceleration = np.random.normal(0, 0.1, 3)  # m/s²
        gyroscope = np.random.normal(0, 0.05, 3)   # rad/s
        magnetometer = np.random.normal([0, 0, 1], 0.1, 3)  # normalized
        
        return SensorData(
            timestamp=asyncio.get_event_loop().time(),
            sensor_type="imu",
            data={
                "acceleration": acceleration,
                "angular_velocity": gyroscope,
                "magnetic_field": magnetometer,
                "orientation": self._calculate_orientation(acceleration, magnetometer)
            },
            confidence=0.88
        )
    
    def _calculate_orientation(self, accel: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """자세 계산 (간단한 버전)"""
        # 실제로는 칼만 필터나 상보 필터 사용
        roll = np.arctan2(accel[1], accel[2])
        pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        yaw = np.arctan2(mag[1], mag[0])
        return np.array([roll, pitch, yaw])
    
    async def calibrate(self) -> bool:
        """IMU 캘리브레이션"""
        logger.info(f"IMU 센서 {self.sensor_id} 캘리브레이션 중...")
        await asyncio.sleep(3)  # 정적 캘리브레이션
        self.calibrated = True
        logger.info(f"IMU 센서 {self.sensor_id} 캘리브레이션 완료")
        return True

class ActuatorInterface(ABC):
    """액추에이터 인터페이스 추상 클래스"""
    
    @abstractmethod
    async def execute_command(self, command: ActuatorCommand) -> bool:
        """명령 실행"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """액추에이터 상태 조회"""
        pass

class ServoMotor(ActuatorInterface):
    """서보 모터"""
    
    def __init__(self, motor_id: str, joint_limits: Tuple[float, float]):
        self.motor_id = motor_id
        self.joint_limits = joint_limits  # (min_angle, max_angle) in radians
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.max_velocity = 3.14  # rad/s
        self.max_torque = 10.0  # Nm
        
    async def execute_command(self, command: ActuatorCommand) -> bool:
        """서보 모터 명령 실행"""
        cmd_type = command.command_type
        params = command.parameters
        
        try:
            if cmd_type == "move_to_position":
                target_position = params.get("position", 0.0)
                speed = params.get("speed", 1.0)
                
                # 관절 한계 확인
                if not (self.joint_limits[0] <= target_position <= self.joint_limits[1]):
                    logger.warning(f"모터 {self.motor_id}: 관절 한계 초과")
                    return False
                
                # 이동 시뮬레이션
                await self._move_to_position(target_position, speed)
                return True
                
            elif cmd_type == "set_velocity":
                target_velocity = params.get("velocity", 0.0)
                target_velocity = np.clip(target_velocity, -self.max_velocity, self.max_velocity)
                
                self.current_velocity = target_velocity
                logger.info(f"모터 {self.motor_id}: 속도 설정 {target_velocity:.2f} rad/s")
                return True
                
            else:
                logger.warning(f"모터 {self.motor_id}: 알 수 없는 명령 {cmd_type}")
                return False
                
        except Exception as e:
            logger.error(f"모터 {self.motor_id}: 명령 실행 실패 - {e}")
            return False
    
    async def _move_to_position(self, target: float, speed: float):
        """위치로 이동"""
        distance = abs(target - self.current_position)
        movement_time = distance / (self.max_velocity * speed)
        
        logger.info(f"모터 {self.motor_id}: {self.current_position:.2f} -> {target:.2f} "
              f"({movement_time:.2f}초)")
        
        # 이동 시뮬레이션
        steps = max(10, int(movement_time * 10))
        for i in range(steps + 1):
            progress = i / steps
            self.current_position = self.current_position + progress * (target - self.current_position)
            await asyncio.sleep(movement_time / steps)
        
        self.current_position = target
    
    async def get_status(self) -> Dict[str, Any]:
        """모터 상태 조회"""
        return {
            "motor_id": self.motor_id,
            "position": self.current_position,
            "velocity": self.current_velocity,
            "temperature": np.random.uniform(25, 45),  # Celsius
            "current_draw": np.random.uniform(0.1, 2.0),  # Amperes
            "error_status": "normal"
        }

class Gripper(ActuatorInterface):
    """그리퍼"""
    
    def __init__(self, gripper_id: str):
        self.gripper_id = gripper_id
        self.is_open = True
        self.grip_force = 0.0
        self.max_force = 50.0  # Newton
        self.object_held = None
        
    async def execute_command(self, command: ActuatorCommand) -> bool:
        """그리퍼 명령 실행"""
        cmd_type = command.command_type
        params = command.parameters
        
        try:
            if cmd_type == "open":
                await self._open_gripper()
                return True
                
            elif cmd_type == "close":
                force = params.get("force", 20.0)
                target_object = params.get("target", None)
                return await self._close_gripper(force, target_object)
                
            elif cmd_type == "set_force":
                force = params.get("force", 10.0)
                self.grip_force = np.clip(force, 0, self.max_force)
                logger.info(f"그리퍼 {self.gripper_id}: 그립 강도 설정 {self.grip_force:.1f}N")
                return True
                
            else:
                logger.warning(f"그리퍼 {self.gripper_id}: 알 수 없는 명령 {cmd_type}")
                return False
                
        except Exception as e:
            logger.error(f"그리퍼 {self.gripper_id}: 명령 실행 실패 - {e}")
            return False
    
    async def _open_gripper(self):
        """그리퍼 열기"""
        logger.info(f"그리퍼 {self.gripper_id}: 열기")
        await asyncio.sleep(1)  # 동작 시간
        self.is_open = True
        self.grip_force = 0.0
        if self.object_held:
            logger.info(f"객체 '{self.object_held}' 해제")
            self.object_held = None
    
    async def _close_gripper(self, force: float, target_object: str = None) -> bool:
        """그리퍼 닫기"""
        logger.info(f"그리퍼 {self.gripper_id}: 닫기 (강도: {force:.1f}N)")
        await asyncio.sleep(1)  # 동작 시간
        
        self.is_open = False
        self.grip_force = force
        
        # 객체 잡기 시뮬레이션 (80% 성공률)
        if target_object and np.random.random() < 0.8:
            self.object_held = target_object
            logger.info(f"객체 '{target_object}' 성공적으로 잡음")
            return True
        else:
            logger.warning("객체 잡기 실패")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """그리퍼 상태 조회"""
        return {
            "gripper_id": self.gripper_id,
            "is_open": self.is_open,
            "grip_force": self.grip_force,
            "object_held": self.object_held,
            "finger_position": 1.0 if self.is_open else 0.0
        }

class SensorFusion:
    """센서 융합"""
    
    def __init__(self):
        self.sensors: Dict[str, SensorInterface] = {}
        self.fusion_algorithms = {
            "object_tracking": self._fuse_object_tracking,
            "localization": self._fuse_localization,
            "contact_detection": self._fuse_contact_detection
        }
        
    def add_sensor(self, sensor_id: str, sensor: SensorInterface):
        """센서 추가"""
        self.sensors[sensor_id] = sensor
        
    async def fuse_sensor_data(self, fusion_type: str) -> Dict[str, Any]:
        """센서 데이터 융합"""
        if fusion_type not in self.fusion_algorithms:
            raise ValueError(f"알 수 없는 융합 타입: {fusion_type}")
        
        # 모든 센서에서 데이터 수집
        sensor_readings = {}
        for sensor_id, sensor in self.sensors.items():
            try:
                data = await sensor.read_data()
                sensor_readings[sensor_id] = data
            except Exception as e:
                logger.error(f"센서 {sensor_id} 읽기 실패: {e}")
        
        # 융합 알고리즘 적용
        return await self.fusion_algorithms[fusion_type](sensor_readings)
    
    async def _fuse_object_tracking(self, sensor_data: Dict[str, SensorData]) -> Dict[str, Any]:
        """객체 추적을 위한 센서 융합"""
        objects = []
        
        # 비전 센서에서 객체 감지 정보 추출
        for sensor_id, data in sensor_data.items():
            if data.sensor_type == "vision":
                vision_objects = data.data.get("objects_detected", [])
                for obj in vision_objects:
                    # 3D 위치 정보 추가
                    objects.append({
                        "id": f"obj_{len(objects)}",
                        "class": obj["class"],
                        "position": obj["position_3d"],
                        "confidence": obj["confidence"] * data.confidence,
                        "source": "vision"
                    })
        
        return {"tracked_objects": objects, "fusion_confidence": 0.85}
    
    async def _fuse_localization(self, sensor_data: Dict[str, SensorData]) -> Dict[str, Any]:
        """로봇 위치 추정을 위한 센서 융합"""
        position = np.array([0.0, 0.0, 0.0])
        orientation = np.array([0.0, 0.0, 0.0])
        confidence = 0.0
        
        # IMU 데이터에서 자세 정보 추출
        for sensor_id, data in sensor_data.items():
            if data.sensor_type == "imu":
                orientation = data.data.get("orientation", np.zeros(3))
                confidence = max(confidence, data.confidence)
        
        return {
            "position": position.tolist(),
            "orientation": orientation.tolist(),
            "confidence": confidence
        }
    
    async def _fuse_contact_detection(self, sensor_data: Dict[str, SensorData]) -> Dict[str, Any]:
        """접촉 감지를 위한 센서 융합"""
        contacts = []
        
        # 촉각 센서에서 접촉 정보 추출
        for sensor_id, data in sensor_data.items():
            if data.sensor_type == "tactile":
                if data.data.get("contact_detected", False):
                    contacts.append({
                        "sensor": sensor_id,
                        "pressure": data.data.get("pressure", 0),
                        "temperature": data.data.get("temperature", 20),
                        "confidence": data.confidence
                    })
        
        return {"contacts": contacts, "contact_detected": len(contacts) > 0}

class HardwareManager:
    """하드웨어 매니저 - HAL의 메인 클래스"""
    
    def __init__(self):
        self.sensors: Dict[str, SensorInterface] = {}
        self.actuators: Dict[str, ActuatorInterface] = {}
        self.sensor_fusion = SensorFusion()
        self.initialized = False
        
    async def initialize(self):
        """하드웨어 매니저 초기화"""
        logger.info("Hardware Manager 초기화 중...")
        
        # 센서 초기화
        await self._initialize_sensors()
        
        # 액추에이터 초기화
        await self._initialize_actuators()
        
        self.initialized = True
        logger.info("Hardware Manager 초기화 완료")
    
    async def _initialize_sensors(self):
        """센서 초기화"""
        # 비전 센서
        vision_sensor = VisionSensor("main_camera")
        self.sensors["main_camera"] = vision_sensor
        self.sensor_fusion.add_sensor("main_camera", vision_sensor)
        await vision_sensor.calibrate()
        
        # 촉각 센서
        tactile_sensor = TactileSensor("gripper_tactile")
        self.sensors["gripper_tactile"] = tactile_sensor
        self.sensor_fusion.add_sensor("gripper_tactile", tactile_sensor)
        await tactile_sensor.calibrate()
        
        # IMU 센서
        imu_sensor = IMUSensor("body_imu")
        self.sensors["body_imu"] = imu_sensor
        self.sensor_fusion.add_sensor("body_imu", imu_sensor)
        await imu_sensor.calibrate()
        
        logger.info("센서 초기화 완료")
    
    async def _initialize_actuators(self):
        """액추에이터 초기화"""
        # 관절 모터들
        joint_limits = [
            (-3.14, 3.14),  # 베이스 회전
            (-1.57, 1.57),  # 어깨 피치
            (-2.09, 2.09),  # 어깨 롤
            (-3.14, 3.14),  # 팔꿈치
            (-1.57, 1.57),  # 손목 피치
            (-2.09, 2.09)   # 손목 롤
        ]
        
        for i, limits in enumerate(joint_limits):
            motor = ServoMotor(f"joint_{i+1}", limits)
            self.actuators[f"joint_{i+1}"] = motor
        
        # 그리퍼
        gripper = Gripper("main_gripper")
        self.actuators["main_gripper"] = gripper
        
        logger.info("액추에이터 초기화 완료")
    
    async def get_sensor_data(self, sensor_id: str) -> Optional[SensorData]:
        """특정 센서 데이터 읽기"""
        if sensor_id not in self.sensors:
            logger.warning(f"센서 {sensor_id}를 찾을 수 없습니다.")
            return None
        
        return await self.sensors[sensor_id].read_data()
    
    async def get_fused_data(self, fusion_type: str) -> Dict[str, Any]:
        """융합된 센서 데이터 가져오기"""
        return await self.sensor_fusion.fuse_sensor_data(fusion_type)
    
    async def control_actuator(self, actuator_id: str, 
                             command_type: str, 
                             parameters: Dict[str, Any]) -> bool:
        """액추에이터 제어"""
        if actuator_id not in self.actuators:
            logger.warning(f"액추에이터 {actuator_id}를 찾을 수 없습니다.")
            return False
        
        command = ActuatorCommand(
            actuator_id=actuator_id,
            command_type=command_type,
            parameters=parameters,
            priority=1
        )
        
        return await self.actuators[actuator_id].execute_command(command)
    
    async def control_gripper(self, action: str, target: str = None) -> bool:
        """그리퍼 제어 (간편 인터페이스)"""
        params = {}
        if target:
            params["target"] = target
        if action == "close":
            params["force"] = 25.0  # 기본 그립 강도
        
        return await self.control_actuator("main_gripper", action, params)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """전체 시스템 상태 조회"""
        status = {
            "initialized": self.initialized,
            "sensors": {},
            "actuators": {}
        }
        
        # 센서 상태
        for sensor_id in self.sensors:
            status["sensors"][sensor_id] = {
                "type": self.sensors[sensor_id].__class__.__name__,
                "calibrated": getattr(self.sensors[sensor_id], "calibrated", False)
            }
        
        # 액추에이터 상태
        for actuator_id in self.actuators:
            actuator_status = await self.actuators[actuator_id].get_status()
            status["actuators"][actuator_id] = actuator_status
        
        return status

# 테스트 코드
if __name__ == "__main__":
    async def test():
        hw_manager = HardwareManager()
        await hw_manager.initialize()
        
        # 센서 데이터 테스트
        vision_data = await hw_manager.get_sensor_data("main_camera")
        logger.info(f"비전 데이터: {len(vision_data.data['objects_detected'])}개 객체 감지")
        
        # 융합 데이터 테스트
        object_tracking = await hw_manager.get_fused_data("object_tracking")
        logger.info(f"융합 데이터: {len(object_tracking['tracked_objects'])}개 객체 추적")
        
        # 액추에이터 제어 테스트
        success = await hw_manager.control_actuator(
            "joint_1", "move_to_position", {"position": 1.0, "speed": 0.5}
        )
        logger.info(f"관절 제어: {'성공' if success else '실패'}")
        
        # 그리퍼 테스트
        gripper_success = await hw_manager.control_gripper("close", "test_object")
        logger.info(f"그리퍼 제어: {'성공' if gripper_success else '실패'}")
        
        # 시스템 상태
        system_status = await hw_manager.get_system_status()
        logger.info(f"시스템 상태: {len(system_status['sensors'])}개 센서, "
              f"{len(system_status['actuators'])}개 액추에이터")
    
    asyncio.run(test())
