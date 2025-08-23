"""
물리 시뮬레이션 환경

PyBullet을 사용한 물리 시뮬레이션 환경으로
실제 하드웨어 없이도 Physical AI를 테스트할 수 있습니다.
"""

import pybullet as p
import pybullet_data
import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import time

@dataclass
class SimulatedObject:
    """시뮬레이션 객체"""
    object_id: int
    name: str
    position: np.ndarray
    orientation: np.ndarray
    mass: float
    
@dataclass
class SimulationState:
    """시뮬레이션 상태"""
    time: float
    objects: Dict[str, SimulatedObject]
    robot_state: Dict[str, Any]
    sensor_data: Dict[str, Any]

class PhysicsSimulation:
    """물리 시뮬레이션 엔진"""
    
    def __init__(self, gui_mode: bool = False):
        self.gui_mode = gui_mode
        self.physics_client = None
        self.robot_id = None
        self.objects = {}
        self.running = False
        self.time_step = 1./240.  # 240Hz
        
    async def initialize(self):
        """시뮬레이션 초기화"""
        print("물리 시뮬레이션 초기화 중...")
        
        # PyBullet 연결
        if self.gui_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # 물리 설정
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # 환경 설정
        await self._setup_environment()
        
        # 로봇 로드
        await self._load_robot()
        
        self.running = True
        print("물리 시뮬레이션 초기화 완료")
        
    async def _setup_environment(self):
        """환경 설정"""
        # 바닥 평면
        plane_id = p.loadURDF("plane.urdf")
        
        # 테이블 생성
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.8, 0.05])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.8, 0.05], 
                                          rgbaColor=[0.6, 0.4, 0.2, 1.0])
        table_id = p.createMultiBody(baseMass=10, 
                                   baseCollisionShapeIndex=table_collision,
                                   baseVisualShapeIndex=table_visual,
                                   basePosition=[1.5, 0, 0.4])
        
        self.objects["table"] = SimulatedObject(
            object_id=table_id,
            name="table",
            position=np.array([1.5, 0, 0.4]),
            orientation=np.array([0, 0, 0, 1]),
            mass=10.0
        )
        
        # 컵 생성
        cup_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.08)
        cup_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.08,
                                        rgbaColor=[0.8, 0.2, 0.2, 1.0])
        cup_id = p.createMultiBody(baseMass=0.1,
                                 baseCollisionShapeIndex=cup_collision,
                                 baseVisualShapeIndex=cup_visual,
                                 basePosition=[1.2, 0.2, 0.5])
        
        self.objects["red_cup"] = SimulatedObject(
            object_id=cup_id,
            name="red_cup", 
            position=np.array([1.2, 0.2, 0.5]),
            orientation=np.array([0, 0, 0, 1]),
            mass=0.1
        )
        
    async def _load_robot(self):
        """로봇 모델 로드"""
        try:
            # 간단한 로봇 팔 모델 (UR5 스타일)
            self.robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0])
            
            # 실제 프로젝트에서는 커스텀 URDF 파일 사용
            # self.robot_id = p.loadURDF("models/robot_arm.urdf", basePosition=[0, 0, 0])
            
        except:
            # URDF 파일이 없으면 간단한 박스 로봇 생성
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2])
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2],
                                             rgbaColor=[0.2, 0.2, 0.8, 1.0])
            self.robot_id = p.createMultiBody(baseMass=1.0,
                                            baseCollisionShapeIndex=collision_shape,
                                            baseVisualShapeIndex=visual_shape,
                                            basePosition=[0, 0, 0.2])
        
    async def step(self):
        """시뮬레이션 한 스텝 실행"""
        if self.running:
            p.stepSimulation()
            await asyncio.sleep(self.time_step)
    
    async def run_simulation(self, duration: float = None):
        """시뮬레이션 실행"""
        start_time = time.time()
        
        while self.running:
            await self.step()
            
            if duration and (time.time() - start_time) >= duration:
                break
                
    def stop(self):
        """시뮬레이션 중지"""
        self.running = False
        if self.physics_client is not None:
            p.disconnect()
        print("물리 시뮬레이션 중지됨")
        
    async def get_robot_state(self) -> Dict[str, Any]:
        """로봇 상태 조회"""
        if self.robot_id is None:
            return {}
            
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        velocity = p.getBaseVelocity(self.robot_id)
        
        return {
            "position": np.array(position),
            "orientation": np.array(orientation), 
            "linear_velocity": np.array(velocity[0]),
            "angular_velocity": np.array(velocity[1])
        }
    
    async def move_robot(self, target_position: np.ndarray, 
                        target_orientation: np.ndarray = None):
        """로봇 이동"""
        if self.robot_id is None:
            return False
            
        if target_orientation is None:
            target_orientation = [0, 0, 0, 1]
            
        # 간단한 위치 제어 (실제로는 더 정교한 제어 필요)
        p.resetBasePositionAndOrientation(self.robot_id, 
                                        target_position.tolist(),
                                        target_orientation)
        return True
    
    async def simulate_grasp(self, object_name: str) -> bool:
        """잡기 동작 시뮬레이션"""
        if object_name not in self.objects:
            return False
            
        obj = self.objects[object_name]
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        obj_pos = obj.position
        
        # 거리 체크
        distance = np.linalg.norm(np.array(robot_pos) - obj_pos)
        if distance > 0.3:  # 너무 멀면 실패
            return False
            
        # 80% 확률로 성공
        success = np.random.random() < 0.8
        
        if success:
            # 객체를 로봇에 붙이기 (constraint 생성)
            constraint_id = p.createConstraint(
                self.robot_id, -1, obj.object_id, -1,
                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.1], [0, 0, 0]
            )
            obj.grasped = True
            obj.constraint_id = constraint_id
            
        return success
    
    async def simulate_release(self, object_name: str) -> bool:
        """놓기 동작 시뮬레이션"""
        if object_name not in self.objects:
            return False
            
        obj = self.objects[object_name]
        if hasattr(obj, 'constraint_id'):
            p.removeConstraint(obj.constraint_id)
            del obj.constraint_id
            obj.grasped = False
            return True
            
        return False
    
    async def get_camera_data(self) -> Dict[str, Any]:
        """카메라 데이터 시뮬레이션"""
        if self.robot_id is None:
            return {}
            
        # 로봇 위치에서 카메라 뷰
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # 카메라 매트릭스 계산
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[robot_pos[0], robot_pos[1], robot_pos[2] + 0.3],
            cameraTargetPosition=[robot_pos[0] + 1, robot_pos[1], robot_pos[2]],
            cameraUpVector=[0, 0, 1]
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=5.0
        )
        
        # 이미지 렌더링
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            640, 480, view_matrix, projection_matrix
        )
        
        # 객체 감지 시뮬레이션
        detected_objects = []
        for name, obj in self.objects.items():
            # 간단한 거리 기반 감지
            distance = np.linalg.norm(np.array(robot_pos) - obj.position)
            if distance < 3.0:  # 3m 이내 객체만 감지
                detected_objects.append({
                    "class": name,
                    "position_3d": obj.position.tolist(),
                    "confidence": max(0.5, 1.0 - distance/3.0),
                    "bbox": [100, 100, 50, 50]  # 더미 bounding box
                })
        
        return {
            "rgb": rgb_img,
            "depth": depth_img,
            "segmentation": seg_img,
            "objects_detected": detected_objects
        }
    
    async def get_simulation_state(self) -> SimulationState:
        """전체 시뮬레이션 상태 조회"""
        robot_state = await self.get_robot_state()
        sensor_data = await self.get_camera_data()
        
        return SimulationState(
            time=time.time(),
            objects=self.objects.copy(),
            robot_state=robot_state,
            sensor_data=sensor_data
        )

class SimulationManager:
    """시뮬레이션 매니저"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sim = None
        self.running = False
        
    async def initialize(self, gui_mode: bool = False):
        """시뮬레이션 매니저 초기화"""
        print("시뮬레이션 매니저 초기화 중...")
        
        self.sim = PhysicsSimulation(gui_mode=gui_mode)
        await self.sim.initialize()
        
        # 백그라운드에서 시뮬레이션 실행
        self.running = True
        asyncio.create_task(self._simulation_loop())
        
        print("시뮬레이션 매니저 초기화 완료")
        
    async def _simulation_loop(self):
        """시뮬레이션 루프"""
        while self.running and self.sim.running:
            await self.sim.step()
            
    def stop(self):
        """시뮬레이션 매니저 중지"""
        self.running = False
        if self.sim:
            self.sim.stop()
    
    async def execute_robot_action(self, action: str, parameters: Dict[str, Any]) -> bool:
        """로봇 동작 실행"""
        if not self.sim:
            return False
            
        if action == "move_to":
            target_pos = np.array(parameters.get("position", [0, 0, 0]))
            return await self.sim.move_robot(target_pos)
            
        elif action == "grasp":
            target_object = parameters.get("target", "")
            return await self.sim.simulate_grasp(target_object)
            
        elif action == "release":
            target_object = parameters.get("target", "")
            return await self.sim.simulate_release(target_object)
            
        return False
    
    async def get_sensor_data(self) -> Dict[str, Any]:
        """센서 데이터 조회"""
        if not self.sim:
            return {}
            
        return await self.sim.get_camera_data()

# 사용 예제
if __name__ == "__main__":
    async def simulation_example():
        print("=== 물리 시뮬레이션 예제 ===")
        
        config = {"simulation": {"gui_mode": True}}
        sim_manager = SimulationManager(config)
        
        try:
            await sim_manager.initialize(gui_mode=True)
            
            # 몇 가지 동작 테스트
            print("로봇을 컵 근처로 이동...")
            await sim_manager.execute_robot_action("move_to", {"position": [1.2, 0.2, 0.3]})
            await asyncio.sleep(2)
            
            print("컵 잡기 시도...")
            success = await sim_manager.execute_robot_action("grasp", {"target": "red_cup"})
            print(f"잡기 {'성공' if success else '실패'}")
            await asyncio.sleep(2)
            
            if success:
                print("컵을 테이블 다른 위치로 이동...")
                await sim_manager.execute_robot_action("move_to", {"position": [1.8, -0.2, 0.5]})
                await asyncio.sleep(2)
                
                print("컵 놓기...")
                await sim_manager.execute_robot_action("release", {"target": "red_cup"})
                await asyncio.sleep(2)
            
            print("센서 데이터 조회...")
            sensor_data = await sim_manager.get_sensor_data()
            print(f"감지된 객체 수: {len(sensor_data.get('objects_detected', []))}")
            
            # 5초 더 실행
            await asyncio.sleep(5)
            
        except KeyboardInterrupt:
            print("사용자에 의해 중단됨")
        finally:
            sim_manager.stop()
            
    asyncio.run(simulation_example())
