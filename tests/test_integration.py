"""
Physical AI System 통합 테스트

전체 시스템의 통합 테스트를 수행합니다.
"""

import asyncio
import pytest
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import PhysicalAI
from foundation_model.slm_foundation import SLMFoundation
from developmental_learning.dev_engine import DevelopmentalEngine
from ai_agent_execution.agent_executor import AgentExecutor
from hardware_abstraction.hal_manager import HardwareManager
from utils.common import ConfigManager, PerformanceMetrics

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPhysicalAISystem:
    """Physical AI 시스템 통합 테스트 클래스"""
    
    @pytest.fixture(autouse=True)
    async def setup_system(self):
        """테스트 시스템 설정"""
        self.config = ConfigManager.get_default_config()
        self.performance_metrics = PerformanceMetrics()
        
        # 시스템 초기화
        self.physical_ai = PhysicalAI("configs/default.yaml")
        await self.physical_ai.initialize()
        
        yield
        
        # 정리
        if hasattr(self, 'physical_ai'):
            del self.physical_ai
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """시스템 초기화 테스트"""
        logger.info("=== 시스템 초기화 테스트 ===")
        
        assert self.physical_ai is not None
        assert self.physical_ai.system_ready == True
        assert self.physical_ai.initialization_time is not None
        assert self.physical_ai.initialization_time > 0
        
        logger.info(f"✅ 시스템 초기화 성공: {self.physical_ai.initialization_time:.2f}초")
    
    @pytest.mark.asyncio
    async def test_foundation_model(self):
        """Foundation Model 테스트"""
        logger.info("=== Foundation Model 테스트 ===")
        
        # PHI-3.5 모델 테스트
        test_missions = [
            "Pick up the red cup and place it on the table",
            "Move to position [1, 0, 0.5]",
            "Explore the surrounding area"
        ]
        
        for mission in test_missions:
            logger.info(f"미션 테스트: {mission}")
            
            # 미션 해석
            task_plan = await self.physical_ai.slm_foundation.interpret_mission(mission)
            
            assert task_plan is not None
            assert task_plan.mission == mission
            assert len(task_plan.subtasks) > 0
            assert task_plan.expected_duration > 0
            
            logger.info(f"✅ 미션 해석 성공: {len(task_plan.subtasks)}개 서브태스크")
    
    @pytest.mark.asyncio
    async def test_developmental_learning(self):
        """발달적 학습 테스트"""
        logger.info("=== 발달적 학습 테스트 ===")
        
        # 스킬 분석 테스트
        fake_task_plan = type('TaskPlan', (), {
            'subtasks': [
                {'action': 'move_to', 'difficulty': 2},
                {'action': 'grasp', 'difficulty': 5}
            ]
        })()
        
        required_skills = await self.physical_ai.dev_engine.analyze_required_skills(fake_task_plan)
        
        assert required_skills is not None
        assert isinstance(required_skills, dict)
        
        logger.info(f"✅ 스킬 분석 성공: {len(required_skills)}개 스킬 필요")
        
        # 학습 진행도 확인
        progress = await self.physical_ai.dev_engine.get_learning_progress()
        assert progress is not None
        
        logger.info("✅ 발달적 학습 테스트 완료")
    
    @pytest.mark.asyncio
    async def test_agent_execution(self):
        """Agent Execution 테스트"""
        logger.info("=== Agent Execution 테스트 ===")
        
        # 가짜 태스크 계획 생성
        fake_task_plan = type('TaskPlan', (), {
            'mission': 'Test mission',
            'subtasks': [
                {
                    'action': 'move_to',
                    'target': 'test_position',
                    'estimated_duration': 5.0
                }
            ],
            'constraints': {'max_velocity': 2.0},
            'expected_duration': 5.0,
            'success_criteria': ['position_reached']
        })()
        
        # 가짜 스킬 상태
        skill_states = {
            'basic_movement': {'ready': True, 'success_rate': 0.9}
        }
        
        # 실행 테스트
        result = await self.physical_ai.agent_executor.execute(fake_task_plan, skill_states)
        
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'execution_time')
        assert hasattr(result, 'actions_performed')
        
        logger.info(f"✅ Agent Execution 성공: {result.success}")
    
    @pytest.mark.asyncio
    async def test_hardware_abstraction(self):
        """하드웨어 추상화 테스트"""
        logger.info("=== 하드웨어 추상화 테스트 ===")
        
        # 센서 데이터 읽기 테스트
        vision_data = await self.physical_ai.hw_manager.get_sensor_data("main_camera")
        assert vision_data is not None
        assert hasattr(vision_data, 'timestamp')
        assert hasattr(vision_data, 'data')
        
        # 시스템 상태 확인
        system_status = await self.physical_ai.hw_manager.get_system_status()
        assert system_status is not None
        assert 'sensors' in system_status
        assert 'actuators' in system_status
        
        logger.info(f"✅ 하드웨어 추상화 성공: {len(system_status['sensors'])}개 센서")
    
    @pytest.mark.asyncio
    async def test_full_mission_execution(self):
        """전체 미션 실행 테스트"""
        logger.info("=== 전체 미션 실행 테스트 ===")
        
        test_mission = "Move to position [0.5, 0.5, 0.5] and return to home"
        
        self.performance_metrics.start_timer("full_mission")
        
        # 미션 실행
        result = await self.physical_ai.execute_mission(test_mission)
        
        execution_time = self.performance_metrics.end_timer("full_mission")
        
        assert result is not None
        assert 'success' in result
        assert 'mission' in result
        assert result['mission'] == test_mission
        
        logger.info(f"✅ 전체 미션 실행 성공: {execution_time:.2f}초")
        logger.info(f"   성공: {result['success']}")
        logger.info(f"   학습 가치: {result.get('learning_value', 0):.3f}")
    
    @pytest.mark.asyncio
    async def test_learning_insights(self):
        """학습 인사이트 테스트"""
        logger.info("=== 학습 인사이트 테스트 ===")
        
        # 학습 인사이트 조회
        insights = await self.physical_ai.get_learning_insights()
        
        assert insights is not None
        assert 'developmental_learning' in insights
        assert 'llm_learning' in insights
        
        logger.info("✅ 학습 인사이트 조회 성공")
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """오류 처리 테스트"""
        logger.info("=== 오류 처리 테스트 ===")
        
        # 잘못된 미션 테스트
        invalid_mission = ""
        
        result = await self.physical_ai.execute_mission(invalid_mission)
        
        # 오류가 적절히 처리되어야 함
        assert result is not None
        assert 'success' in result
        
        logger.info("✅ 오류 처리 테스트 완료")
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self):
        """성능 벤치마크 테스트"""
        logger.info("=== 성능 벤치마크 테스트 ===")
        
        test_missions = [
            "Simple movement test",
            "Object manipulation test",
            "Complex task test"
        ]
        
        total_time = 0
        success_count = 0
        
        for i, mission in enumerate(test_missions):
            self.performance_metrics.start_timer(f"mission_{i}")
            
            result = await self.physical_ai.execute_mission(mission)
            
            mission_time = self.performance_metrics.end_timer(f"mission_{i}")
            total_time += mission_time
            
            if result.get('success', False):
                success_count += 1
            
            logger.info(f"미션 {i+1}: {mission_time:.2f}초, 성공: {result.get('success', False)}")
        
        success_rate = success_count / len(test_missions)
        avg_time = total_time / len(test_missions)
        
        logger.info(f"✅ 성능 벤치마크 완료:")
        logger.info(f"   평균 실행 시간: {avg_time:.2f}초")
        logger.info(f"   성공률: {success_rate:.2%}")
        
        # 성능 기준 확인
        assert avg_time < 30.0  # 평균 30초 이내
        assert success_rate > 0.5  # 50% 이상 성공률

class TestIndividualComponents:
    """개별 컴포넌트 테스트"""
    
    @pytest.mark.asyncio
    async def test_slm_foundation_standalone(self):
        """SLM Foundation 독립 테스트"""
        logger.info("=== SLM Foundation 독립 테스트 ===")
        
        foundation = SLMFoundation(
            model_type="phi35",
            model_name="microsoft/Phi-3.5-mini-instruct",
            device="cpu"  # 테스트용 CPU 사용
        )
        
        await foundation.initialize()
        
        # 미션 해석 테스트
        mission = "Test mission for standalone foundation"
        task_plan = await foundation.interpret_mission(mission)
        
        assert task_plan is not None
        assert task_plan.mission == mission
        
        logger.info("✅ SLM Foundation 독립 테스트 완료")
    
    @pytest.mark.asyncio
    async def test_developmental_engine_standalone(self):
        """Developmental Engine 독립 테스트"""
        logger.info("=== Developmental Engine 독립 테스트 ===")
        
        dev_engine = DevelopmentalEngine()
        await dev_engine.initialize()
        
        # 스킬 상태 확인
        skills = dev_engine.skill_engine.skills_db
        assert len(skills) > 0
        
        # 기본 스킬 확인
        assert "basic_movement" in skills
        assert "object_recognition" in skills
        
        logger.info("✅ Developmental Engine 독립 테스트 완료")
    
    @pytest.mark.asyncio
    async def test_agent_executor_standalone(self):
        """Agent Executor 독립 테스트"""
        logger.info("=== Agent Executor 독립 테스트 ===")
        
        executor = AgentExecutor()
        
        # 하드웨어 매니저 없이 초기화
        await executor.initialize(None)
        
        # 기본 동작 테스트
        motion_controller = executor.motion_controller
        assert motion_controller is not None
        
        # 안전 모니터 테스트
        safety_monitor = executor.safety_monitor
        assert safety_monitor is not None
        
        logger.info("✅ Agent Executor 독립 테스트 완료")
    
    @pytest.mark.asyncio
    async def test_hardware_manager_standalone(self):
        """Hardware Manager 독립 테스트"""
        logger.info("=== Hardware Manager 독립 테스트 ===")
        
        hw_manager = HardwareManager()
        await hw_manager.initialize()
        
        # 센서 초기화 확인
        assert len(hw_manager.sensors) > 0
        
        # 액추에이터 초기화 확인
        assert len(hw_manager.actuators) > 0
        
        # 센서 데이터 읽기 테스트
        sensor_data = await hw_manager.get_sensor_data("main_camera")
        assert sensor_data is not None
        
        logger.info("✅ Hardware Manager 독립 테스트 완료")

def run_integration_tests():
    """통합 테스트 실행"""
    logger.info("🚀 Physical AI System 통합 테스트 시작")
    
    # pytest 실행
    import pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])

if __name__ == "__main__":
    run_integration_tests()
