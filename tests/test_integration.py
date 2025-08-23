"""
통합 테스트 - 전체 시스템 테스트

Physical AI 시스템의 모든 컴포넌트가 
함께 동작하는지 테스트합니다.
"""

import asyncio
import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import PhysicalAI
from utils.common import load_config

@pytest.mark.asyncio
async def test_full_system_initialization():
    """전체 시스템 초기화 테스트"""
    config_path = "configs/default.yaml"
    
    try:
        physical_ai = PhysicalAI(config_path)
        await physical_ai.initialize()
        
        # 각 컴포넌트가 제대로 초기화되었는지 확인
        assert physical_ai.slm_foundation is not None
        assert physical_ai.dev_engine is not None
        assert physical_ai.agent_executor is not None
        assert physical_ai.hw_manager is not None
        
        print("✓ 전체 시스템 초기화 성공")
        
    except Exception as e:
        pytest.fail(f"시스템 초기화 실패: {e}")

@pytest.mark.asyncio
async def test_simple_mission_execution():
    """간단한 미션 실행 테스트"""
    config_path = "configs/default.yaml"
    
    try:
        physical_ai = PhysicalAI(config_path)
        await physical_ai.initialize()
        
        # 간단한 미션 실행
        mission = "Move to position [1, 0, 0.5] and explore the area"
        result = await physical_ai.execute_mission(mission)
        
        assert result is not None
        assert "success" in result.__dict__
        
        print(f"✓ 미션 실행 테스트: {'성공' if result.success else '실패'}")
        print(f"  실행 시간: {result.execution_time:.2f}초")
        print(f"  수행된 동작: {len(result.actions_performed)}개")
        
    except Exception as e:
        pytest.fail(f"미션 실행 실패: {e}")

@pytest.mark.asyncio  
async def test_pick_and_place_mission():
    """Pick and Place 미션 테스트"""
    config_path = "configs/default.yaml"
    
    try:
        physical_ai = PhysicalAI(config_path)
        await physical_ai.initialize()
        
        # Pick and Place 미션
        mission = "Pick up the red cup and place it on the table"
        result = await physical_ai.execute_mission(mission)
        
        assert result is not None
        
        # 예상되는 서브태스크들이 실행되었는지 확인
        expected_actions = ["move_to", "grasp", "place"]
        performed_actions = [action["subtask"]["action"] 
                           for action in result.actions_performed]
        
        for expected in expected_actions:
            assert any(expected in action for action in performed_actions), \
                f"예상 동작 '{expected}'이 실행되지 않았습니다"
        
        print(f"✓ Pick and Place 테스트: {'성공' if result.success else '실패'}")
        
    except Exception as e:
        pytest.fail(f"Pick and Place 실행 실패: {e}")

@pytest.mark.asyncio
async def test_developmental_learning():
    """발달적 학습 테스트"""
    config_path = "configs/default.yaml"
    
    try:
        physical_ai = PhysicalAI(config_path)
        await physical_ai.initialize()
        
        # 초기 스킬 상태 확인
        initial_skills = physical_ai.dev_engine.skill_engine.skills_db.copy()
        
        # 몇 번의 자율 학습 실행
        for _ in range(5):
            await physical_ai.dev_engine.autonomous_exploration()
        
        # 스킬이 개선되었는지 확인
        final_skills = physical_ai.dev_engine.skill_engine.skills_db
        
        improvements = 0
        for skill_name, initial_skill in initial_skills.items():
            final_skill = final_skills[skill_name]
            if final_skill.success_rate > initial_skill.success_rate:
                improvements += 1
        
        assert improvements > 0, "어떤 스킬도 개선되지 않았습니다"
        
        print(f"✓ 발달적 학습 테스트: {improvements}개 스킬 개선")
        
    except Exception as e:
        pytest.fail(f"발달적 학습 실패: {e}")

@pytest.mark.asyncio
async def test_safety_system():
    """안전 시스템 테스트"""
    config_path = "configs/default.yaml"
    
    try:
        physical_ai = PhysicalAI(config_path)
        await physical_ai.initialize()
        
        # 안전 모니터링 테스트
        safety_status = await physical_ai.agent_executor.safety_monitor.monitor_safety()
        
        assert "safe" in safety_status
        assert "warnings" in safety_status
        assert "violations" in safety_status
        
        print(f"✓ 안전 시스템 테스트: 안전 상태 = {safety_status['safe']}")
        
    except Exception as e:
        pytest.fail(f"안전 시스템 테스트 실패: {e}")

@pytest.mark.asyncio
async def test_hardware_abstraction():
    """하드웨어 추상화 테스트"""
    config_path = "configs/default.yaml"
    
    try:
        physical_ai = PhysicalAI(config_path)
        await physical_ai.initialize()
        
        # 센서 데이터 읽기 테스트
        vision_data = await physical_ai.hw_manager.get_sensor_data("main_camera")
        assert vision_data is not None
        assert vision_data.sensor_type == "vision"
        
        # 융합 데이터 테스트
        fused_data = await physical_ai.hw_manager.get_fused_data("object_tracking")
        assert "tracked_objects" in fused_data
        
        # 액추에이터 제어 테스트
        success = await physical_ai.hw_manager.control_gripper("open")
        assert success is True or success is False  # boolean 반환 확인
        
        print("✓ 하드웨어 추상화 테스트 성공")
        
    except Exception as e:
        pytest.fail(f"하드웨어 추상화 테스트 실패: {e}")

def run_integration_tests():
    """통합 테스트 실행"""
    print("=== Physical AI 시스템 통합 테스트 ===\n")
    
    # pytest로 모든 테스트 실행
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--asyncio-mode=auto"
    ])

if __name__ == "__main__":
    run_integration_tests()
