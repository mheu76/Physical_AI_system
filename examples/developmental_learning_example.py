"""
발달적 학습 예제 - AI가 스스로 학습하는 과정 시연

이 예제는 AI가 아기처럼 점진적으로 스킬을 습득하는 
발달적 학습 과정을 보여줍니다.
"""

import asyncio
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import PhysicalAI
from utils.common import setup_logging, load_config, format_duration

async def developmental_learning_example():
    """발달적 학습 예제"""
    print("=== 발달적 학습 예제 ===\n")
    print("AI가 아기처럼 자라나는 과정을 관찰해보세요...\n")
    
    # 설정 및 시스템 초기화
    config = load_config("configs/default.yaml")
    setup_logging(config)
    
    physical_ai = PhysicalAI("configs/default.yaml")
    await physical_ai.initialize()
    
    # 학습 추적을 위한 데이터
    learning_data = {
        "iterations": [],
        "skill_progress": {},
        "learning_stages": []
    }
    
    # 초기 스킬 상태 기록
    skills_db = physical_ai.dev_engine.skill_engine.skills_db
    for skill_name in skills_db:
        learning_data["skill_progress"][skill_name] = []
    
    print("=== 발달 단계별 학습 과정 ===\n")
    
    # 30번의 학습 사이클 실행
    for iteration in range(1, 31):
        print(f"학습 사이클 {iteration}/30")
        
        # 자율 탐색 학습
        await physical_ai.dev_engine.autonomous_exploration()
        
        # 현재 스킬 상태 기록
        learning_data["iterations"].append(iteration)
        current_stage = physical_ai.dev_engine.skill_engine.curriculum.current_stage
        learning_data["learning_stages"].append(current_stage)
        
        for skill_name, skill in skills_db.items():
            learning_data["skill_progress"][skill_name].append(skill.success_rate)
        
        # 진행 상황 출력
        if iteration % 5 == 0:
            print(f"  현재 학습 단계: {current_stage}")
            print("  스킬 성공률:")
            
            for skill_name, skill in skills_db.items():
                bar = "█" * int(skill.success_rate * 20)
                spaces = " " * (20 - int(skill.success_rate * 20))
                print(f"    {skill_name:20}: [{bar}{spaces}] {skill.success_rate:.2f}")
            print()
        
        await asyncio.sleep(0.1)  # 짧은 대기
    
    # 학습 결과 분석
    print("\n=== 학습 결과 분석 ===")
    
    for skill_name, progress in learning_data["skill_progress"].items():
        initial_rate = progress[0]
        final_rate = progress[-1]
        improvement = final_rate - initial_rate
        
        print(f"{skill_name}:")
        print(f"  초기 성공률: {initial_rate:.2f}")
        print(f"  최종 성공률: {final_rate:.2f}")
        print(f"  개선도: {improvement:.2f} ({'+'  if improvement > 0 else ''}{improvement:.2f})")
        print()
    
    # 학습 그래프 생성
    await create_learning_graph(learning_data)
    
    # 실제 미션 테스트
    print("=== 학습된 AI 성능 테스트 ===\n")
    
    test_missions = [
        "Pick up the red cup and place it on the table",
        "Explore the room and identify all objects",
        "Move to [2, 1, 0.8] and perform precise manipulation"
    ]
    
    for i, mission in enumerate(test_missions, 1):
        print(f"테스트 {i}: {mission}")
        
        result = await physical_ai.execute_mission(mission)
        
        if result.success:
            print(f"  ✓ 성공! 실행시간: {format_duration(result.execution_time)}")
            print(f"    성능 지표:")
            for metric, value in result.performance_metrics.items():
                print(f"      {metric}: {value:.3f}")
        else:
            print(f"  ✗ 실패 (오류 {len(result.errors)}개)")
        
        print()
    
    print("=== 발달적 학습 예제 완료 ===")

async def create_learning_graph(learning_data):
    """학습 진행 그래프 생성"""
    try:
        plt.figure(figsize=(12, 8))
        
        # 서브플롯 1: 스킬별 성공률 변화
        plt.subplot(2, 1, 1)
        
        for skill_name, progress in learning_data["skill_progress"].items():
            plt.plot(learning_data["iterations"], progress, 
                    label=skill_name, marker='o', markersize=2)
        
        plt.xlabel('학습 사이클')
        plt.ylabel('성공률')
        plt.title('스킬별 학습 진행도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 2: 학습 단계 변화
        plt.subplot(2, 1, 2)
        plt.plot(learning_data["iterations"], learning_data["learning_stages"], 
                'g-', linewidth=2, marker='s', markersize=4)
        plt.xlabel('학습 사이클')
        plt.ylabel('학습 단계')
        plt.title('커리큘럼 학습 단계 진행')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learning_progress.png', dpi=150, bbox_inches='tight')
        print("학습 진행 그래프가 'learning_progress.png'로 저장되었습니다.")
        
    except ImportError:
        print("matplotlib가 설치되지 않아 그래프를 생성할 수 없습니다.")
        print("pip install matplotlib 로 설치해주세요.")
    except Exception as e:
        print(f"그래프 생성 중 오류: {e}")

if __name__ == "__main__":
    asyncio.run(developmental_learning_example())
