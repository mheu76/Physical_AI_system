"""
기본 예제 - Physical AI 시스템 기본 사용법

이 예제는 Physical AI 시스템의 기본적인 사용법을 보여줍니다.
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import PhysicalAI
from utils.common import setup_logging, load_config

async def basic_example():
    """기본 예제 실행"""
    print("=== Physical AI 기본 예제 ===\n")
    
    # 설정 로드
    config = load_config("configs/default.yaml")
    setup_logging(config)
    
    # Physical AI 시스템 생성
    physical_ai = PhysicalAI("configs/default.yaml")
    
    print("1. 시스템 초기화 중...")
    await physical_ai.initialize()
    print("   ✓ 초기화 완료\n")
    
    # 예제 미션들
    missions = [
        "Move to position [1, 0, 0.5]",
        "Explore the surrounding area", 
        "Pick up the blue object",
        "Place the object on the table"
    ]
    
    for i, mission in enumerate(missions, 1):
        print(f"{i}. 미션 실행: {mission}")
        
        try:
            result = await physical_ai.execute_mission(mission)
            
            if result.success:
                print(f"   ✓ 성공! ({result.execution_time:.2f}초)")
                print(f"     수행 동작: {len(result.actions_performed)}개")
                print(f"     학습 가치: {result.learning_value:.2f}")
            else:
                print(f"   ✗ 실패 ({len(result.errors)}개 오류)")
                for error in result.errors:
                    print(f"     - {error}")
                    
        except Exception as e:
            print(f"   ✗ 예외 발생: {e}")
        
        print()  # 빈 줄
        await asyncio.sleep(1)  # 미션 간 대기
    
    print("=== 기본 예제 완료 ===")

if __name__ == "__main__":
    asyncio.run(basic_example())
