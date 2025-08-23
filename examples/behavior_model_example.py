"""
행동모델 대화형 정의 시스템 사용 예제

PHI-3.5와 대화하면서 행동모델을 정의하는 방법을 보여줍니다.
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from behavior_model_dialog import BehaviorModelDialog

async def behavior_model_example():
    """행동모델 정의 예제"""
    print("🎯 행동모델 대화형 정의 시스템 예제")
    print("=" * 50)
    
    # 시스템 초기화
    dialog_system = BehaviorModelDialog()
    await dialog_system.initialize()
    
    print("\n💡 예제 대화 시나리오:")
    print("1. 커피 만들기 행동모델 정의")
    print("2. 청소 행동모델 정의")
    print("3. 요리 행동모델 정의")
    print("4. 모델 테스트 및 수정")
    
    # 예제 대화 실행
    await run_example_dialogs(dialog_system)
    
    print("\n✅ 예제 완료!")
    print("이제 실제로 대화형 시스템을 실행해보세요:")
    print("  python behavior_model_dialog.py")
    print("  또는")
    print("  python behavior_model_gui.py")

async def run_example_dialogs(dialog_system):
    """예제 대화 실행"""
    
    # 예제 1: 커피 만들기 모델
    print("\n" + "="*30)
    print("예제 1: 커피 만들기 행동모델")
    print("="*30)
    
    user_input = "커피를 만드는 행동모델을 정의해주세요. 커피머신에 접근해서 원두를 넣고, 물을 부어서 커피를 추출하는 과정이 필요해요."
    
    print(f"사용자: {user_input}")
    response = await dialog_system._process_user_input(user_input)
    print(f"PHI-3.5: {response}")
    
    # 예제 2: 청소 모델
    print("\n" + "="*30)
    print("예제 2: 청소 행동모델")
    print("="*30)
    
    user_input = "방을 청소하는 행동모델을 만들어주세요. 먼저 쓰레기를 수집하고, 바닥을 쓸고, 물걸레로 닦는 과정이 필요해요."
    
    print(f"사용자: {user_input}")
    response = await dialog_system._process_user_input(user_input)
    print(f"PHI-3.5: {response}")
    
    # 예제 3: 요리 모델
    print("\n" + "="*30)
    print("예제 3: 요리 행동모델")
    print("="*30)
    
    user_input = "간단한 요리를 하는 행동모델을 정의해주세요. 재료를 준비하고, 조리하고, 접시에 담는 과정이 필요해요."
    
    print(f"사용자: {user_input}")
    response = await dialog_system._process_user_input(user_input)
    print(f"PHI-3.5: {response}")
    
    # 모델 목록 확인
    print("\n" + "="*30)
    print("정의된 모델들 확인")
    print("="*30)
    
    models_list = await dialog_system.list_models()
    print(models_list)
    
    # 모델 테스트
    print("\n" + "="*30)
    print("모델 테스트")
    print("="*30)
    
    if "커피_만들기" in dialog_system.behavior_models:
        test_result = await dialog_system.test_model("커피_만들기")
        print(test_result)

def show_usage_guide():
    """사용 가이드 표시"""
    print("\n" + "="*60)
    print("📖 행동모델 대화형 정의 시스템 사용 가이드")
    print("="*60)
    
    print("""
🎯 시스템 특징:
- PHI-3.5와 자연어 대화로 행동모델 정의
- 실시간 모델 생성 및 수정
- 시각적 모델 관리 및 테스트
- JSON 형태로 구조화된 모델 저장

💬 대화 방법:
1. 자연어로 원하는 행동을 설명
2. PHI-3.5가 자동으로 구조화된 모델 생성
3. 생성된 모델을 확인하고 수정
4. 모델 테스트 및 실행

🔧 사용 가능한 명령어:
- "새 모델 만들기": 새로운 행동모델 생성
- "모델 수정하기": 기존 모델 수정
- "모델 보기": 정의된 모델들 확인
- "모델 테스트": 모델 실행 테스트
- "종료": 대화 종료

📝 행동모델 구조:
{
  "name": "모델명",
  "description": "모델 설명",
  "motion_primitives": [
    {
      "name": "동작명",
      "parameters": {"매개변수": "값"},
      "preconditions": ["전제조건"],
      "postconditions": ["결과조건"]
    }
  ],
  "parameters": {"전역매개변수": "값"},
  "constraints": {"제약조건": "값"}
}

🚀 실행 방법:
1. 콘솔 버전: python behavior_model_dialog.py
2. GUI 버전: python behavior_model_gui.py

💡 예제 대화:
사용자: "커피를 만드는 행동모델을 정의해주세요"
PHI-3.5: [구조화된 JSON 모델 생성]

사용자: "청소하는 행동모델을 만들어주세요"
PHI-3.5: [청소 행동모델 생성]

사용자: "모델을 수정해주세요"
PHI-3.5: [기존 모델 수정]
""")

if __name__ == "__main__":
    print("🎯 행동모델 대화형 정의 시스템 예제")
    print("=" * 50)
    
    # 사용 가이드 표시
    show_usage_guide()
    
    # 예제 실행 여부 확인
    response = input("\n예제를 실행하시겠습니까? (y/n): ").strip().lower()
    
    if response in ['y', 'yes', '예']:
        asyncio.run(behavior_model_example())
    else:
        print("\n사용 가이드를 참고하여 직접 시스템을 실행해보세요!")
        print("  python behavior_model_dialog.py")
        print("  또는")
        print("  python behavior_model_gui.py")
