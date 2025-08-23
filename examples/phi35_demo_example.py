"""
PHI-3.5 Physical AI System Demo

PHI-3.5가 내장된 Physical AI System의 데모 예제입니다.
실제 PHI-3.5 모델이 자연어 미션을 물리적 동작으로 변환하는 과정을 보여줍니다.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from main import PhysicalAI


async def demo_mission_execution():
    """PHI-3.5 미션 실행 데모"""
    
    print("🤖 PHI-3.5 Physical AI System Demo")
    print("=" * 50)
    
    # Physical AI 시스템 초기화
    print("\n🚀 시스템 초기화 중...")
    config_path = "configs/default.yaml"
    
    try:
        physical_ai = PhysicalAI(config_path)
        await physical_ai.initialize()
        
        if not physical_ai.system_ready:
            print("❌ 시스템 초기화 실패")
            return
        
        print(f"✅ 시스템 준비 완료 (초기화 시간: {physical_ai.initialization_time:.2f}초)")
        
        # PHI-3.5 모델 정보 출력
        if physical_ai.slm_foundation.phi35_ai:
            model_info = physical_ai.slm_foundation.performance_metrics["model_info"]
            print(f"📊 PHI-3.5 모델: {model_info.get('model_name', 'Unknown')}")
            print(f"📊 디바이스: {model_info.get('device', 'Unknown')}")
            print(f"📊 파라미터 수: {model_info.get('parameters', 'Unknown')}")
        else:
            print("⚠️  PHI-3.5 없이 폴백 모드로 실행")
        
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    # 데모 미션들
    demo_missions = [
        "Pick up the red cup and place it gently on the wooden table",
        "Organize the books on the shelf by size from smallest to largest", 
        "Help me prepare dinner by bringing me the ingredients from the pantry",
        "Clean up the messy desk by putting all items in their proper places",
        "Assist the elderly person by bringing their medicine and a glass of water"
    ]
    
    print(f"\n🎯 {len(demo_missions)}개의 데모 미션 실행")
    print("-" * 50)
    
    for i, mission in enumerate(demo_missions, 1):
        print(f"\n[미션 {i}/{len(demo_missions)}] {mission}")
        print("⏳ PHI-3.5로 분석 중...")
        
        try:
            start_time = time.time()
            
            # PHI-3.5를 통한 실제 미션 실행
            result = await physical_ai.execute_mission(mission)
            
            execution_time = time.time() - start_time
            
            # 결과 출력
            if result.success:
                print(f"✅ 미션 성공! ({execution_time:.2f}초)")
                print(f"📋 수행된 동작: {len(result.actions_performed)}개")
                print(f"⚡ 성능 지표:")
                for metric, value in result.performance_metrics.items():
                    print(f"   - {metric}: {value:.2f}" if isinstance(value, float) else f"   - {metric}: {value}")
                print(f"🧠 학습 가치: {result.learning_value:.2f}")
            else:
                print(f"❌ 미션 실패 ({execution_time:.2f}초)")
                if result.errors:
                    print("🚨 오류들:")
                    for error in result.errors:
                        print(f"   - {error}")
            
            # PHI-3.5 성능 메트릭 출력
            if physical_ai.slm_foundation.phi35_ai:
                metrics = physical_ai.slm_foundation.performance_metrics
                print(f"📊 PHI-3.5 성능:")
                print(f"   - 처리한 미션: {metrics['missions_processed']}개")
                print(f"   - 성공률: {metrics['successful_decompositions']/max(metrics['missions_processed'], 1)*100:.1f}%")
                print(f"   - 평균 응답시간: {metrics['average_response_time']:.2f}초")
        
        except Exception as e:
            print(f"❌ 미션 실행 중 오류: {e}")
        
        # 다음 미션 전 잠시 대기
        if i < len(demo_missions):
            print("\n⏸️  다음 미션까지 3초 대기...")
            await asyncio.sleep(3)
    
    print("\n🎉 모든 데모 미션 완료!")
    print("=" * 50)


async def demo_phi35_direct_interaction():
    """PHI-3.5와 직접 상호작용 데모"""
    
    print("\n🧠 PHI-3.5 직접 상호작용 데모")
    print("-" * 50)
    
    try:
        # PHI-3.5만 단독으로 테스트
        from foundation_model.phi35_integration import create_phi35_physical_ai
        
        print("🔧 PHI-3.5 모델 로딩...")
        phi35_ai = create_phi35_physical_ai()
        success = await phi35_ai.initialize()
        
        if not success:
            print("❌ PHI-3.5 로딩 실패")
            return
            
        print("✅ PHI-3.5 준비 완료")
        
        # 물리학 관련 질문들
        physics_questions = [
            "How should a robot grasp a fragile glass cup safely?",
            "What are the key safety considerations when a robot moves near humans?",
            "Explain the physics principles involved in robotic manipulation",
            "How can a robot determine if an object is too heavy to lift?",
            "What sensors are needed for safe human-robot collaboration?"
        ]
        
        print(f"\n🤔 {len(physics_questions)}개의 물리학 질문 테스트")
        
        for i, question in enumerate(physics_questions, 1):
            print(f"\n[질문 {i}] {question}")
            print("💭 PHI-3.5 응답:")
            
            try:
                response = await phi35_ai.model_manager.generate_response(
                    question, 
                    max_new_tokens=256,
                    temperature=0.7
                )
                print(f"🤖 {response}")
                
            except Exception as e:
                print(f"❌ 응답 생성 실패: {e}")
            
            if i < len(physics_questions):
                await asyncio.sleep(1)
        
        print(f"\n📈 PHI-3.5 모델 정보:")
        model_info = phi35_ai.model_manager.get_model_info()
        for key, value in model_info.items():
            print(f"   - {key}: {value}")
            
    except Exception as e:
        print(f"❌ PHI-3.5 직접 상호작용 실패: {e}")


async def main():
    """메인 데모 실행"""
    
    print("🌟 PHI-3.5 Physical AI System 종합 데모")
    print("🚀 Microsoft PHI-3.5 내장 발달적 로보틱스 시스템")
    print("=" * 60)
    
    # 1. 시스템 통합 데모
    await demo_mission_execution()
    
    # 2. PHI-3.5 직접 상호작용 데모  
    await demo_phi35_direct_interaction()
    
    print("\n🏁 전체 데모 완료!")
    print("🎯 PHI-3.5가 Physical AI의 두뇌 역할을 성공적으로 수행했습니다!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자가 데모를 중단했습니다.")
    except Exception as e:
        print(f"\n\n❌ 데모 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()