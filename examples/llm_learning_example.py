"""
LLM Foundation Learning Module Example

PHI-3.5 기반 Physical AI 시스템의 LLM 학습 모듈을 테스트하는 예제입니다.
지속적 학습, 적응적 추론, 지식 증강 등의 기능을 시연합니다.
"""

import asyncio
import sys
import os
import time
import json
from pathlib import Path

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from foundation_model.slm_foundation import SLMFoundation


async def test_llm_learning_module():
    """LLM 학습 모듈 테스트"""
    
    print("🧠 LLM Foundation Learning Module 테스트")
    print("=" * 60)
    
    # 1. Foundation Model 초기화 (LLM 학습 모듈 포함)
    print("\n🚀 Foundation Model 초기화 중...")
    
    learning_config = {
        "enabled": True,
        "learning_rate": 0.01,
        "min_confidence_threshold": 0.7,
        "max_examples": 1000,
        "pattern_update_interval": 10,
        "adaptation_threshold": 0.6
    }
    
    foundation = SLMFoundation(
        model_type="phi35",
        model_name="microsoft/Phi-3.5-mini-instruct",
        device="auto",
        learning_config=learning_config
    )
    
    try:
        await foundation.initialize()
        print("✅ Foundation Model 초기화 완료")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    # 2. 다양한 미션으로 학습 테스트
    test_missions = [
        {
            "mission": "Pick up the red cup and place it on the table",
            "context": {"environment": "simple", "safety_level": "normal"}
        },
        {
            "mission": "Organize the books on the shelf by size",
            "context": {"environment": "complex", "safety_level": "high"}
        },
        {
            "mission": "Clean up the messy desk by putting items in their proper places",
            "context": {"environment": "complex", "safety_level": "normal"}
        },
        {
            "mission": "Help me prepare dinner by bringing ingredients from the pantry",
            "context": {"environment": "simple", "safety_level": "high"}
        },
        {
            "mission": "Assist the elderly person by bringing their medicine and water",
            "context": {"environment": "complex", "safety_level": "high"}
        }
    ]
    
    print(f"\n📚 {len(test_missions)}개의 미션으로 학습 테스트 시작")
    print("-" * 60)
    
    for i, test_case in enumerate(test_missions, 1):
        print(f"\n[학습 세션 {i}/{len(test_missions)}]")
        print(f"미션: {test_case['mission']}")
        print(f"컨텍스트: {test_case['context']}")
        
        try:
            # 학습이 포함된 미션 처리
            start_time = time.time()
            result = await foundation.process_mission_with_learning(
                mission=test_case['mission'],
                context=test_case['context']
            )
            processing_time = time.time() - start_time
            
            if result['success']:
                print(f"✅ 처리 완료 ({processing_time:.2f}초)")
                print(f"📊 학습 가치: {result['learning_value']:.3f}")
                print(f"📋 서브태스크: {len(result['subtasks'])}개")
                print(f"⚡ 실행 결과: {'성공' if result['execution_result']['success'] else '실패'}")
                
                # 성능 메트릭 출력
                perf_metrics = result['execution_result']['performance_metrics']
                print(f"   - 효율성: {perf_metrics['efficiency']:.2f}")
                print(f"   - 정확도: {perf_metrics['accuracy']:.2f}")
                print(f"   - 안전성: {perf_metrics['safety_score']:.2f}")
                
            else:
                print(f"❌ 처리 실패: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        # 잠시 대기 (학습 처리 시간)
        await asyncio.sleep(1)
    
    # 3. 학습 인사이트 조회
    print(f"\n📊 학습 인사이트 분석")
    print("-" * 60)
    
    try:
        insights = await foundation.get_learning_insights()
        
        if "error" not in insights:
            print(f"📚 총 학습 예제: {insights['total_examples']}개")
            print(f"🧩 지식 패턴: {insights['knowledge_patterns']}개")
            print(f"🔄 성공한 적응: {insights['successful_adaptations']}회")
            
            # 최근 성능
            recent_perf = insights['recent_performance']
            if recent_perf:
                print(f"📈 최근 성공률: {recent_perf['success_rate']:.1%}")
                print(f"📈 평균 학습 가치: {recent_perf['average_learning_value']:.3f}")
                print(f"📈 예제 수: {recent_perf['examples_count']}개")
            
            # 상위 패턴
            top_patterns = insights['top_patterns']
            if top_patterns:
                print(f"\n🏆 상위 지식 패턴:")
                for i, pattern in enumerate(top_patterns[:3], 1):
                    print(f"   {i}. {pattern['description']} (신뢰도: {pattern['confidence']:.2f})")
            
            # 학습 트렌드
            trends = insights['learning_trends']
            if trends:
                print(f"\n📈 학습 트렌드: {trends['trend_direction']}")
        
        else:
            print(f"❌ 학습 인사이트 조회 실패: {insights['error']}")
    
    except Exception as e:
        print(f"❌ 학습 인사이트 조회 오류: {e}")
    
    # 4. 학습 전략 최적화
    print(f"\n🔧 학습 전략 최적화")
    print("-" * 60)
    
    try:
        optimization = await foundation.optimize_learning_strategy()
        
        if "error" not in optimization:
            print(f"📊 최적화 점수: {optimization['optimization_score']:.2f}")
            
            recommendations = optimization['recommendations']
            if recommendations:
                print(f"💡 최적화 제안:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec['description']}")
            else:
                print("✅ 현재 학습 전략이 최적입니다!")
        
        else:
            print(f"❌ 최적화 실패: {optimization['error']}")
    
    except Exception as e:
        print(f"❌ 최적화 오류: {e}")
    
    # 5. 지식 패턴 조회
    print(f"\n🧩 지식 패턴 상세 분석")
    print("-" * 60)
    
    try:
        patterns = await foundation.get_knowledge_patterns()
        
        if "error" not in patterns:
            print(f"📊 총 패턴 수: {patterns['total_patterns']}개")
            
            if patterns['patterns']:
                print(f"\n📋 패턴 목록:")
                for pattern in patterns['patterns'][:5]:  # 상위 5개만
                    print(f"   - {pattern['id']} ({pattern['type']})")
                    print(f"     신뢰도: {pattern['confidence']:.2f}, 사용횟수: {pattern['usage_count']}")
                    print(f"     설명: {pattern['description']}")
                    print()
        
        else:
            print(f"❌ 패턴 조회 실패: {patterns['error']}")
    
    except Exception as e:
        print(f"❌ 패턴 조회 오류: {e}")
    
    # 6. 연속 학습 시뮬레이션
    print(f"\n🔄 연속 학습 시뮬레이션")
    print("-" * 60)
    
    # 같은 미션을 여러 번 반복하여 학습 효과 관찰
    repeated_mission = "Pick up the blue cup and place it gently on the shelf"
    context = {"environment": "simple", "safety_level": "normal"}
    
    print(f"미션: {repeated_mission}")
    print("5회 반복 실행으로 학습 효과 관찰...")
    
    learning_progress = []
    
    for iteration in range(5):
        try:
            result = await foundation.process_mission_with_learning(
                mission=repeated_mission,
                context=context
            )
            
            if result['success']:
                learning_value = result['learning_value']
                success = result['execution_result']['success']
                efficiency = result['execution_result']['performance_metrics']['efficiency']
                
                learning_progress.append({
                    'iteration': iteration + 1,
                    'learning_value': learning_value,
                    'success': success,
                    'efficiency': efficiency
                })
                
                print(f"   반복 {iteration + 1}: 학습가치={learning_value:.3f}, 성공={success}, 효율성={efficiency:.2f}")
            
            await asyncio.sleep(0.5)
        
        except Exception as e:
            print(f"   반복 {iteration + 1}: 오류 - {e}")
    
    # 학습 진행 상황 분석
    if learning_progress:
        print(f"\n📈 학습 진행 분석:")
        initial_learning = learning_progress[0]['learning_value']
        final_learning = learning_progress[-1]['learning_value']
        improvement = final_learning - initial_learning
        
        print(f"   초기 학습 가치: {initial_learning:.3f}")
        print(f"   최종 학습 가치: {final_learning:.3f}")
        print(f"   개선도: {improvement:+.3f}")
        
        if improvement > 0:
            print("   ✅ 학습이 효과적으로 진행되고 있습니다!")
        else:
            print("   ⚠️  학습 개선이 필요합니다.")
    
    print(f"\n🎉 LLM Foundation Learning Module 테스트 완료!")
    print("=" * 60)


async def test_advanced_learning_features():
    """고급 학습 기능 테스트"""
    
    print("\n🚀 고급 학습 기능 테스트")
    print("=" * 60)
    
    # Foundation Model 초기화
    foundation = SLMFoundation(
        model_type="phi35",
        learning_config={"enabled": True, "learning_rate": 0.02}
    )
    
    try:
        await foundation.initialize()
        
        # 1. 복잡한 미션으로 학습 테스트
        complex_missions = [
            "Navigate through the cluttered room, avoid obstacles, and find the hidden object",
            "Collaborate with another robot to move a heavy table across the room",
            "Learn to use a new tool by observing human demonstration and then perform the task",
            "Adapt to a changing environment where objects are moved while the robot is working",
            "Solve a puzzle by manipulating multiple objects in a specific sequence"
        ]
        
        print(f"🧩 복잡한 미션 학습 테스트 ({len(complex_missions)}개)")
        
        for i, mission in enumerate(complex_missions, 1):
            print(f"\n[복잡 미션 {i}] {mission}")
            
            result = await foundation.process_mission_with_learning(
                mission=mission,
                context={"environment": "complex", "difficulty": "high"}
            )
            
            if result['success']:
                print(f"   학습 가치: {result['learning_value']:.3f}")
                print(f"   성공: {result['execution_result']['success']}")
            else:
                print(f"   실패: {result.get('error', 'Unknown')}")
        
        # 2. 학습 인사이트 상세 분석
        insights = await foundation.get_learning_insights()
        print(f"\n📊 상세 학습 분석:")
        print(json.dumps(insights, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ 고급 학습 테스트 실패: {e}")


if __name__ == "__main__":
    print("🧠 LLM Foundation Learning Module Example")
    print("PHI-3.5 기반 Physical AI 시스템의 고급 학습 기능을 테스트합니다.")
    
    # 기본 학습 모듈 테스트
    asyncio.run(test_llm_learning_module())
    
    # 고급 학습 기능 테스트 (선택적)
    try:
        asyncio.run(test_advanced_learning_features())
    except Exception as e:
        print(f"⚠️  고급 학습 테스트 건너뜀: {e}")
    
    print("\n✨ 모든 테스트 완료!")
