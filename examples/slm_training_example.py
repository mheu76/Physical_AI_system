"""
sLM Foundation Model Training Example

Physical AI 시스템을 위한 sLM Foundation Model의 
훈련, 파인튜닝, 성능 평가를 시연하는 예제입니다.
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
from foundation_model.slm_training_module import TrainingExample, TrainingConfig

async def test_slm_training_comprehensive():
    """sLM Foundation Model 종합 훈련 테스트"""
    
    print("🧠 sLM Foundation Model 종합 훈련 테스트")
    print("=" * 80)
    
    # 1. Foundation Model 초기화 (훈련 모듈 포함)
    print("\n🚀 Foundation Model 초기화 중...")
    
    training_config = {
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
        learning_config=training_config,
        training_output_dir="models/slm_foundation",
        num_epochs=2,  # 빠른 테스트를 위해 2 에포크
        batch_size=2,   # 메모리 절약을 위해 작은 배치
        learning_rate=1e-4
    )
    
    try:
        await foundation.initialize()
        print("✅ Foundation Model 초기화 완료")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        print("⚠️ 폴백 모드로 계속 진행합니다.")
        return
    
    # 2. 다양한 미션으로 훈련 데이터 생성
    print(f"\n📚 훈련 데이터 생성 중...")
    print("-" * 60)
    
    training_missions = [
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
        },
        {
            "mission": "Sort the laundry by color and fabric type",
            "context": {"environment": "simple", "safety_level": "normal"}
        },
        {
            "mission": "Set up the dining table with plates, utensils, and glasses",
            "context": {"environment": "complex", "safety_level": "normal"}
        },
        {
            "mission": "Water the plants in the living room",
            "context": {"environment": "simple", "safety_level": "low"}
        }
    ]
    
    print(f"📋 {len(training_missions)}개의 미션으로 훈련 데이터 생성")
    
    for i, mission_data in enumerate(training_missions, 1):
        print(f"\n[훈련 데이터 {i}/{len(training_missions)}]")
        print(f"미션: {mission_data['mission']}")
        
        try:
            # 학습이 포함된 미션 처리
            start_time = time.time()
            result = await foundation.process_mission_with_learning(
                mission=mission_data['mission'],
                context=mission_data['context']
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
        await asyncio.sleep(0.5)
    
    # 3. 훈련 상태 확인
    print(f"\n📊 훈련 상태 확인")
    print("-" * 60)
    
    try:
        training_status = await foundation.get_training_status()
        
        if "error" not in training_status:
            print(f"📚 총 훈련 예제: {training_status['total_examples']}개")
            print(f"📚 훈련 예제: {training_status['total_examples']}개")
            print(f"📚 검증 예제: {training_status['validation_examples']}개")
            print(f"🔄 훈련 중: {training_status['is_training']}")
            
            # 훈련 메트릭
            metrics = training_status['training_metrics']
            print(f"📈 최고 검증 손실: {metrics['best_validation_loss']:.4f}")
            print(f"⏱️ 총 훈련 시간: {metrics['training_time']:.1f}초")
        
        else:
            print(f"❌ 훈련 상태 조회 실패: {training_status['error']}")
    
    except Exception as e:
        print(f"❌ 훈련 상태 조회 오류: {e}")
    
    # 4. 모델 훈련 실행
    print(f"\n🚀 모델 훈련 실행")
    print("-" * 60)
    
    try:
        print("🎯 모델 훈련 시작...")
        start_time = time.time()
        
        training_result = await foundation.train_model(resume_from_checkpoint=False)
        training_time = time.time() - start_time
        
        if training_result['success']:
            print(f"✅ 훈련 완료 ({training_time:.1f}초)")
            print(f"📊 훈련 손실: {training_result['training_loss']:.4f}")
            print(f"📊 검증 손실: {training_result['validation_loss']:.4f}")
            print(f"💾 모델 경로: {training_result['model_path']}")
        else:
            print(f"❌ 훈련 실패: {training_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ 훈련 실행 오류: {e}")
    
    # 5. 모델 성능 평가
    print(f"\n🔍 모델 성능 평가")
    print("-" * 60)
    
    try:
        print("📊 모델 성능 평가 중...")
        
        evaluation_result = await foundation.evaluate_model()
        
        if evaluation_result['success']:
            print(f"✅ 평가 완료")
            print(f"📊 정확도: {evaluation_result['accuracy']:.3f}")
            print(f"📊 평균 손실: {evaluation_result['average_loss']:.4f}")
            print(f"📊 총 예제: {evaluation_result['total_examples']}개")
            print(f"📊 정확 예측: {evaluation_result['correct_predictions']}개")
        else:
            print(f"❌ 평가 실패: {evaluation_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ 성능 평가 오류: {e}")
    
    # 6. 학습 인사이트 분석
    print(f"\n🧠 학습 인사이트 분석")
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
            
            # 상위 패턴
            top_patterns = insights['top_patterns']
            if top_patterns:
                print(f"\n🏆 상위 지식 패턴:")
                for i, pattern in enumerate(top_patterns[:3], 1):
                    print(f"   {i}. {pattern['description']} (신뢰도: {pattern['confidence']:.2f})")
        
        else:
            print(f"❌ 학습 인사이트 조회 실패: {insights['error']}")
    
    except Exception as e:
        print(f"❌ 학습 인사이트 조회 오류: {e}")
    
    # 7. 훈련된 모델 내보내기
    print(f"\n💾 훈련된 모델 내보내기")
    print("-" * 60)
    
    try:
        print("📦 모델 내보내기 중...")
        
        export_result = await foundation.export_trained_model("models/slm_foundation_exported")
        
        if export_result['success']:
            print(f"✅ 모델 내보내기 완료")
            print(f"📁 내보내기 경로: {export_result['export_path']}")
            
            # 설정 정보 출력
            config = export_result['config']
            print(f"🔧 모델 이름: {config['model_name']}")
            print(f"📊 훈련 설정: {config['training_config']['num_epochs']} 에포크")
            print(f"📅 내보내기 시간: {config['export_timestamp']}")
        else:
            print(f"❌ 모델 내보내기 실패: {export_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ 모델 내보내기 오류: {e}")
    
    # 8. 최종 성능 요약
    print(f"\n📊 최종 성능 요약")
    print("-" * 60)
    
    try:
        # 훈련 상태 최종 확인
        final_status = await foundation.get_training_status()
        
        if "error" not in final_status:
            print(f"📚 총 훈련 예제: {final_status['total_examples']}개")
            print(f"📚 검증 예제: {final_status['validation_examples']}개")
            
            # 훈련 메트릭
            metrics = final_status['training_metrics']
            print(f"📈 최고 검증 손실: {metrics['best_validation_loss']:.4f}")
            print(f"⏱️ 총 훈련 시간: {metrics['training_time']:.1f}초")
            
            # 성능 평가
            eval_result = await foundation.evaluate_model()
            if eval_result['success']:
                print(f"📊 최종 정확도: {eval_result['accuracy']:.3f}")
                print(f"📊 평균 손실: {eval_result['average_loss']:.4f}")
        
        else:
            print(f"❌ 최종 상태 조회 실패: {final_status['error']}")
    
    except Exception as e:
        print(f"❌ 최종 성능 요약 오류: {e}")
    
    print(f"\n🎉 sLM Foundation Model 종합 훈련 테스트 완료!")
    print("=" * 80)

async def test_incremental_training():
    """점진적 훈련 테스트"""
    
    print("\n🔄 점진적 훈련 테스트")
    print("=" * 60)
    
    # Foundation Model 초기화
    foundation = SLMFoundation(
        model_type="phi35",
        training_output_dir="models/slm_foundation_incremental",
        num_epochs=1,  # 빠른 테스트
        batch_size=2
    )
    
    try:
        await foundation.initialize()
        
        # 1단계: 기본 훈련
        print("📚 1단계: 기본 훈련")
        basic_missions = [
            "Pick up the cup and place it on the table",
            "Move the book to the shelf"
        ]
        
        for mission in basic_missions:
            await foundation.process_mission_with_learning(
                mission=mission,
                context={"environment": "simple"}
            )
        
        # 기본 훈련 실행
        await foundation.train_model()
        
        # 2단계: 추가 훈련
        print("📚 2단계: 추가 훈련")
        advanced_missions = [
            "Organize the desk by sorting items into categories",
            "Set up the dining table with proper arrangement"
        ]
        
        for mission in advanced_missions:
            await foundation.process_mission_with_learning(
                mission=mission,
                context={"environment": "complex"}
            )
        
        # 추가 훈련 실행 (체크포인트에서 재개)
        await foundation.train_model(resume_from_checkpoint=True)
        
        # 최종 평가
        eval_result = await foundation.evaluate_model()
        print(f"📊 최종 정확도: {eval_result['accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ 점진적 훈련 실패: {e}")

async def test_custom_training_examples():
    """사용자 정의 훈련 예제 테스트"""
    
    print("\n🎯 사용자 정의 훈련 예제 테스트")
    print("=" * 60)
    
    # Foundation Model 초기화
    foundation = SLMFoundation(
        model_type="phi35",
        training_output_dir="models/slm_foundation_custom"
    )
    
    try:
        await foundation.initialize()
        
        # 사용자 정의 훈련 예제 생성
        custom_examples = [
            TrainingExample(
                mission="Assist in laboratory by preparing chemical solutions",
                context={"environment": "laboratory", "safety_level": "critical"},
                subtasks=[
                    {"action": "move_to", "target": "chemical_storage"},
                    {"action": "grasp", "target": "chemical_bottle"},
                    {"action": "move_to", "target": "workbench"},
                    {"action": "place", "target": "mixing_area"}
                ],
                constraints={"max_force": 10.0, "safety_distance": 0.5},
                success_criteria=["chemical_prepared", "safety_maintained"],
                execution_result={"success": True, "efficiency": 0.9},
                learning_value=0.95
            ),
            TrainingExample(
                mission="Help in kitchen by chopping vegetables safely",
                context={"environment": "kitchen", "safety_level": "high"},
                subtasks=[
                    {"action": "move_to", "target": "cutting_board"},
                    {"action": "grasp", "target": "knife"},
                    {"action": "grasp", "target": "vegetable"},
                    {"action": "place", "target": "chopped_vegetables"}
                ],
                constraints={"max_force": 15.0, "safety_distance": 0.2},
                success_criteria=["vegetables_chopped", "no_injuries"],
                execution_result={"success": True, "efficiency": 0.8},
                learning_value=0.85
            )
        ]
        
        # 훈련 모듈에 예제 추가
        for example in custom_examples:
            await foundation.training_module.add_training_example(example)
        
        print(f"📚 {len(custom_examples)}개 사용자 정의 예제 추가됨")
        
        # 훈련 실행
        await foundation.train_model()
        
        # 평가
        eval_result = await foundation.evaluate_model()
        print(f"📊 사용자 정의 예제 평가: 정확도 {eval_result['accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ 사용자 정의 훈련 실패: {e}")

if __name__ == "__main__":
    print("🧠 sLM Foundation Model Training Example")
    print("Physical AI 시스템을 위한 sLM Foundation Model의 훈련을 테스트합니다.")
    
    # 메인 훈련 테스트
    asyncio.run(test_slm_training_comprehensive())
    
    # 추가 테스트들 (선택적)
    try:
        asyncio.run(test_incremental_training())
        asyncio.run(test_custom_training_examples())
    except Exception as e:
        print(f"⚠️ 추가 테스트 건너뜀: {e}")
    
    print("\n✨ 모든 훈련 테스트 완료!")
