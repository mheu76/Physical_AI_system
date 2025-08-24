#!/usr/bin/env python3
"""
sLM Foundation Model Training Script

Physical AI 시스템을 위한 sLM Foundation Model의 
훈련을 실행하는 메인 스크립트입니다.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('slm_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def train_slm_foundation_model(config: dict):
    """sLM Foundation Model 훈련 실행"""
    
    logger.info("🚀 sLM Foundation Model 훈련 시작")
    logger.info(f"📋 설정: {config}")
    
    try:
        # Foundation Model 초기화
        from foundation_model.slm_foundation import SLMFoundation
        
        foundation = SLMFoundation(
            model_type=config.get("model_type", "phi35"),
            model_name=config.get("model_name", "microsoft/Phi-3.5-mini-instruct"),
            device=config.get("device", "auto"),
            learning_config=config.get("learning_config", {}),
            training_output_dir=config.get("training_output_dir", "models/slm_foundation"),
            num_epochs=config.get("num_epochs", 3),
            batch_size=config.get("batch_size", 4),
            learning_rate=config.get("learning_rate", 5e-5)
        )
        
        # 초기화
        logger.info("🔧 Foundation Model 초기화 중...")
        await foundation.initialize()
        logger.info("✅ Foundation Model 초기화 완료")
        
        # 훈련 데이터 생성 (기본 예제들)
        if config.get("generate_training_data", True):
            logger.info("📚 훈련 데이터 생성 중...")
            await generate_training_data(foundation, config)
        
        # 모델 훈련
        logger.info("모델 훈련 시작...")
        training_result = await foundation.train_model(
            resume_from_checkpoint=config.get("resume_from_checkpoint", False)
        )
        
        if training_result["success"]:
            logger.info("✅ 모델 훈련 완료")
            logger.info(f"📊 훈련 손실: {training_result['training_loss']:.4f}")
            logger.info(f"📊 검증 손실: {training_result['validation_loss']:.4f}")
            logger.info(f"⏱️ 훈련 시간: {training_result['training_time']:.1f}초")
        else:
            logger.error(f"모델 훈련 실패: {training_result.get('error', 'Unknown error')}")
            return False
        
        # 모델 평가
        if config.get("evaluate_model", True):
            logger.info("🔍 모델 성능 평가 중...")
            eval_result = await foundation.evaluate_model()
            
            if eval_result["success"]:
                logger.info("✅ 모델 평가 완료")
                logger.info(f"📊 정확도: {eval_result['accuracy']:.3f}")
                logger.info(f"📊 평균 손실: {eval_result['average_loss']:.4f}")
            else:
                logger.warning(f"⚠️ 모델 평가 실패: {eval_result.get('error', 'Unknown error')}")
        
        # 모델 내보내기
        if config.get("export_model", True):
            logger.info("💾 모델 내보내기 중...")
            export_result = await foundation.export_trained_model(
                config.get("export_path", "models/slm_foundation_exported")
            )
            
            if export_result["success"]:
                logger.info("✅ 모델 내보내기 완료")
                logger.info(f"📁 내보내기 경로: {export_result['export_path']}")
            else:
                logger.warning(f"⚠️ 모델 내보내기 실패: {export_result.get('error', 'Unknown error')}")
        
        # 최종 상태 출력
        final_status = await foundation.get_training_status()
        logger.info("📊 최종 훈련 상태:")
        logger.info(f"   - 총 훈련 예제: {final_status.get('total_examples', 0)}개")
        logger.info(f"   - 검증 예제: {final_status.get('validation_examples', 0)}개")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 훈련 실행 실패: {e}")
        return False

async def generate_training_data(foundation, config: dict):
    """훈련 데이터 생성"""
    
    # 기본 훈련 미션들
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
        },
        {
            "mission": "Assist in laboratory by preparing chemical solutions",
            "context": {"environment": "laboratory", "safety_level": "critical"}
        },
        {
            "mission": "Help in kitchen by chopping vegetables safely",
            "context": {"environment": "kitchen", "safety_level": "high"}
        }
    ]
    
    logger.info(f"📋 {len(training_missions)}개의 미션으로 훈련 데이터 생성")
    
    for i, mission_data in enumerate(training_missions, 1):
        try:
            logger.info(f"📚 훈련 데이터 {i}/{len(training_missions)}: {mission_data['mission']}")
            
            result = await foundation.process_mission_with_learning(
                mission=mission_data['mission'],
                context=mission_data['context']
            )
            
            if result['success']:
                logger.info(f"   ✅ 학습 가치: {result['learning_value']:.3f}")
            else:
                logger.warning(f"   ⚠️ 처리 실패: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"   ❌ 오류: {e}")
        
        # 잠시 대기
        await asyncio.sleep(0.1)

def load_config(config_path: str = None) -> dict:
    """설정 파일 로드"""
    
    default_config = {
        "model_type": "phi35",
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "device": "auto",
        "learning_config": {
            "enabled": True,
            "learning_rate": 0.01,
            "min_confidence_threshold": 0.7,
            "max_examples": 1000,
            "pattern_update_interval": 10,
            "adaptation_threshold": 0.6
        },
        "training_output_dir": "models/slm_foundation",
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "generate_training_data": True,
        "evaluate_model": True,
        "export_model": True,
        "resume_from_checkpoint": False
    }
    
    if config_path and Path(config_path).exists():
        import json
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
        logger.info(f"📄 설정 파일 로드됨: {config_path}")
    
    return default_config

def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="sLM Foundation Model 훈련 스크립트")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--model-type", type=str, default="phi35", help="모델 타입")
    parser.add_argument("--model-name", type=str, help="모델 이름")
    parser.add_argument("--device", type=str, default="auto", help="디바이스 (auto/cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=3, help="훈련 에포크 수")
    parser.add_argument("--batch-size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="학습률")
    parser.add_argument("--output-dir", type=str, default="models/slm_foundation", help="출력 디렉토리")
    parser.add_argument("--resume", action="store_true", help="체크포인트에서 재개")
    parser.add_argument("--no-eval", action="store_true", help="모델 평가 건너뛰기")
    parser.add_argument("--no-export", action="store_true", help="모델 내보내기 건너뛰기")
    parser.add_argument("--no-data-gen", action="store_true", help="훈련 데이터 생성 건너뛰기")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 명령행 인수로 설정 덮어쓰기
    if args.model_type:
        config["model_type"] = args.model_type
    if args.model_name:
        config["model_name"] = args.model_name
    if args.device:
        config["device"] = args.device
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.output_dir:
        config["training_output_dir"] = args.output_dir
    if args.resume:
        config["resume_from_checkpoint"] = True
    if args.no_eval:
        config["evaluate_model"] = False
    if args.no_export:
        config["export_model"] = False
    if args.no_data_gen:
        config["generate_training_data"] = False
    
    # 훈련 실행
    success = asyncio.run(train_slm_foundation_model(config))
    
    if success:
        logger.info("sLM Foundation Model 훈련 완료!")
        sys.exit(0)
    else:
        logger.error("sLM Foundation Model 훈련 실패!")
        sys.exit(1)

if __name__ == "__main__":
    main()
