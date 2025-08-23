"""
피지컬 AI 시스템 메인 엔트리포인트

발달적 학습과 체화된 지능을 구현하는 Physical AI 시스템의 
메인 실행 파일입니다.
"""

import argparse
import asyncio
import logging
from pathlib import Path

from foundation_model.slm_foundation import SLMFoundation
from developmental_learning.dev_engine import DevelopmentalEngine
from ai_agent_execution.agent_executor import AgentExecutor
from hardware_abstraction.hal_manager import HardwareManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysicalAI:
    """메인 Physical AI 시스템 클래스 - PHI-3.5 내장"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # PHI-3.5 Foundation Model 설정 추출
        foundation_config = self.config.get("foundation_model", {})
        model_type = foundation_config.get("model_type", "phi35")
        model_config = foundation_config.get("phi35", {})
        
        # 4개 핵심 레이어 초기화
        self.slm_foundation = SLMFoundation(model_type, **model_config)
        self.dev_engine = DevelopmentalEngine()
        self.agent_executor = AgentExecutor()
        self.hw_manager = HardwareManager()
        
        # 시스템 상태
        self.system_ready = False
        self.initialization_time = None
        
        logger.info("🤖 Physical AI 시스템 (PHI-3.5 내장) 초기화 완료")
    
    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ 설정 파일 로드 성공: {config_path}")
                return config
        except Exception as e:
            logger.warning(f"⚠️  설정 파일 로드 실패: {e}")
            logger.info("기본 설정 사용")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            "foundation_model": {
                "model_type": "phi35",
                "phi35": {
                    "model_name": "microsoft/Phi-3.5-mini-instruct",
                    "device": "auto"
                }
            },
            "system": {
                "name": "Physical AI System",
                "version": "2.0.0-phi35"
            }
        }
    
    async def initialize(self):
        """시스템 전체 초기화 - PHI-3.5 포함"""
        import time
        start_time = time.time()
        
        logger.info("🚀 Physical AI 시스템 초기화 시작...")
        
        try:
            # 1. 하드웨어 레이어 초기화
            logger.info("🔌 하드웨어 추상화 레이어 초기화...")
            await self.hw_manager.initialize()
            
            # 2. AI Agent 실행 레이어 초기화
            logger.info("⚡ AI Agent 실행 레이어 초기화...")
            await self.agent_executor.initialize(self.hw_manager)
            
            # 3. 발달적 학습 엔진 초기화
            logger.info("🌱 발달적 학습 엔진 초기화...")
            await self.dev_engine.initialize()
            
            # 4. PHI-3.5 Foundation Model 초기화 (가장 시간이 오래 걸림)
            logger.info("🧠 PHI-3.5 Foundation Model 초기화...")
            await self.slm_foundation.initialize()
            
            # 초기화 완료
            self.initialization_time = time.time() - start_time
            self.system_ready = True
            
            logger.info(f"✅ 시스템 초기화 완료! ({self.initialization_time:.2f}초)")
            
            # PHI-3.5 모델 정보 출력
            if self.slm_foundation.phi35_ai:
                model_info = self.slm_foundation.performance_metrics["model_info"]
                logger.info(f"📊 PHI-3.5 모델 정보: {model_info.get('model_name', 'Unknown')} on {model_info.get('device', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
            self.system_ready = False
            raise
    
    async def execute_mission(self, mission: str):
        """미션 실행 메인 루프"""
        logger.info(f"미션 수행: {mission}")
        
        # 1. Foundation Model이 미션 해석 및 계획 수립
        task_plan = await self.slm_foundation.interpret_mission(mission)
        
        # 2. Developmental Engine이 필요한 스킬 확인/학습
        required_skills = await self.dev_engine.analyze_required_skills(task_plan)
        
        # 3. Agent Executor가 실제 물리적 실행
        execution_result = await self.agent_executor.execute(task_plan, required_skills)
        
        # 4. 실행 결과를 Developmental Engine에 피드백
        await self.dev_engine.learn_from_experience(execution_result)
        
        return execution_result
    
    async def developmental_learning_cycle(self):
        """지속적인 발달적 학습 사이클"""
        while True:
            # 자율적인 탐색 및 학습
            await self.dev_engine.autonomous_exploration()
            
            # 30분마다 실행
            await asyncio.sleep(1800)

async def main():
    parser = argparse.ArgumentParser(description='Physical AI System')
    parser.add_argument('--config', default='configs/default.yaml', 
                       help='Configuration file path')
    parser.add_argument('--mission', type=str, 
                       help='Mission to execute')
    
    args = parser.parse_args()
    
    # Physical AI 시스템 생성 및 초기화
    physical_ai = PhysicalAI(args.config)
    await physical_ai.initialize()
    
    if args.mission:
        # 특정 미션 실행
        result = await physical_ai.execute_mission(args.mission)
        logger.info(f"미션 실행 완료: {result}")
    else:
        # 지속적인 학습 모드
        logger.info("지속적인 발달적 학습 모드 시작")
        await physical_ai.developmental_learning_cycle()

if __name__ == "__main__":
    asyncio.run(main())
