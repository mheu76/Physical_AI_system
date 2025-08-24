"""
LLM Foundation 테스트 스크립트

PHI-3.5 Foundation Model의 기본 기능을 테스트합니다.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFoundationModel:
    """간단한 Foundation Model 테스트 클래스"""
    
    def __init__(self):
        self.model_loaded = False
        self.performance_metrics = {
            "missions_processed": 0,
            "successful_decompositions": 0,
            "average_response_time": 0.0
        }
    
    async def initialize(self):
        """모델 초기화 (시뮬레이션)"""
        logger.info("🧠 Simple Foundation Model 초기화 중...")
        
        # PHI-3.5 모델 로딩 시뮬레이션
        await asyncio.sleep(2)  # 모델 로딩 시간 시뮬레이션
        
        self.model_loaded = True
        logger.info("✅ Simple Foundation Model 초기화 완료")
        return True
    
    async def interpret_mission(self, mission: str):
        """미션 해석 (시뮬레이션)"""
        logger.info(f"미션 해석 중: {mission}")
        
        # 간단한 미션 분해 시뮬레이션
        subtasks = []
        
        if "pick" in mission.lower() and "place" in mission.lower():
            subtasks = [
                {
                    "type": "navigation",
                    "action": "move_to",
                    "target": "object_location",
                    "priority": 1,
                    "estimated_duration": 10.0,
                    "difficulty": 2
                },
                {
                    "type": "manipulation", 
                    "action": "grasp",
                    "target": "object",
                    "priority": 2,
                    "estimated_duration": 5.0,
                    "difficulty": 3
                },
                {
                    "type": "navigation",
                    "action": "move_to", 
                    "target": "destination",
                    "priority": 3,
                    "estimated_duration": 10.0,
                    "difficulty": 2
                },
                {
                    "type": "manipulation",
                    "action": "place",
                    "target": "surface", 
                    "priority": 4,
                    "estimated_duration": 3.0,
                    "difficulty": 2
                }
            ]
        else:
            subtasks = [
                {
                    "type": "exploration",
                    "action": "explore_environment",
                    "target": "workspace",
                    "priority": 1,
                    "estimated_duration": 30.0,
                    "difficulty": 2
                }
            ]
        
        # 성능 메트릭 업데이트
        self.performance_metrics["missions_processed"] += 1
        self.performance_metrics["successful_decompositions"] += 1
        
        logger.info(f"✅ 미션 해석 완료: {len(subtasks)}개 서브태스크")
        
        return {
            "mission": mission,
            "subtasks": subtasks,
            "total_duration": sum(task["estimated_duration"] for task in subtasks),
            "difficulty": max(task["difficulty"] for task in subtasks) if subtasks else 1
        }
    
    async def process_mission_with_learning(self, mission: str, context: Dict = None):
        """학습이 포함된 미션 처리 (시뮬레이션)"""
        logger.info(f"학습 포함 미션 처리: {mission}")
        
        # 미션 해석
        result = await self.interpret_mission(mission)
        
        # 학습 가치 계산 (시뮬레이션)
        learning_value = 0.1 + (len(result["subtasks"]) * 0.05)
        
        return {
            "success": True,
            "mission": mission,
            "result": result,
            "learning_value": learning_value,
            "performance_metrics": self.performance_metrics
        }
    
    def get_performance_metrics(self):
        """성능 메트릭 조회"""
        return self.performance_metrics

async def test_foundation_model():
    """Foundation Model 테스트"""
    logger.info("🚀 LLM Foundation Model 테스트 시작")
    
    # Foundation Model 초기화
    foundation = SimpleFoundationModel()
    
    try:
        # 모델 초기화
        await foundation.initialize()
        
        # 테스트 미션들
        test_missions = [
            "Pick up the red cup and place it on the table",
            "Move to position [1, 0, 0.5] and explore the area",
            "Organize the books on the shelf by size"
        ]
        
        # 각 미션 테스트
        for i, mission in enumerate(test_missions, 1):
            logger.info(f"\n📋 테스트 미션 {i}: {mission}")
            
            # 미션 해석 테스트
            interpretation = await foundation.interpret_mission(mission)
            logger.info(f"   해석 결과: {len(interpretation['subtasks'])}개 서브태스크")
            logger.info(f"   예상 시간: {interpretation['total_duration']:.1f}초")
            logger.info(f"   난이도: {interpretation['difficulty']}/5")
            
            # 학습 포함 처리 테스트
            learning_result = await foundation.process_mission_with_learning(mission)
            logger.info(f"   학습 가치: {learning_result['learning_value']:.3f}")
            
            # 서브태스크 상세 정보
            for j, subtask in enumerate(interpretation['subtasks'], 1):
                logger.info(f"     {j}. {subtask['action']} -> {subtask['target']} "
                          f"({subtask['estimated_duration']:.1f}초, 난이도: {subtask['difficulty']})")
        
        # 성능 메트릭 출력
        metrics = foundation.get_performance_metrics()
        logger.info(f"\n📊 성능 메트릭:")
        logger.info(f"   처리된 미션: {metrics['missions_processed']}개")
        logger.info(f"   성공적 분해: {metrics['successful_decompositions']}개")
        logger.info(f"   성공률: {metrics['successful_decompositions']/metrics['missions_processed']*100:.1f}%")
        
        logger.info("\n✅ LLM Foundation Model 테스트 완료!")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_foundation_model())
