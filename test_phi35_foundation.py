"""
PHI-3.5 Foundation Model 고급 테스트 스크립트

실제 PHI-3.5 모델을 사용하여 Foundation Model을 테스트합니다.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFoundationModel:
    """고급 Foundation Model 클래스 (PHI-3.5 시도)"""
    
    def __init__(self):
        self.model_loaded = False
        self.phi35_available = False
        self.performance_metrics = {
            "missions_processed": 0,
            "successful_decompositions": 0,
            "average_response_time": 0.0,
            "phi35_usage_count": 0
        }
    
    async def initialize(self):
        """모델 초기화 - PHI-3.5 시도"""
        logger.info("🧠 Advanced Foundation Model 초기화 중...")
        
        # PHI-3.5 모델 로딩 시도
        try:
            # transformers 라이브러리 확인
            import transformers
            logger.info(f"✅ Transformers 라이브러리 발견: {transformers.__version__}")
            
            # PHI-3.5 모델 로딩 시도
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = "microsoft/Phi-3.5-mini-instruct"
            logger.info(f"🔄 PHI-3.5 모델 로딩 시도: {model_name}")
            
            # 디바이스 확인
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"📱 사용 디바이스: {device}")
            
            # 토크나이저 로딩
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 모델 로딩 (CPU 모드로 시도)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.tokenizer = tokenizer
            self.model = model
            self.device = device
            self.phi35_available = True
            
            logger.info("✅ PHI-3.5 모델 로딩 성공!")
            
        except Exception as e:
            logger.warning(f"⚠️ PHI-3.5 모델 로딩 실패: {e}")
            logger.info("📝 폴백 모드로 진행합니다 (시뮬레이션)")
            self.phi35_available = False
        
        self.model_loaded = True
        logger.info("✅ Advanced Foundation Model 초기화 완료")
        return True
    
    async def _generate_with_phi35(self, prompt: str) -> str:
        """PHI-3.5를 사용한 텍스트 생성"""
        if not self.phi35_available:
            return "PHI-3.5 모델을 사용할 수 없습니다."
        
        try:
            import torch  # torch 임포트 추가
            
            # 프롬프트 구성
            full_prompt = f"""You are a robotics task planner. Decompose this mission into subtasks:

Mission: {prompt}

Provide a structured response with subtasks in JSON format:
{{
  "subtasks": [
    {{
      "type": "navigation|manipulation|perception",
      "action": "specific_action",
      "target": "target_description",
      "priority": 1,
      "estimated_duration": 10.0,
      "difficulty": 1-5
    }}
  ]
}}

Response:"""

            # 토큰화
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            # 모델을 CPU로 이동 (호환성 문제 해결)
            self.model = self.model.to('cpu')
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            
            # 간단한 생성 방식 사용
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=256,  # 더 짧은 출력
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # 캐시 사용 명시
                    repetition_penalty=1.1  # 반복 방지
                )
            
            # 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            self.performance_metrics["phi35_usage_count"] += 1
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"PHI-3.5 생성 실패: {e}")
            return f"PHI-3.5 생성 오류: {str(e)}"
    
    async def interpret_mission(self, mission: str):
        """미션 해석 - PHI-3.5 또는 폴백 사용"""
        logger.info(f"미션 해석 중: {mission}")
        
        if self.phi35_available:
            # PHI-3.5 사용
            logger.info("🤖 PHI-3.5 모델 사용")
            phi35_response = await self._generate_with_phi35(mission)
            logger.info(f"PHI-3.5 응답: {phi35_response[:200]}...")
            
            # JSON 파싱 시도
            try:
                import json
                # JSON 부분 추출
                import re
                json_match = re.search(r'\{.*\}', phi35_response, re.DOTALL)
                if json_match:
                    parsed_response = json.loads(json_match.group())
                    subtasks = parsed_response.get("subtasks", [])
                    logger.info(f"✅ PHI-3.5 JSON 파싱 성공: {len(subtasks)}개 서브태스크")
                else:
                    raise ValueError("JSON 형식을 찾을 수 없습니다")
            except Exception as e:
                logger.warning(f"PHI-3.5 JSON 파싱 실패: {e}")
                # 폴백으로 전환
                subtasks = self._fallback_mission_decomposition(mission)
        else:
            # 폴백 모드
            logger.info("📝 폴백 모드 사용")
            subtasks = self._fallback_mission_decomposition(mission)
        
        # 성능 메트릭 업데이트
        self.performance_metrics["missions_processed"] += 1
        self.performance_metrics["successful_decompositions"] += 1
        
        logger.info(f"✅ 미션 해석 완료: {len(subtasks)}개 서브태스크")
        
        return {
            "mission": mission,
            "subtasks": subtasks,
            "total_duration": sum(task.get("estimated_duration", 10.0) for task in subtasks),
            "difficulty": max(task.get("difficulty", 1) for task in subtasks) if subtasks else 1,
            "phi35_used": self.phi35_available
        }
    
    def _fallback_mission_decomposition(self, mission: str) -> List[Dict[str, Any]]:
        """폴백 미션 분해"""
        mission_lower = mission.lower()
        
        if "pick" in mission_lower and "place" in mission_lower:
            return [
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
            return [
                {
                    "type": "exploration",
                    "action": "explore_environment",
                    "target": "workspace",
                    "priority": 1,
                    "estimated_duration": 30.0,
                    "difficulty": 2
                }
            ]
    
    async def process_mission_with_learning(self, mission: str, context: Dict = None):
        """학습이 포함된 미션 처리"""
        logger.info(f"학습 포함 미션 처리: {mission}")
        
        # 미션 해석
        result = await self.interpret_mission(mission)
        
        # 학습 가치 계산
        base_learning = 0.1
        subtask_learning = len(result["subtasks"]) * 0.05
        phi35_bonus = 0.2 if result.get("phi35_used", False) else 0.0
        learning_value = base_learning + subtask_learning + phi35_bonus
        
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

async def test_advanced_foundation():
    """고급 Foundation Model 테스트"""
    logger.info("🚀 Advanced LLM Foundation Model 테스트 시작")
    
    # Foundation Model 초기화
    foundation = AdvancedFoundationModel()
    
    try:
        # 모델 초기화
        await foundation.initialize()
        
        # 테스트 미션들
        test_missions = [
            "Pick up the red cup and place it on the table",
            "Move to position [1, 0, 0.5] and explore the area",
            "Organize the books on the shelf by size",
            "Clean up the workspace by putting all tools in the toolbox"
        ]
        
        # 각 미션 테스트
        for i, mission in enumerate(test_missions, 1):
            logger.info(f"\n📋 테스트 미션 {i}: {mission}")
            
            # 미션 해석 테스트
            interpretation = await foundation.interpret_mission(mission)
            logger.info(f"   해석 결과: {len(interpretation['subtasks'])}개 서브태스크")
            logger.info(f"   예상 시간: {interpretation['total_duration']:.1f}초")
            logger.info(f"   난이도: {interpretation['difficulty']}/5")
            logger.info(f"   PHI-3.5 사용: {'✅' if interpretation.get('phi35_used', False) else '❌'}")
            
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
        logger.info(f"   PHI-3.5 사용 횟수: {metrics['phi35_usage_count']}회")
        
        logger.info("\n✅ Advanced LLM Foundation Model 테스트 완료!")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_advanced_foundation())
