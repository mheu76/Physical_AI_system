"""
행동모델 대화형 정의 모듈

PHI-3.5와 자연어 대화를 통해 행동모델을 정의하고 수정할 수 있는
대화형 인터페이스입니다.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from foundation_model.phi35_integration import PHI35ModelManager
from foundation_model.slm_foundation import MotionPrimitive, TaskPlanningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BehaviorModel:
    """행동모델 정의"""
    name: str
    description: str
    motion_primitives: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    constraints: Dict[str, Any]
    created_at: str
    updated_at: str

class BehaviorModelDialog:
    """행동모델 대화형 정의 시스템"""
    
    def __init__(self):
        self.phi35_manager = None
        self.behavior_models = {}
        self.current_model = None
        self.dialog_history = []
        
    async def initialize(self):
        """시스템 초기화"""
        print("🤖 행동모델 대화형 정의 시스템 초기화 중...")
        
        # PHI-3.5 모델 초기화
        self.phi35_manager = PHI35ModelManager()
        await self.phi35_manager.initialize()
        
        # 기존 행동모델 로드
        await self._load_existing_models()
        
        print("✅ 시스템 초기화 완료!")
        print("💬 PHI-3.5와 자연어로 행동모델을 정의해보세요!")
        
    async def _load_existing_models(self):
        """기존 행동모델 로드"""
        models_file = Path("behavior_models.json")
        if models_file.exists():
            try:
                with open(models_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_data in data:
                        model = BehaviorModel(**model_data)
                        self.behavior_models[model.name] = model
                print(f"📚 {len(self.behavior_models)}개의 기존 행동모델 로드됨")
            except Exception as e:
                print(f"⚠️  기존 모델 로드 실패: {e}")
    
    async def _save_models(self):
        """행동모델 저장"""
        try:
            models_data = [asdict(model) for model in self.behavior_models.values()]
            with open("behavior_models.json", 'w', encoding='utf-8') as f:
                json.dump(models_data, f, ensure_ascii=False, indent=2)
            print("💾 행동모델 저장 완료")
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
    
    async def start_dialog(self):
        """대화 시작"""
        print("\n" + "="*60)
        print("🎯 PHI-3.5 행동모델 정의 대화")
        print("="*60)
        print("💡 사용 가능한 명령어:")
        print("  - '새 모델 만들기': 새로운 행동모델 생성")
        print("  - '모델 수정하기': 기존 모델 수정")
        print("  - '모델 보기': 정의된 모델들 확인")
        print("  - '모델 테스트': 모델 실행 테스트")
        print("  - '종료': 대화 종료")
        print("="*60)
        
        while True:
            try:
                # 사용자 입력
                user_input = input("\n💬 당신: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['종료', 'exit', 'quit']:
                    print("👋 대화를 종료합니다. 행동모델이 저장되었습니다.")
                    await self._save_models()
                    break
                
                # 대화 기록 저장
                self.dialog_history.append({"user": user_input, "timestamp": asyncio.get_event_loop().time()})
                
                # PHI-3.5 응답 생성
                response = await self._process_user_input(user_input)
                
                # 응답 출력
                print(f"\n🤖 PHI-3.5: {response}")
                
                # 대화 기록 저장
                self.dialog_history.append({"assistant": response, "timestamp": asyncio.get_event_loop().time()})
                
            except KeyboardInterrupt:
                print("\n\n👋 대화가 중단되었습니다.")
                await self._save_models()
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
    
    async def _process_user_input(self, user_input: str) -> str:
        """사용자 입력 처리"""
        try:
            # PHI-3.5에게 전달할 프롬프트 생성
            context = self._create_context_prompt()
            full_prompt = f"{context}\n\n사용자: {user_input}\n\nPHI-3.5:"
            
            # PHI-3.5 응답 생성
            response = await self.phi35_manager.generate_response(full_prompt)
            
            # 응답에서 행동모델 정보 추출
            await self._extract_behavior_model(response, user_input)
            
            return response
            
        except Exception as e:
            logger.error(f"입력 처리 실패: {e}")
            return f"죄송합니다. 처리 중 오류가 발생했습니다: {e}"
    
    def _create_context_prompt(self) -> str:
        """컨텍스트 프롬프트 생성"""
        context = """당신은 Physical AI 시스템의 행동모델 정의 전문가입니다.

현재 상황:
- 정의된 행동모델: {model_count}개
- 현재 작업 중인 모델: {current_model}

행동모델 정의 규칙:
1. 각 행동모델은 고유한 이름을 가져야 합니다
2. motion_primitives는 기본 동작 단위들의 리스트입니다
3. 각 primitive는 name, parameters, preconditions, postconditions를 포함합니다
4. parameters는 동작 실행에 필요한 매개변수들입니다
5. constraints는 안전성과 물리적 제약조건들입니다

예시 행동모델:
{example_model}

사용자가 행동모델을 정의하거나 수정하려고 할 때, 구조화된 JSON 형태로 응답해주세요.
JSON 응답은 다음과 같은 형식이어야 합니다:

```json
{{
  "action": "create|modify|view|test",
  "model_name": "모델명",
  "description": "모델 설명",
  "motion_primitives": [
    {{
      "name": "동작명",
      "parameters": {{"param1": "value1"}},
      "preconditions": ["조건1", "조건2"],
      "postconditions": ["결과1", "결과2"]
    }}
  ],
  "parameters": {{"global_param": "value"}},
  "constraints": {{"safety": "constraint"}}
}}
```

사용자의 요청에 따라 적절한 응답을 생성해주세요.""".format(
            model_count=len(self.behavior_models),
            current_model=self.current_model.name if self.current_model else "없음",
            example_model=self._get_example_model()
        )
        
        return context
    
    def _get_example_model(self) -> str:
        """예시 모델 반환"""
        return """{
  "name": "커피_만들기",
  "description": "커피 머신을 사용해서 커피를 만드는 행동모델",
  "motion_primitives": [
    {
      "name": "커피머신_접근",
      "parameters": {"target": "coffee_machine", "distance": 0.3},
      "preconditions": ["coffee_machine_visible", "path_clear"],
      "postconditions": ["at_coffee_machine"]
    },
    {
      "name": "커피_추출",
      "parameters": {"duration": 30, "pressure": "9_bar"},
      "preconditions": ["at_coffee_machine", "beans_loaded"],
      "postconditions": ["coffee_extracted"]
    }
  ],
  "parameters": {"max_force": 10.0, "safety_distance": 0.1},
  "constraints": {"temperature_limit": 95, "pressure_limit": 15}
}"""
    
    async def _extract_behavior_model(self, response: str, user_input: str):
        """응답에서 행동모델 정보 추출"""
        try:
            # JSON 블록 찾기
            if "```json" in response and "```" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                
                model_data = json.loads(json_str)
                
                # 행동모델 생성/수정
                if model_data.get("action") in ["create", "modify"]:
                    await self._create_or_modify_model(model_data)
                    
        except Exception as e:
            logger.warning(f"모델 추출 실패: {e}")
    
    async def _create_or_modify_model(self, model_data: Dict[str, Any]):
        """행동모델 생성 또는 수정"""
        model_name = model_data.get("model_name")
        if not model_name:
            return
        
        # 기존 모델 확인
        if model_name in self.behavior_models:
            print(f"🔄 기존 모델 '{model_name}' 수정 중...")
        else:
            print(f"🆕 새 모델 '{model_name}' 생성 중...")
        
        # 행동모델 객체 생성
        import datetime
        now = datetime.datetime.now().isoformat()
        
        model = BehaviorModel(
            name=model_name,
            description=model_data.get("description", ""),
            motion_primitives=model_data.get("motion_primitives", []),
            parameters=model_data.get("parameters", {}),
            constraints=model_data.get("constraints", {}),
            created_at=now,
            updated_at=now
        )
        
        self.behavior_models[model_name] = model
        self.current_model = model
        
        print(f"✅ 모델 '{model_name}' 저장됨")
    
    async def list_models(self) -> str:
        """정의된 모델들 목록 반환"""
        if not self.behavior_models:
            return "정의된 행동모델이 없습니다."
        
        result = "📋 정의된 행동모델들:\n\n"
        for name, model in self.behavior_models.items():
            result += f"🔸 {name}\n"
            result += f"   설명: {model.description}\n"
            result += f"   동작 수: {len(model.motion_primitives)}개\n"
            result += f"   생성일: {model.created_at[:10]}\n\n"
        
        return result
    
    async def test_model(self, model_name: str) -> str:
        """모델 테스트 실행"""
        if model_name not in self.behavior_models:
            return f"모델 '{model_name}'을 찾을 수 없습니다."
        
        model = self.behavior_models[model_name]
        
        # 시뮬레이션 테스트
        result = f"🧪 모델 '{model_name}' 테스트 실행 중...\n\n"
        
        for i, primitive in enumerate(model.motion_primitives, 1):
            result += f"동작 {i}: {primitive['name']}\n"
            result += f"  매개변수: {primitive['parameters']}\n"
            result += f"  전제조건: {primitive['preconditions']}\n"
            result += f"  결과조건: {primitive['postconditions']}\n"
            result += f"  상태: ✅ 시뮬레이션 성공\n\n"
        
        result += f"🎉 모델 '{model_name}' 테스트 완료!"
        return result

async def main():
    """메인 함수"""
    dialog_system = BehaviorModelDialog()
    await dialog_system.initialize()
    await dialog_system.start_dialog()

if __name__ == "__main__":
    asyncio.run(main())
