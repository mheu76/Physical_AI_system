# Physical AI Code - Unified Interface 🤖✨

**Claude Code 스타일의 Physical AI 통합 개발 환경**

Physical AI System의 모든 기능을 하나의 일관된 대화형 인터페이스에서 사용할 수 있는 혁신적인 통합 환경입니다.

## 🚀 Quick Start

### 설치 및 실행
```bash
# 필수 의존성 설치 (Rich UI 라이브러리)
pip install rich

# 통합 인터페이스 실행
python physical_ai_code.py

# 단일 미션 실행 모드
python physical_ai_code.py --mission "로봇아, 빨간 컵을 집어서 테이블에 놓아줘"

# 사용자 설정 파일 사용
python physical_ai_code.py --config configs/modular.yaml --debug
```

### 첫 실행 화면
```
🤖 Physical AI Code - 통합 개발 환경

Claude Code 스타일의 Physical AI 시스템 인터페이스
자연어로 로봇을 제어하고, 학습하고, 시뮬레이션하세요!

✅ 시스템 준비 완료!

Physical AI> 
```

## 💬 사용법

### 1. **자연어 대화 방식**
```
Physical AI> 로봇아, 빨간 공을 상자에 넣어줘

🤖 AI 응답:
미션 '빨간 공을 상자에 넣어줘'를 분석하고 실행 계획을 수립하겠습니다.

사용된 도구:
┌────────────────┬──────────┐
│ 도구명         │ 결과     │
├────────────────┼──────────┤
│ mission_executor│ Complete │
└────────────────┴──────────┘
```

### 2. **슬래시 명령어 방식**
```
Physical AI> /mission 테이블 위의 컵을 정리해줘
✅ 미션 '테이블 위의 컵을 정리해줘' 실행 결과:
  Mission '테이블 위의 컵을 정리해줘' queued for execution

Physical AI> /learn grasp_skill
✅ 'grasp_skill' 학습 결과:
  Learning session for 'grasp_skill' initiated

Physical AI> /hardware
📊 시스템 상태:
┌──────────────┬─────────────┐
│ 항목         │ 상태        │  
├──────────────┼─────────────┤
│ 초기화 완료  │ ✅          │
│ Physical AI  │ ✅          │
│ 사용 가능한 도구│ 5        │
│ 세션 활성    │ ✅          │
└──────────────┴─────────────┘
```

## 🛠️ 사용 가능한 명령어

### **미션 실행**
```bash
/mission <작업>           # 물리적 미션 실행
```
**자연어 예시:**
- "로봇아, 빨간 컵을 테이블로 옮겨줘"
- "상자 안의 물건들을 정리해줘"  
- "책을 책장에 꽂아줘"

### **학습 시스템**
```bash
/learn <기술>             # 새로운 기술 학습
```
**자연어 예시:**
- "새로운 잡기 동작을 학습해줘"
- "더 정확하게 물건을 놓는 방법을 연습해줘"
- "장애물 회피 스킬을 개선해줘"

### **하드웨어 제어**
```bash
/hardware [컴포넌트]      # 하드웨어 상태 확인
```
**자연어 예시:**
- "로봇 상태를 확인해줘"
- "센서들이 제대로 작동하나?"
- "관절 상태는 어때?"

### **시뮬레이션**
```bash
/simulate [시나리오]      # 물리 시뮬레이션 실행
```
**자연어 예시:**
- "시뮬레이션에서 테스트해보자"
- "가상 환경에서 연습해줘"
- "안전한 환경에서 먼저 해봐"

### **컴퓨터 비전**
```bash
/vision [작업]            # 비전 시스템 작업
```
**자연어 예시:**
- "무엇이 보이나?"
- "빨간 물체를 찾아줘"
- "테이블 위에 뭐가 있지?"

### **시스템 관리**
```bash
/status                   # 시스템 상태 확인
/tools                    # 사용 가능한 도구 목록
/help [주제]             # 도움말
/quit                     # 종료
```

## 🎯 주요 특징

### **1. 통합된 도구 시스템**
- **MissionExecutorTool**: 물리적 미션 실행
- **LearningTool**: 발달적 학습 시스템
- **HardwareStatusTool**: 하드웨어 모니터링
- **SimulationTool**: 물리 시뮬레이션
- **VisionTool**: 컴퓨터 비전 처리

### **2. 지능형 명령 처리**
```python
# 자연어 → 구조화된 명령으로 자동 변환
"빨간 컵을 테이블로 옮겨줘" 
→ {
    "type": "mission",
    "intent": "move_object", 
    "entities": {
        "objects": ["빨간 컵"],
        "surfaces": ["테이블"]
    },
    "tools": [{"name": "mission_executor", ...}]
}
```

### **3. 세션 관리**
- 대화 히스토리 저장
- 컨텍스트 유지
- 세션 복구 가능
- 다중 세션 지원

### **4. Rich 터미널 UI**
- 아름다운 표와 패널
- 실시간 상태 표시
- 진행 상황 표시
- 컬러풀한 출력

## 🏗️ 아키텍처

```
┌─────────────────────────────────────┐
│        CLI Interface                │
│  (Rich Terminal UI + Natural NLP)   │
├─────────────────────────────────────┤
│      Interface Manager              │
│   (Session + Conversation)          │
├─────────────────────────────────────┤
│     Command Processor               │
│  (NLP Analysis + Intent Recognition)│
├─────────────────────────────────────┤
│        Tool System                  │
│  (Physical AI Functions as Tools)   │
├─────────────────────────────────────┤
│      Physical AI System             │
│ (Foundation + Learning + Execution) │
└─────────────────────────────────────┘
```

### **핵심 컴포넌트**

#### **InterfaceManager**
```python
# 메인 통합 인터페이스 관리
await interface.process_input("로봇아, 컵을 옮겨줘")
→ 자연어 분석 → 도구 실행 → PHI-3.5 응답 생성
```

#### **CommandProcessor** 
```python
# 자연어 → 구조화된 명령 변환
"새로운 잡기 기술을 학습해줘"
→ {"type": "learning", "intent": "learn_skill", ...}
```

#### **ToolSystem**
```python
# 모든 Physical AI 기능을 도구로 추상화
tools = ["mission_executor", "learning_system", 
         "hardware_status", "physics_simulation", "vision_system"]
```

#### **SessionManager**
```python
# 대화 세션 관리 및 히스토리 유지
session.conversation_history = [user_input, ai_response, ...]
```

## 🎨 사용자 경험

### **대화 예시 1: 물리적 미션**
```
👤 로봇아, 테이블 위의 빨간 컵을 싱크대로 옮겨줘

🤖 네, 테이블 위의 빨간 컵을 싱크대로 옮기는 작업을 시작하겠습니다.

[도구 실행 중: mission_executor]
┌────────────────────────────┬──────────┐
│ 1. 객체 인식 (빨간 컵)     │ ✅ 완료  │
│ 2. 경로 계획 수립          │ ✅ 완료  │  
│ 3. 안전성 검사             │ ✅ 완료  │
│ 4. 물리적 이동 실행        │ 🔄 진행중│
└────────────────────────────┴──────────┘

✅ 빨간 컵을 성공적으로 싱크대로 옮겼습니다!
```

### **대화 예시 2: 학습 시스템**
```
👤 더 정확하게 물건을 잡는 방법을 배워줘

🤖 정밀한 그립 기술을 개선하는 학습을 시작하겠습니다.

[학습 진행 상황]
기술: precision_grasp
📊 진행도: ████████░░ 80% (8/10 반복)
📈 성공률: 85% → 92% (개선됨!)
⏱️  소요 시간: 2분 35초

✅ 그립 정확도가 7% 향상되었습니다!
```

### **대화 예시 3: 시스템 통합**
```
👤 시뮬레이션에서 먼저 테스트한 다음 실제로 해줘

🤖 먼저 가상 환경에서 시뮬레이션을 진행한 후 실제 작업을 수행하겠습니다.

[1단계: 시뮬레이션]
🎮 물리 시뮬레이션 시작...
✅ 가상 환경에서 성공률: 95%
✅ 충돌 위험: 없음
✅ 예상 실행 시간: 45초

[2단계: 실제 실행]  
🤖 시뮬레이션 결과가 좋습니다. 실제 작업을 시작합니다...
✅ 작업 완료!
```

## 🔧 개발자 가이드

### **새로운 도구 추가**
```python
class CustomTool(PhysicalAITool):
    def __init__(self):
        super().__init__("custom_tool", "사용자 정의 도구")
        self.add_parameter(ToolParameter(
            name="param", type="string", description="매개변수"
        ))
    
    async def execute(self, **kwargs) -> ToolResult:
        # 도구 로직 구현
        return ToolResult(success=True, result="완료")

# 도구 등록
tool_system.register_tool(CustomTool())
```

### **명령 패턴 확장**
```python
# command_processor.py에서 패턴 추가
"custom": [
    {
        "pattern": r"사용자정의\s+(.+)",
        "intent": "custom_action",
        "confidence": 0.9
    }
]
```

## 🚀 향후 계획

### **단기 목표 (1개월)**
- [ ] 음성 인식/합성 통합 
- [ ] 웹 인터페이스 추가
- [ ] 모바일 앱 연동
- [ ] 다국어 지원 확장

### **중기 목표 (3개월)**
- [ ] AR/VR 인터페이스
- [ ] 제스처 인식
- [ ] 감정 인식 및 반응
- [ ] 다중 로봇 협업

### **장기 목표 (6개월)**
- [ ] 클라우드 서비스 통합
- [ ] AI 어시스턴트 생태계 구축
- [ ] 산업용 배포 솔루션
- [ ] 커뮤니티 플랫폼

---

## 🤝 기여하기

Physical AI Code는 오픈소스 프로젝트입니다!

- 🐛 **버그 리포트**: GitHub Issues에 등록
- 💡 **새 기능**: Pull Request로 제안  
- 📖 **문서**: 번역 및 개선
- ⭐ **별점**: 도움이 되었다면 Star 클릭!

**Physical AI Code**로 로봇과의 자연스러운 대화를 경험해보세요! 🤖✨