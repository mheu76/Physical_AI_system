# 🤖 Dynamic AI Agent System - PHI-3.5 기반 에이전트 생성

**Physical AI Code에서 `/agent` 명령어로 AI 에이전트를 동적으로 생성하고 관리하세요!**

PHI-3.5 언어모델의 강력한 추론 능력을 활용하여, 사용자의 자연어 지시사항만으로 맞춤형 AI 에이전트를 실시간으로 생성하고 업데이트할 수 있습니다.

## 🚀 Quick Start

### 기본 사용법
```bash
# Physical AI Code 실행
python physical_ai_code.py

# 에이전트 생성
Physical AI> /agent create 테이블을 깨끗하게 정리하는 청소 전문 에이전트

# 에이전트 목록 확인
Physical AI> /agent list

# 에이전트 실행
Physical AI> /agent run CleaningSpecialist

# 에이전트 업데이트
Physical AI> /agent update CleaningSpecialist 더 꼼꼼하게 구석구석 청소하도록 개선해줘
```

## 🎯 주요 기능

### **1. 자연어 기반 에이전트 생성**
```bash
# 예시 1: 작업 자동화 에이전트
/agent create 매일 아침 커피를 준비하고 신문을 가져다주는 모닝 루틴 에이전트

🤖 새로운 에이전트 생성됨:
┌────────────┬──────────────────────────────────┐
│ 속성       │ 값                               │
├────────────┼──────────────────────────────────┤
│ ID         │ a1b2c3d4-e5f6-7890-abcd-1234567890│
│ 이름       │ Morning Routine Assistant        │
│ 설명       │ 매일 아침 커피 준비와 신문 배달을 담당│
│ 능력       │ Coffee Making, Newspaper Fetching │
│ 행동 패턴  │ 3개                              │
│ 태그       │ morning, automation, routine     │
│ 생성일     │ 2024-08-25T10:30:15             │
└────────────┴──────────────────────────────────┘
```

### **2. PHI-3.5 기반 지능형 설계**
PHI-3.5가 사용자 요청을 분석하여 최적의 에이전트를 자동 설계:

```json
{
  "agent_type": "automation",
  "name": "Morning Routine Assistant",
  "description": "매일 아침 커피 준비와 신문 배달을 자동화하는 에이전트",
  "capabilities": [
    {
      "name": "Coffee Preparation",
      "description": "커피머신을 사용하여 커피 제조",
      "tools_required": ["mission_executor", "hardware_status"],
      "parameters": {"coffee_type": "americano", "strength": "medium"}
    },
    {
      "name": "Newspaper Delivery", 
      "description": "현관문에서 신문을 가져와 테이블에 배치",
      "tools_required": ["mission_executor", "vision_system"],
      "parameters": {"delivery_location": "dining_table"}
    }
  ],
  "behaviors": [
    {
      "trigger": "morning_time",
      "action_sequence": [
        {"action": "check_coffee_machine", "parameters": {}},
        {"action": "make_coffee", "parameters": {"type": "americano"}},
        {"action": "fetch_newspaper", "parameters": {}},
        {"action": "place_on_table", "parameters": {"item": "newspaper"}}
      ],
      "priority": 1
    }
  ],
  "personality": {"efficient": 0.9, "reliable": 0.9, "friendly": 0.7},
  "tags": ["morning", "automation", "routine"]
}
```

### **3. 실시간 에이전트 업데이트**
```bash
# 에이전트 개선
Physical AI> /agent update Morning\ Routine\ Assistant 우유 거품도 만들어서 라테로 업그레이드하고, 날씨 정보도 알려주세요

🔄 에이전트 업데이트됨:
┌────────────┬──────────────────────────────────┐
│ 속성       │ 값                               │
├────────────┼──────────────────────────────────┤
│ 이름       │ Morning Routine Assistant        │
│ 업데이트일 │ 2024-08-25T10:45:22             │
│ 최근 업데이트│ 라테 제조 및 날씨 정보 제공 기능 추가│
└────────────┴──────────────────────────────────┘
```

## 📋 사용 가능한 명령어

### **에이전트 생성**
```bash
# 기본 생성
/agent create <지시사항>

# 예시들
/agent create 창고에서 재고를 자동으로 정리하는 에이전트
/agent create 방문자를 맞이하고 안내하는 접수 도우미
/agent create 위험 상황을 감지하고 알려주는 보안 모니터링 에이전트
```

### **에이전트 관리**
```bash
# 목록 보기
/agent list
/agent ls

# 상세 정보
/agent info <에이전트명>
/agent show <에이전트명>

# 실행
/agent run <에이전트명>  
/agent execute <에이전트명>

# 업데이트
/agent update <에이전트명> <업데이트 지시사항>

# 삭제
/agent delete <에이전트명>
/agent remove <에이전트명>
```

## 🎨 에이전트 타입 및 템플릿

### **1. Assistant (어시스턴트)**
```bash
/agent create 손님을 응대하고 주문을 받는 카페 직원 도우미

# 생성되는 특징:
- 친화적이고 도움을 주는 성격
- 대화 및 작업 계획 능력 강화
- 사용자 상호작용 최적화
```

### **2. Specialist (전문가)**
```bash  
/agent create 전자기기 고장을 진단하고 수리하는 기술 전문가

# 생성되는 특징:
- 분석적이고 정확한 성격
- 특정 영역 전문 지식 활용
- 체계적인 문제 해결 접근
```

### **3. Automation (자동화)**
```bash
/agent create 매시간 온도를 체크하고 에어컨을 조절하는 환경 관리 시스템

# 생성되는 특징:
- 효율적이고 일관된 성격
- 반복 작업 및 모니터링 특화
- 규칙 기반 자동 실행
```

### **4. Learning (학습)**
```bash
/agent create 사용자의 습관을 학습하고 개인화된 서비스를 제공하는 적응형 도우미

# 생성되는 특징:
- 호기심 많고 적응적인 성격
- 지속적인 학습 및 개선
- 패턴 인식 및 개인화
```

## 🔄 에이전트 실행 예시

### **대화형 실행**
```bash
Physical AI> /agent run Morning\ Routine\ Assistant

🚀 에이전트 'Morning Routine Assistant' 실행 완료:
┌──────────┬─────┐
│ 메트릭   │ 값  │
├──────────┼─────┤
│ 실행 횟수│ 5   │
│ 성공률   │ 96% │
│ 실행된 행동│ 3   │
└──────────┴─────┘

실행 결과:
  1. morning_time: ✅ 성공
  2. coffee_preparation: ✅ 성공  
  3. newspaper_delivery: ✅ 성공
```

### **자연어로 에이전트 호출**
```bash
Physical AI> 모닝 루틴 에이전트를 실행해줘

🤖 AI 응답:
Morning Routine Assistant 에이전트를 실행하겠습니다.

[에이전트 실행 중...]
✅ 커피 준비 완료 (아메리카노)
✅ 신문 배달 완료 (테이블 위에 배치)
✅ 날씨 정보 확인 완료 (오늘은 맑음, 23도)

모든 아침 루틴이 성공적으로 완료되었습니다!
```

## 🧠 PHI-3.5 Integration

### **지능형 에이전트 설계**
PHI-3.5가 다음과 같이 분석하고 설계합니다:

1. **요구사항 분석**: 사용자 지시사항에서 핵심 기능 추출
2. **능력 매핑**: 필요한 도구와 기능을 Physical AI 시스템에 매핑
3. **행동 패턴 생성**: 논리적이고 효율적인 작업 순서 구성
4. **개성 부여**: 작업 특성에 맞는 에이전트 성격 설정

### **지속적 학습 및 개선**
```bash
# 에이전트 성능 분석 기반 자동 개선
Physical AI> /agent info Morning\ Routine\ Assistant

🤖 에이전트 상세 정보:
...
성공률: 96%

업데이트 히스토리:
- 2024-08-25 10:45: 라테 제조 및 날씨 정보 기능 추가
- 2024-08-25 11:20: 커피 온도 최적화 (75도 → 70도)
- 2024-08-25 12:10: 신문 배치 위치 개선 (테이블 중앙)
```

## 🎯 실제 사용 시나리오

### **시나리오 1: 개인 비서 에이전트**
```bash
Physical AI> /agent create 매일의 일정을 확인하고 중요한 약속을 알려주며, 필요한 준비물을 챙겨주는 개인 비서

🤖 Personal Secretary Assistant 생성됨!
💪 주요 능력:
- Schedule Management: 일정 확인 및 알림
- Reminder System: 중요 약속 사전 알림  
- Preparation Assistant: 준비물 체크리스트 제공

Physical AI> /agent run Personal\ Secretary\ Assistant

📋 오늘의 일정:
- 09:00: 팀 회의 (회의실 A) - 노트북, 자료 필요
- 14:00: 고객 미팅 (카페) - 계약서, 명함 필요
- 18:00: 저녁 약속 (레스토랑) - 예약 확인됨

✅ 모든 준비물이 체크되었습니다!
```

### **시나리오 2: 보안 모니터링 에이전트**
```bash
Physical AI> /agent create 사무실 보안을 24시간 모니터링하고 이상 상황 발생시 즉시 알려주는 보안 관제 에이전트

🤖 Security Monitor Agent 생성됨!
🎯 주요 행동:
- 실시간 카메라 모니터링
- 움직임 감지 및 분석
- 비상시 알림 및 대응

# 자동 실행 (24시간 모니터링)
🔍 보안 모니터링 중...
- 21:35: 정상 - 사무실 내 이상 없음
- 21:36: 감지 - 창문 근처 움직임 (고양이로 판단)
- 21:37: 정상 - 모든 구역 안전
```

### **시나리오 3: 학습 도우미 에이전트**
```bash
Physical AI> /agent create 사용자의 로봇 조작 패턴을 학습하고 더 나은 방법을 제안하는 스마트 튜터

🤖 Smart Tutor Agent 생성됨!

# 에이전트 업데이트 (사용자 패턴 학습 후)
Physical AI> /agent update Smart\ Tutor 사용자가 물건을 집을 때 성공률이 낮으니 그립 각도를 조정하는 팁을 제공해줘

🔄 에이전트 업데이트됨: 그립 최적화 조언 기능 추가

Physical AI> /agent run Smart\ Tutor

💡 튜터 조언:
- 현재 그립 성공률: 78%
- 권장 개선사항: 접근 각도를 15도 더 기울여보세요
- 예상 개선 효과: +12% 성공률 향상
```

## 🛠️ 개발자 가이드

### **커스텀 에이전트 타입 추가**
```python
# agent_system.py에서 새로운 템플릿 추가
TEMPLATES = {
    "custom_type": {
        "name_pattern": "{domain} Specialist",
        "description_pattern": "{domain} 분야의 맞춤형 전문가", 
        "base_capabilities": ["custom_analysis", "specialized_execution"],
        "personality_traits": {"innovative": 0.9, "adaptable": 0.8}
    }
}
```

### **PHI-3.5 프롬프트 커스터마이징**
```python
# 에이전트 생성 프롬프트 수정
self.agent_generation_prompt = """
당신은 {domain}의 전문 에이전트 아키텍트입니다.
특별히 {특화_영역}에 집중하여 에이전트를 설계해주세요.
...
"""
```

## 📊 성능 및 통계

### **에이전트 성능 추적**
```bash
Physical AI> /agent info Morning\ Routine\ Assistant

📈 성능 메트릭:
- 총 실행 횟수: 47회
- 평균 성공률: 94.3%
- 평균 실행 시간: 2분 35초
- 마지막 실행: 2024-08-25 08:30:15

🔄 업데이트 히스토리:
- 총 7회 업데이트
- 마지막 업데이트: PHI-3.5 기반 성능 최적화
- 개선된 기능: 날씨 API 연동, 커피 온도 조절
```

## 🌟 혁신적인 특징들

### **1. 자연어 → 코드 자동 변환**
사용자의 한국어 설명이 PHI-3.5를 통해 실행 가능한 에이전트 코드로 자동 변환됩니다.

### **2. 지속적 학습 시스템** 
에이전트 실행 결과를 바탕으로 PHI-3.5가 자동으로 개선점을 분석하고 업데이트를 제안합니다.

### **3. 다중 에이전트 협업**
```bash
# 여러 에이전트가 협력하여 복잡한 작업 수행
Physical AI> /agent create 청소 에이전트와 협력하여 파티 준비를 담당하는 이벤트 기획자

🤖 Event Planner Agent 생성됨!
🤝 협업 에이전트: Cleaning Specialist, Kitchen Assistant
```

### **4. 에이전트 생태계**
생성된 에이전트들이 서로의 능력을 활용하여 더 복잡하고 정교한 작업을 수행할 수 있습니다.

---

## 🚀 시작해보기

```bash
# 1. Physical AI Code 실행
python physical_ai_code.py

# 2. 첫 번째 에이전트 생성
Physical AI> /agent create 나만의 개인 비서 에이전트

# 3. 에이전트와 대화하기
Physical AI> 개인 비서 에이전트를 실행해줘

# 4. 더 많은 에이전트 만들기
Physical AI> /agent create 집안일을 도와주는 가사 도우미
```

**PHI-3.5의 강력한 언어 이해 능력과 Physical AI의 실제 작업 실행 능력이 결합된 혁신적인 에이전트 생성 시스템을 경험해보세요!** 🤖✨

---

*"상상하는 모든 AI 에이전트를 자연어 한 문장으로 만들어보세요!"*