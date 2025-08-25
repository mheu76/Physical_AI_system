# 🚀 Physical AI Code - 빠른 시작 가이드

## 5분만에 시작하기

### 1️⃣ 설치 (1분)
```bash
git clone https://github.com/mheu76/Physical_AI_system.git
cd Physical_AI_system
pip install -r requirements.txt
```

### 2️⃣ 테스트 (1분)
```bash
python test_basic_functionality.py
```
✅ "모든 테스트 통과!" 메시지 확인

### 3️⃣ 실행 (1분)
```bash
python physical_ai_code.py
```

### 4️⃣ 기본 명령어 사용해보기 (2분)

**시스템 상태 확인**
```
/status
```

**도구 목록 보기**
```
/tools
```

**AI 에이전트 생성하기**
```
/agent create 물체를 정리하는 도우미
```

**자연어 명령 실행**
```
안녕하세요! 시스템 테스트를 해주세요.
```

## 🎯 주요 기능 한눈에 보기

| 기능 | 명령어 | 설명 |
|------|--------|------|
| 🤖 에이전트 생성 | `/agent create [설명]` | PHI-3.5로 AI 에이전트 자동 생성 |
| 📋 상태 확인 | `/status` | 시스템 상태 실시간 확인 |
| 🛠️ 도구 목록 | `/tools` | 6개 핵심 도구 확인 |
| 💬 자연어 실행 | `[자유로운 문장]` | 한국어/영어 자연어 명령 |
| 📊 에이전트 관리 | `/agent list/info/delete` | 생성된 에이전트 관리 |

## 🔧 문제가 있다면?

**"제한된 모드" 메시지가 나와요**
→ 정상입니다! 실제 하드웨어 없이도 모든 기능 사용 가능

**한글이 깨져요**
→ `configs/default.yaml`에서 `language: "en"` 으로 변경

**더 자세한 설명이 필요해요**
→ `사용설명서.md` 또는 `USER_MANUAL.md` 참조

---

## 🎉 축하합니다!

Physical AI Code 시스템을 성공적으로 실행했습니다!
이제 자연어로 로봇을 제어하고 AI 에이전트를 만들어보세요.

**다음 단계**: `사용설명서.md`에서 고급 기능을 알아보세요!