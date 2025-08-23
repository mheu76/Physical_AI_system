# Contributing to Physical AI System

🤖 물리적 AI 시스템 프로젝트에 기여해주셔서 감사합니다!

## 기여 방법

### 1. 이슈 제출
버그 발견이나 새로운 기능 제안이 있으시면 GitHub Issues를 통해 알려주세요.

- **버그 리포트**: 재현 가능한 단계와 환경 정보 포함
- **기능 제안**: 사용 사례와 예상되는 이점 설명
- **문서 개선**: 불명확한 부분이나 누락된 정보 지적

### 2. 코드 기여

#### 개발 환경 설정
```bash
# 프로젝트 클론
git clone https://github.com/your-username/physical-ai-system.git
cd physical-ai-system

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 개발용 도구 설치
pip install -r requirements-dev.txt
```

#### 코드 스타일
- **코드 포맷팅**: Black 사용
- **Import 정렬**: isort 사용
- **타입 힌팅**: mypy로 타입 검사
- **문서화**: Google 스타일 독스트링

```bash
# 코드 포맷팅 실행
black .
isort .

# 타입 체크
mypy .

# 테스트 실행
pytest tests/
```

#### Pull Request 가이드라인
1. **브랜치 생성**: `feature/기능명` 또는 `fix/버그명`
2. **작은 단위 커밋**: 각 커밋은 하나의 논리적 변경사항
3. **명확한 커밋 메시지**: 변경사항을 간결하게 설명
4. **테스트 추가**: 새로운 기능에는 테스트 코드 포함
5. **문서 업데이트**: 필요시 README나 문서 수정

### 3. 특별 관심 분야

#### 🧠 Foundation Model 개선
- 실제 LLM 통합 (GPT, LLaMA 등)
- 물리 추론 알고리즘 향상
- 자연어 처리 정확도 개선

#### 🌱 Developmental Learning 확장
- 새로운 학습 알고리즘 구현
- 메타러닝 기법 추가
- 커리큘럼 학습 최적화

#### ⚡ Real-time Execution 최적화
- 제어 알고리즘 성능 향상
- 안전 시스템 강화
- 실시간 성능 튜닝

#### 🔌 Hardware Integration 확장
- 새로운 센서 드라이버 추가
- 로봇 플랫폼 지원 확대
- ROS2 통합 개선

### 4. 코드 리뷰 프로세스

모든 Pull Request는 다음 기준으로 검토됩니다:

- **기능성**: 의도한 대로 작동하는가?
- **코드 품질**: 가독성, 유지보수성이 좋은가?
- **테스트**: 충분한 테스트 커버리지를 갖고 있는가?
- **문서화**: 필요한 문서가 포함되어 있는가?
- **호환성**: 기존 코드와 잘 호환되는가?

### 5. 행동 강령

- 존중하는 태도로 소통
- 건설적인 피드백 제공
- 다양한 관점과 경험 수용
- 학습과 성장을 위한 환경 조성

### 6. 질문이나 도움이 필요하신가요?

- GitHub Discussions에서 토론
- Issues에서 질문 태그 사용
- 이메일: [maintainer@physical-ai.dev]

## 감사 인사

Physical AI System은 커뮤니티의 기여로 발전합니다. 모든 기여자들에게 진심으로 감사드립니다! 🙏

---

**"The future of AI is not just digital, it's physical."** 🤖✨