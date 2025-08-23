# Changelog

Physical AI System 프로젝트의 모든 주요 변경사항이 이 파일에 문서화됩니다.

이 프로젝트는 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)을 따릅니다.

## [Unreleased]

### Added
- 새로운 기능들이 여기에 추가됩니다

### Changed
- 기존 기능의 변경사항들이 여기에 기록됩니다

### Deprecated
- 곧 제거될 기능들이 여기에 나열됩니다

### Removed
- 제거된 기능들이 여기에 기록됩니다

### Fixed
- 버그 수정사항들이 여기에 기록됩니다

### Security
- 보안 관련 변경사항들이 여기에 기록됩니다

## [1.0.0] - 2025-08-23

### Added
- 🧠 **sLM Foundation Model**: 자연어 미션 해석 및 태스크 계획 수립
  - 미션을 서브태스크로 분해하는 TaskPlanningModule
  - 물리 법칙 기반 동작 최적화하는 MotionReasoningModule
  - 에너지 효율성 계산 시스템
  
- 🌱 **Developmental Learning Engine**: 발달적 학습 시스템
  - 점진적 스킬 습득 엔진 (SkillAcquisitionEngine)
  - 에피소드/의미 메모리 관리 시스템 (MemoryManagement)
  - 커리큘럼 학습 시스템 (CurriculumLearning)
  - 자율적 탐색 및 학습 기능

- ⚡ **AI Agent Executor**: 실시간 물리적 실행 레이어
  - 고정밀 동작 제어 시스템 (MotionController)
  - 실시간 안전 모니터링 (SafetyMonitor)
  - 성능 지표 계산 및 학습 가치 평가
  - 백그라운드 안전 모니터링 루프

- 🔌 **Hardware Abstraction Layer**: 하드웨어 추상화 시스템
  - 다중 센서 인터페이스 (Vision, Tactile, IMU)
  - 액추에이터 제어 인터페이스 (ServoMotor, Gripper)
  - 센서 융합 알고리즘 (SensorFusion)
  - 자동 하드웨어 캘리브레이션

- 📋 **Configuration System**: 완전한 설정 관리
  - YAML 기반 설정 파일 (default.yaml)
  - 시스템, 모델, 학습, 실행, 하드웨어 별 세부 설정
  - 개발/테스트 모드 지원

- 🧪 **Examples & Tests**: 실행 가능한 예제 및 테스트
  - 기본 사용법 예제 (basic_example.py)
  - 발달적 학습 시연 예제 (developmental_learning_example.py)
  - 통합 테스트 시스템 (test_integration.py)

- 📦 **Development Infrastructure**: 개발 환경 지원
  - Docker 컨테이너화 지원
  - GitHub Actions CI/CD 파이프라인
  - 코드 품질 도구 (Black, Flake8, MyPy)
  - Pre-commit hooks

### Technical Details
- **Architecture**: 4-layer modular design with async/await pattern
- **Language**: Python 3.8+ with type hints
- **Dependencies**: PyTorch, NumPy, OpenCV, PyBullet, Redis
- **Testing**: pytest with asyncio support
- **Documentation**: Comprehensive README and API documentation

### Features Highlights
- 🤖 **자율적 스킬 습득**: AI가 스스로 새로운 동작을 학습
- 🔄 **지속적 개선**: 경험을 통한 성능 향상
- 🛡️ **실시간 안전 시스템**: 충돌 감지 및 비상 정지
- 🔧 **하드웨어 호환성**: 다양한 로봇 플랫폼 지원
- 📊 **모니터링**: Prometheus 기반 시스템 모니터링

## Contributing

변경사항을 기록할 때는 다음 가이드라인을 따라주세요:

- 변경 유형에 따라 적절한 섹션에 추가
- 명확하고 구체적인 설명 작성
- 관련 이슈나 PR 번호 포함 (해당하는 경우)
- 사용자에게 영향을 주는 변경사항 우선 기록

### 변경 유형 설명
- **Added**: 새로운 기능
- **Changed**: 기존 기능의 변경
- **Deprecated**: 곧 제거될 기능 (다음 major 버전에서 제거 예정)
- **Removed**: 제거된 기능
- **Fixed**: 버그 수정
- **Security**: 보안 관련 변경사항