# Changelog

Physical AI System 프로젝트의 모든 주요 변경사항이 이 파일에 문서화됩니다.

이 프로젝트는 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)을 따릅니다.

## [Unreleased]

### 🚀 Major Features Added
- **PHI-3.5 기반 Physical AI 시스템** 완전 구현
- **8-bit 양자화 GPU 최적화** (BitsAndBytes 통합)
- **대화형 행동모델 정의 GUI** (tkinter 기반)
- **발달적 학습 시스템** (5단계 성장 모델)
- **4계층 아키텍처** 완전 구현

### 🎯 Core Components
- **Foundation Layer**: PHI-3.5 언어모델 통합
- **Developmental Learning**: 점진적 스킬 습득 시스템
- **AI Agent Execution**: 실시간 행동 실행 엔진
- **Hardware Abstraction**: 물리적 하드웨어 제어

### 🔧 Technical Improvements
- **메모리 최적화**: 8-bit 양자화로 GPU 메모리 50% 절약
- **스마트 디바이스 선택**: GPU/CPU 자동 감지 및 전환
- **비동기 처리**: 이벤트 루프 충돌 해결
- **에러 처리**: CUDA 메모리 부족 시 자동 CPU 폴백

### 📁 New Files Added
- `behavior_model_dialog.py` - 콘솔 기반 대화형 시스템
- `behavior_model_gui.py` - GUI 기반 대화형 시스템
- `examples/behavior_model_example.py` - 사용 예제
- `foundation_model/phi35_integration.py` - PHI-3.5 통합 모듈

### 🛠️ Dependencies Updated
- `bitsandbytes>=0.41.0` - 8-bit 양자화 지원
- `transformers>=4.35.0` - PHI-3.5 모델 지원
- `accelerate>=0.24.0` - 모델 로딩 최적화

### 🎮 User Interface
- **좌측 패널**: PHI-3.5와 자연어 대화
- **우측 패널**: 행동모델 관리 및 테스트
- **빠른 명령**: 커피/청소/요리 모델 정의 버튼
- **실시간 모델 생성**: JSON 구조화된 행동모델

### 🔬 Research Features
- **발달적 학습**: 아기처럼 점진적으로 성장하는 AI
- **물리적 제약**: 안전성과 물리 법칙 고려
- **실시간 적응**: 환경 변화에 동적 대응
- **협업 능력**: 인간-로봇 협업 지원

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