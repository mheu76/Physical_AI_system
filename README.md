# Physical AI System 🤖 - PHI-3.5 내장

[![CI](https://github.com/your-username/physical-ai-system/workflows/CI/badge.svg)](https://github.com/your-username/physical-ai-system/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PHI-3.5](https://img.shields.io/badge/PHI--3.5-Embedded-brightgreen.svg)](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)

**Microsoft PHI-3.5 내장 Developmental Robotics & Embodied AI 시스템**

> 🧠 **PHI-3.5 소형 언어모델**이 내장된 차세대 Physical AI 시스템  
> 🚀 **자연어 → 물리적 동작** 실시간 변환  
> 🌱 **발달적 학습**으로 스스로 성장하는 로봇 지능

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (4GB+ for optimal performance)
- 8GB+ RAM
- Windows/Linux/macOS

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/physical-ai-system.git
cd physical-ai-system

# Install dependencies
pip install -r requirements.txt

# Run GUI-based behavior model definition system
python behavior_model_gui.py

# Or run basic example
python examples/basic_example.py
```

### 🎮 Interactive Behavior Model Definition

**GUI 시스템 실행:**
```bash
python behavior_model_gui.py
```

**특징:**
- 🎯 **PHI-3.5와 자연어 대화**로 행동모델 정의
- ⚡ **8-bit 양자화**로 GPU 메모리 50% 절약
- 🤖 **실시간 행동모델 생성** 및 테스트
- 📊 **시각적 모델 관리** 인터페이스

**사용 예시:**
```
사용자: "커피를 만드는 행동모델을 정의해주세요"
PHI-3.5: [구조화된 JSON 행동모델 생성]

사용자: "청소하는 행동모델을 만들어주세요"
PHI-3.5: [청소 행동모델 생성]
```

## 1. 핵심 개념 이해

### Developmental Robotics (발달 로보틱스)

**정의**: 생물학적 발달 과정을 모방하여 로봇이 점진적으로 학습하고 성장하는 AI 패러다임

#### 핵심 원리
- **점진적 학습**: 단순한 동작부터 복잡한 행동까지 단계별 습득
- **자기조직화**: 외부 지시 없이 스스로 학습 목표와 방향 설정  
- **실패 기반 학습**: 시행착오를 통한 자연스러운 개선
- **다중 시간척도**: 빠른 적응과 느린 구조적 학습의 조합

#### 발달 단계 모델
```
Level 0: 반사적 반응 (센서 자극 → 단순 반응)
Level 1: 감각-운동 협응 (손과 눈의 협응 학습)
Level 2: 객체 영속성 (사라진 물체도 존재함을 이해)
Level 3: 인과관계 이해 (행동과 결과의 연결)
Level 4: 추상적 사고 (계획과 목표 설정)
```

### Embodied AI (체화된 AI)

**정의**: 물리적 몸체와 환경의 상호작용을 통해 지능이 발현되는 AI 접근법

#### 핵심 개념
- **체화된 인지**: 몸체의 물리적 특성이 인지 과정에 직접 영향
- **환경과의 상호작용**: 수동적 관찰이 아닌 능동적 탐색을 통한 학습
- **감각-운동 통합**: 인식과 행동이 분리되지 않고 통합된 루프
- **상황 의존적 지능**: 특정 환경과 과제에 최적화된 지능 발현

#### 체화의 4단계
1. **물리적 체화**: 실제 몸체와 센서를 통한 물리적 상호작용
2. **감각적 체화**: 다양한 감각 모달리티의 통합적 처리  
3. **운동적 체화**: 동작과 인지의 밀접한 연결
4. **사회적 체화**: 다른 개체와의 상호작용을 통한 학습

---

## 2. 피지컬 AI 기술 스택 아키텍처

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│                Mission Interface Layer                   │
├─────────────────────────────────────────────────────────┤
│           sLM Foundation Model (추론 엔진)               │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Task Planning  │  │  Motion Reasoning│              │
│  │      Module      │  │     Module       │              │
│  └─────────────────┘  └─────────────────┘              │
├─────────────────────────────────────────────────────────┤
│              Developmental Learning Layer                │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Skill Acquisition│  │Memory & Experience│             │
│  │     Engine       │  │    Management    │             │
│  └─────────────────┘  └─────────────────┘              │
├─────────────────────────────────────────────────────────┤
│                AI Agent Execution Layer                  │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Motion Control  │  │ Real-time Safety │             │
│  │    & Dynamics    │  │   Monitoring     │             │
│  └─────────────────┘  └─────────────────┘              │
├─────────────────────────────────────────────────────────┤
│              Hardware Abstraction Layer                 │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Sensor Fusion  │  │  Actuator Control│             │
│  │    Interface     │  │   Interface      │             │
│  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 상세 기술 스택

#### **Layer 1: sLM Foundation Model** 
**역할**: 미션 해석 및 고차원적 동작 로직 추론

```yaml
Architecture:
  - Model: Transformer 기반 sLM (1-7B 파라미터)
  - Specialization: Physical reasoning + Spatial understanding
  - Training Data: 물리 시뮬레이션 + 실제 로봇 데이터
  
Core Components:
  Task Planning Module:
    - 자연어 미션을 구체적 서브태스크로 분해
    - 제약조건 분석 (물리적 한계, 안전성)
    - 우선순위 기반 실행 계획 수립
    
  Motion Reasoning Module:
    - 물리 법칙 기반 동작 예측
    - 에너지 효율성 최적화
    - 다중 해법 생성 및 평가

Tech Stack:
  - Framework: PyTorch / JAX
  - Model: Custom Transformer + Physics-informed layers
  - Inference: ONNX Runtime / TensorRT 최적화
```

#### **Layer 2: Developmental Learning Engine**
**역할**: 지속적 학습 및 기술 습득 관리

```yaml
Skill Acquisition Engine:
  - Curriculum Learning: 단순 → 복잡 기술 점진적 학습
  - Meta-Learning: 새로운 기술을 빠르게 학습하는 방법 학습
  - Imitation Learning: 시연 데이터로부터 기본 동작 습득
  - Reinforcement Learning: 환경 피드백 기반 최적화
  
Memory & Experience Management:
  - Episodic Memory: 구체적 경험 저장 (상황-행동-결과)
  - Semantic Memory: 일반화된 지식 구조 (동작 패턴, 물리 규칙)
  - Working Memory: 실시간 정보 처리 버퍼
  - Experience Replay: 과거 경험 재활용 학습

Tech Stack:
  - RL Framework: Stable Baselines3 / Ray RLlib
  - Memory System: Redis + Vector DB (Pinecone/Weaviate)
  - Curriculum: OpenAI Gym 환경 + Custom Physics Sim
  - Meta-Learning: MAML / Reptile 알고리즘
```

#### **Layer 3: AI Agent Execution Layer**
**역할**: 실시간 물리적 실행 및 안전 제어

```yaml
Motion Control & Dynamics:
  - Model Predictive Control (MPC): 예측 기반 최적 제어
  - Adaptive Control: 하드웨어 변화에 실시간 적응
  - Force/Torque Control: 정밀한 물리적 상호작용
  - Trajectory Optimization: 부드럽고 효율적인 경로 생성
  
Real-time Safety Monitoring:
  - Collision Detection: 3D 공간 충돌 예측 및 회피
  - Emergency Stop: 위험 상황 즉시 정지 시스템
  - Constraint Validation: 관절 한계, 속도 제한 실시간 검증
  - Human Safety: 인간 근접시 자동 안전모드 진입

Tech Stack:
  - Control Framework: ROS2 / Drake Manipulation
  - Real-time OS: RT-Linux / VxWorks
  - Safety System: IEC 61508 표준 준수
  - Physics Engine: Bullet3 / MuJoCo 실시간 시뮬레이션
```

#### **Layer 4: Hardware Abstraction Layer**
**역할**: 다양한 하드웨어 플랫폼 추상화

```yaml
Sensor Fusion Interface:
  - Vision: RGB-D 카메라, LiDAR, 스테레오 비전
  - Tactile: 압력, 온도, 질감 센서 통합
  - Proprioceptive: 관절 위치, 속도, 토크 센서
  - IMU: 가속도, 각속도, 자기장 센서
  
Actuator Control Interface:
  - Motors: 서보, 스테퍼, BLDC 모터 통합 제어
  - Pneumatic: 공압 액추에이터 압력 제어
  - Hydraulic: 유압 시스템 밸브 제어
  - End-effectors: 그리퍼, 툴 체인저 인터페이스

Tech Stack:
  - HAL Framework: Robot Operating System 2 (ROS2)
  - Communication: EtherCAT / CAN Bus 실시간 통신
  - Driver Layer: Custom Hardware Drivers + SDK
  - Calibration: 자동 센서-액추에이터 캘리브레이션
```

---

## 3. 구현 로드맵

### Phase 1: Foundation 구축 (3-6개월)
- [ ] sLM Foundation Model 개발 및 훈련
- [ ] 기본 하드웨어 추상화 계층 구현
- [ ] 시뮬레이션 환경 구축 (MuJoCo/Isaac Sim)

### Phase 2: 학습 시스템 개발 (6-9개월)  
- [ ] Developmental Learning Engine 구현
- [ ] 메모리 시스템 및 경험 관리 개발
- [ ] 기본적인 스킬 습득 파이프라인 구축

### Phase 3: 실제 배포 (9-12개월)
- [ ] 실제 로봇 하드웨어 통합
- [ ] 안전 시스템 완전 구현
- [ ] 실환경 테스트 및 성능 최적화

### Phase 4: 고도화 (12개월+)
- [ ] 다중 로봇 협업 학습
- [ ] 복잡한 조작 기술 습득
- [ ] 인간-로봇 협업 최적화

---

## 4. 기대 효과 및 혁신성

### 기술적 혁신
- **자율적 기술 습득**: 프로그래밍 없이 스스로 새로운 동작 학습
- **하드웨어 적응성**: 다양한 로봇 플랫폼에 빠른 적응 가능
- **지속적 개선**: 사용할수록 더 정교하고 효율적으로 진화

### 실용적 가치
- **개발 비용 절감**: 각 태스크별 개별 프로그래밍 불필요
- **유지보수 효율성**: 자가 진단 및 적응으로 인한 다운타임 최소화  
- **확장성**: 새로운 환경과 태스크에 빠른 전이 학습

## 5. 실제 구현 코드 구조

### 프로젝트 디렉토리
```
physical_ai_system/
├── main.py                    # 시스템 메인 실행
├── foundation_model/          # sLM Foundation 구현
├── developmental_learning/    # 발달적 학습 엔진  
├── ai_agent_execution/       # 실시간 실행 레이어
├── hardware_abstraction/     # 하드웨어 추상화
├── simulation/               # 물리 시뮬레이션
├── examples/                 # 사용 예제
└── tests/                    # 통합 테스트
```

### 실행 방법
```bash
# 기본 미션 실행
python main.py --mission "Pick up the red cup and place it on the table"

# 발달적 학습 예제
python examples/developmental_learning_example.py

# 물리 시뮬레이션
python simulation/physics_sim.py
```

### 핵심 API
```python
# Physical AI 시스템 초기화
physical_ai = PhysicalAI("configs/default.yaml")
await physical_ai.initialize()

# 미션 실행
result = await physical_ai.execute_mission("자연어 미션")

# 자율적 학습
await physical_ai.developmental_learning_cycle()
```

---

## 🚀 시작하기

### 시스템 요구사항
- **Python 3.8+**
- **PyTorch 2.1+** 
- **GPU 권장** (PHI-3.5 모델 최적화)
- **8GB+ RAM** (모델 로딩용)

### 설치 및 실행

#### 1. 기본 설치
```bash
# 저장소 클론
git clone https://github.com/your-username/physical-ai-system.git
cd physical-ai-system

# 의존성 설치 (PHI-3.5 지원)
pip install -r requirements.txt
```

#### 2. PHI-3.5 데모 실행
```bash
# 완전한 PHI-3.5 통합 데모
python examples/phi35_demo_example.py

# 기본 예제 (폴백 모드 포함)
python examples/basic_example.py
```

#### 3. 인터랙티브 미션 실행
```bash
# PHI-3.5로 실제 미션 수행
python main.py --mission "Pick up the red cup and place it on the table"

# 지속적 학습 모드
python main.py --config configs/default.yaml
```

### 성능 최적화

#### GPU 가속 (권장)
```yaml
# configs/default.yaml
foundation_model:
  phi35:
    device: "cuda"  # GPU 사용
    optimization:
      torch_dtype: "float16"  # 메모리 절약
```

#### CPU 최적화
```yaml
# configs/default.yaml  
foundation_model:
  phi35:
    device: "cpu"
    optimization:
      low_cpu_mem_usage: true
```

## 🤝 기여하기

Physical AI System은 오픈소스 프로젝트입니다! 기여를 환영합니다.

- 🐛 **버그 리포트**: Issues 탭에서 버그를 신고해주세요
- 💡 **새로운 기능**: Pull Request로 새로운 기능을 제안해주세요  
- 📖 **문서 개선**: 문서를 개선하거나 번역해주세요
- ⭐ **Star**: 프로젝트가 도움이 되었다면 Star를 눌러주세요!

자세한 기여 가이드는 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

## 📄 라이센스

이 프로젝트는 MIT License 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 연락처

- **GitHub Issues**: [이슈 생성](https://github.com/your-username/physical-ai-system/issues)
- **Discussions**: [토론 참여](https://github.com/your-username/physical-ai-system/discussions)
- **Email**: mheu76@gmail.com

## 🙏 감사의 말

이 프로젝트는 다음 연구와 프로젝트들에서 영감을 받았습니다:

- Developmental Robotics 연구 커뮤니티
- Embodied AI 연구진
- 오픈소스 로보틱스 커뮤니티

---

**이러한 피지컬 AI는 단순한 자동화를 넘어서 진정한 지능형 물리적 파트너로 진화할 수 있는 기반을 제공합니다.** 🤖✨

*"The future of AI is not just digital, it's physical."*
