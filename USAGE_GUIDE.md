# Physical AI System 실행 가이드

## 빠른 시작

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 기본 실행
```bash
# 특정 미션 실행
python main.py --mission "Pick up the red cup and place it on the table"

# 지속적 학습 모드
python main.py

# 설정 파일 지정
python main.py --config configs/custom.yaml
```

### 3. 예제 실행
```bash
# 기본 예제
python examples/basic_example.py

# 발달적 학습 예제
python examples/developmental_learning_example.py
```

### 4. 테스트 실행
```bash
# 전체 통합 테스트
python tests/test_integration.py

# pytest 사용
pytest tests/ -v
```

## 시스템 구성 요소

### Foundation Model (추론 엔진)
- **역할**: 미션 해석 및 동작 로직 추론
- **위치**: `foundation_model/slm_foundation.py`
- **핵심 기능**: 
  - 자연어 미션을 구체적 서브태스크로 분해
  - 물리 법칙 기반 동작 예측 및 최적화

### Developmental Learning (발달적 학습)
- **역할**: 아기처럼 점진적 스킬 습득 및 개선
- **위치**: `developmental_learning/dev_engine.py`
- **핵심 기능**:
  - 커리큘럼 학습 (단순 → 복잡)
  - 경험 기반 메모리 관리
  - 자율적 탐색 학습

### Agent Executor (실행 엔진)
- **역할**: 실시간 물리적 실행 및 안전 제어
- **위치**: `ai_agent_execution/agent_executor.py`
- **핵심 기능**:
  - 정밀한 동작 제어
  - 실시간 안전 모니터링
  - 성능 지표 수집

### Hardware Abstraction (하드웨어 추상화)
- **역할**: 다양한 로봇 하드웨어 플랫폼 지원
- **위치**: `hardware_abstraction/hal_manager.py`
- **핵심 기능**:
  - 센서 데이터 융합
  - 액추에이터 통합 제어
  - 플랫폼 독립적 인터페이스

## 주요 미션 예제

### 기본 동작 미션
```python
missions = [
    "Move to position [1, 0, 0.5]",
    "Explore the surrounding area",
    "Return to home position"
]
```

### 조작 미션
```python
missions = [
    "Pick up the red cup",
    "Place it on the table", 
    "Stack the blue blocks",
    "Open the drawer and retrieve the key"
]
```

### 복합 미션
```python
missions = [
    "Clean up the workspace by organizing all tools",
    "Prepare coffee and serve it to the human",
    "Inspect all equipment and report any issues"
]
```

## 설정 커스터마이징

### 학습 설정 조정
```yaml
developmental_learning:
  skill_acquisition:
    success_threshold: 0.8  # 더 높은 기준
    practice_iterations: 10  # 더 많은 연습
  autonomous_learning:
    exploration_interval: 900  # 15분 간격
```

### 안전 설정 강화
```yaml
agent_execution:
  safety:
    emergency_stop_distance: 0.02  # 더 짧은 안전 거리
    monitoring_frequency: 20       # 더 빈번한 모니터링
```

### 하드웨어 설정
```yaml
hardware:
  actuators:
    joints:
      max_velocity: 2.0      # 더 느린 속도
      position_accuracy: 0.005  # 더 높은 정밀도
```

## 개발 팁

### 1. 디버그 모드 활성화
```yaml
development:
  debug_mode: true
  mock_hardware: true  # 하드웨어 없이 테스트
```

### 2. 로깅 레벨 조정
```yaml
monitoring:
  logging:
    level: "DEBUG"  # 더 상세한 로그
```

### 3. 시뮬레이션 환경 사용
```python
from simulation.physics_sim import SimulationManager

sim_manager = SimulationManager(config)
await sim_manager.initialize(gui_mode=True)  # GUI로 시각화
```

## 성능 최적화

### 1. GPU 가속 활성화
```yaml
foundation_model:
  device: "cuda"  # GPU 사용
```

### 2. 병렬 처리 설정
```yaml
system:
  max_workers: 8  # CPU 코어 수에 맞게 조정
```

### 3. 메모리 관리
```yaml
developmental_learning:
  memory:
    max_episodic_memories: 5000  # 메모리 사용량 제한
```

## 문제 해결

### 일반적 오류들

1. **모듈 임포트 오류**
   ```bash
   # Python 경로에 프로젝트 루트 추가
   export PYTHONPATH="${PYTHONPATH}:/path/to/physical_ai_system"
   ```

2. **PyBullet 설치 문제**
   ```bash
   pip install pybullet --upgrade
   ```

3. **메모리 부족**
   - 설정 파일에서 메모리 관련 제한값 조정
   - 배치 크기 줄이기

4. **하드웨어 연결 문제**
   - `mock_hardware: true` 설정으로 시뮬레이션 모드 사용
   - 하드웨어 드라이버 및 권한 확인

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f logs/physical_ai.log

# 특정 레벨 로그 필터링  
grep "ERROR" logs/physical_ai.log
```

## 확장 개발

### 새로운 스킬 추가
```python
# developmental_learning/dev_engine.py에서
new_skill = Skill(
    name="custom_skill",
    difficulty_level=5,
    prerequisites=["basic_movement"],
    success_rate=0.1,
    # ... 기타 속성
)
```

### 새로운 센서 추가
```python
# hardware_abstraction/hal_manager.py에서
class CustomSensor(SensorInterface):
    async def read_data(self) -> SensorData:
        # 센서 데이터 읽기 구현
        pass
```

### 새로운 미션 타입 추가
```python
# foundation_model/slm_foundation.py에서
# decompose_mission 메서드 확장
```

이 가이드를 통해 Physical AI 시스템을 효과적으로 활용하고 개발할 수 있습니다!
