# sLM Foundation Model 개발 및 훈련 가이드

## 개요

이 문서는 Physical AI 시스템을 위한 sLM Foundation Model의 개발 및 훈련 과정을 상세히 설명합니다. PHI-3.5 기반의 지능형 로봇 제어 시스템을 구축하고 지속적으로 개선하는 방법을 다룹니다.

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                sLM Foundation Model                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │   PHI-3.5 Core  │  │  Training Module│              │
│  │     Engine      │  │                 │              │
│  └─────────────────┘  └─────────────────┘              │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Learning Module │  │  Task Planning  │              │
│  │                 │  │     Module      │              │
│  └─────────────────┘  └─────────────────┘              │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Motion Reasoning│  │  Safety Monitor │              │
│  │     Module      │  │                 │              │
│  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

## 1. 개발 환경 설정

### 1.1 필수 요구사항

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 지원 GPU (권장)

### 1.2 설치

```bash
# 저장소 클론
git clone <repository-url>
cd Physical_AI_system-1

# 의존성 설치
pip install -r requirements.txt
pip install -r requirements-hf.txt

# 개발 의존성 설치 (선택사항)
pip install -r requirements-dev.txt
```

### 1.3 환경 설정

```bash
# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0  # GPU 사용
export TRANSFORMERS_CACHE=/path/to/cache  # 모델 캐시 경로
export HF_HOME=/path/to/huggingface  # Hugging Face 홈
```

## 2. 모델 구조

### 2.1 핵심 컴포넌트

**1. PHI-3.5 Core Engine**
- Microsoft PHI-3.5 모델 통합
- 양자화 및 최적화 지원
- 실시간 추론 엔진

**2. Training Module**
- 지속적 학습 및 파인튜닝
- 데이터셋 관리
- 성능 평가 및 모니터링

**3. Learning Module**
- 지식 패턴 생성
- 적응적 학습
- 경험 기반 개선

**4. Task Planning Module**
- 미션 분해 및 계획 수립
- 동작 시퀀스 최적화
- 제약 조건 분석

### 2.2 데이터 구조

```python
@dataclass
class TrainingExample:
    """훈련 예제"""
    mission: str
    context: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    success_criteria: List[str]
    execution_result: Dict[str, Any]
    learning_value: float

@dataclass
class TaskPlan:
    """태스크 계획"""
    mission: str
    subtasks: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    expected_duration: float
    success_criteria: List[str]
```

## 3. 훈련 과정

### 3.1 기본 훈련 실행

```bash
# 기본 훈련 실행
python train_slm_foundation.py

# 설정 파일 사용
python train_slm_foundation.py --config configs/slm_training_config.json

# 커스텀 설정
python train_slm_foundation.py \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --output-dir models/my_slm_model
```

### 3.2 고급 훈련 옵션

```bash
# 체크포인트에서 재개
python train_slm_foundation.py --resume

# 평가만 실행
python train_slm_foundation.py --no-data-gen --no-export

# 특정 디바이스 사용
python train_slm_foundation.py --device cuda:1
```

### 3.3 훈련 설정

```json
{
  "model_type": "phi35",
  "model_name": "microsoft/Phi-3.5-mini-instruct",
  "device": "auto",
  "num_epochs": 3,
  "batch_size": 4,
  "learning_rate": 5e-5,
  "warmup_steps": 100,
  "weight_decay": 0.01,
  "gradient_accumulation_steps": 4,
  "fp16": true
}
```

## 4. 데이터 준비

### 4.1 훈련 데이터 생성

```python
from foundation_model.slm_foundation import SLMFoundation
from foundation_model.slm_training_module import TrainingExample

# Foundation Model 초기화
foundation = SLMFoundation(
    model_type="phi35",
    training_output_dir="models/slm_foundation"
)
await foundation.initialize()

# 훈련 예제 추가
example = TrainingExample(
    mission="Pick up the cup and place it on the table",
    context={"environment": "simple", "safety_level": "normal"},
    subtasks=[
        {"action": "move_to", "target": "cup_location"},
        {"action": "grasp", "target": "cup"},
        {"action": "move_to", "target": "table"},
        {"action": "place", "target": "table"}
    ],
    constraints={"max_force": 50.0, "safety_distance": 0.1},
    success_criteria=["cup_picked", "cup_placed"],
    execution_result={"success": True, "efficiency": 0.8},
    learning_value=0.7
)

await foundation.training_module.add_training_example(example)
```

### 4.2 데이터 검증

```python
# 훈련 상태 확인
status = await foundation.get_training_status()
print(f"총 훈련 예제: {status['total_examples']}개")
print(f"검증 예제: {status['validation_examples']}개")

# 데이터셋 준비
train_dataset, val_dataset = await foundation.training_module.prepare_training_data()
```

## 5. 모델 훈련

### 5.1 기본 훈련

```python
# 모델 훈련 실행
training_result = await foundation.train_model()

if training_result["success"]:
    print(f"훈련 완료: 손실 {training_result['training_loss']:.4f}")
    print(f"검증 손실: {training_result['validation_loss']:.4f}")
    print(f"훈련 시간: {training_result['training_time']:.1f}초")
```

### 5.2 점진적 훈련

```python
# 1단계: 기본 훈련
basic_missions = [
    "Pick up the cup and place it on the table",
    "Move the book to the shelf"
]

for mission in basic_missions:
    await foundation.process_mission_with_learning(
        mission=mission,
        context={"environment": "simple"}
    )

await foundation.train_model()

# 2단계: 추가 훈련
advanced_missions = [
    "Organize the desk by sorting items",
    "Set up the dining table"
]

for mission in advanced_missions:
    await foundation.process_mission_with_learning(
        mission=mission,
        context={"environment": "complex"}
    )

# 체크포인트에서 재개
await foundation.train_model(resume_from_checkpoint=True)
```

### 5.3 훈련 모니터링

```python
# 훈련 상태 실시간 모니터링
import asyncio

async def monitor_training():
    while True:
        status = await foundation.get_training_status()
        if status['is_training']:
            print(f"훈련 중... 에포크 {status['current_epoch']}")
            print(f"글로벌 스텝: {status['global_step']}")
        else:
            break
        await asyncio.sleep(10)

# 백그라운드에서 모니터링
asyncio.create_task(monitor_training())
```

## 6. 성능 평가

### 6.1 모델 평가

```python
# 모델 성능 평가
eval_result = await foundation.evaluate_model()

if eval_result["success"]:
    print(f"정확도: {eval_result['accuracy']:.3f}")
    print(f"평균 손실: {eval_result['average_loss']:.4f}")
    print(f"총 예제: {eval_result['total_examples']}개")
    print(f"정확 예측: {eval_result['correct_predictions']}개")
```

### 6.2 학습 인사이트

```python
# 학습 인사이트 분석
insights = await foundation.get_learning_insights()

print(f"총 학습 예제: {insights['total_examples']}개")
print(f"지식 패턴: {insights['knowledge_patterns']}개")
print(f"성공한 적응: {insights['successful_adaptations']}회")

# 최근 성능
recent_perf = insights['recent_performance']
print(f"최근 성공률: {recent_perf['success_rate']:.1%}")
print(f"평균 학습 가치: {recent_perf['average_learning_value']:.3f}")

# 상위 패턴
top_patterns = insights['top_patterns']
for pattern in top_patterns[:3]:
    print(f"- {pattern['description']} (신뢰도: {pattern['confidence']:.2f})")
```

## 7. 모델 배포

### 7.1 모델 내보내기

```python
# 훈련된 모델 내보내기
export_result = await foundation.export_trained_model("models/slm_foundation_exported")

if export_result["success"]:
    print(f"모델 내보내기 완료: {export_result['export_path']}")
    
    # 설정 정보
    config = export_result['config']
    print(f"모델 이름: {config['model_name']}")
    print(f"훈련 설정: {config['training_config']['num_epochs']} 에포크")
    print(f"내보내기 시간: {config['export_timestamp']}")
```

### 7.2 모델 로드

```python
# 내보낸 모델 로드
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "models/slm_foundation_exported"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 설정 파일 로드
import json
with open(f"{model_path}/config.json", 'r') as f:
    config = json.load(f)
```

## 8. 고급 기능

### 8.1 사용자 정의 훈련 예제

```python
# 특정 도메인에 특화된 훈련 예제
custom_examples = [
    TrainingExample(
        mission="Assist in laboratory by preparing chemical solutions",
        context={"environment": "laboratory", "safety_level": "critical"},
        subtasks=[
            {"action": "move_to", "target": "chemical_storage"},
            {"action": "grasp", "target": "chemical_bottle"},
            {"action": "move_to", "target": "workbench"},
            {"action": "place", "target": "mixing_area"}
        ],
        constraints={"max_force": 10.0, "safety_distance": 0.5},
        success_criteria=["chemical_prepared", "safety_maintained"],
        execution_result={"success": True, "efficiency": 0.9},
        learning_value=0.95
    )
]

# 훈련 모듈에 추가
for example in custom_examples:
    await foundation.training_module.add_training_example(example)
```

### 8.2 학습 전략 최적화

```python
# 학습 전략 최적화
optimization = await foundation.optimize_learning_strategy()

print(f"최적화 점수: {optimization['optimization_score']:.2f}")

for rec in optimization['recommendations']:
    print(f"- {rec['description']}")
    print(f"  액션: {rec['action']}")
```

### 8.3 지식 패턴 분석

```python
# 지식 패턴 조회
patterns = await foundation.get_knowledge_patterns()

print(f"총 패턴 수: {patterns['total_patterns']}개")

for pattern in patterns['patterns'][:5]:
    print(f"- {pattern['id']} ({pattern['type']})")
    print(f"  신뢰도: {pattern['confidence']:.2f}")
    print(f"  사용횟수: {pattern['usage_count']}")
    print(f"  설명: {pattern['description']}")
```

## 9. 문제 해결

### 9.1 일반적인 문제들

**1. 메모리 부족**
```bash
# 배치 크기 줄이기
python train_slm_foundation.py --batch-size 2

# FP16 비활성화
python train_slm_foundation.py --no-fp16

# CPU 사용
python train_slm_foundation.py --device cpu
```

**2. 훈련 속도가 느림**
```bash
# GPU 사용 확인
nvidia-smi

# 배치 크기 증가
python train_slm_foundation.py --batch-size 8

# FP16 활성화
python train_slm_foundation.py --fp16
```

**3. 모델 초기화 실패**
```bash
# 캐시 정리
rm -rf ~/.cache/huggingface
rm -rf models/slm_foundation

# 재설치
pip install --force-reinstall transformers torch
```

### 9.2 디버깅

```python
# 상세 로깅 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 훈련 상태 확인
status = await foundation.get_training_status()
print(json.dumps(status, indent=2, default=str))

# 모델 정보 확인
model_info = foundation.phi35_ai.model_manager.get_model_info()
print(json.dumps(model_info, indent=2, default=str))
```

## 10. 성능 최적화

### 10.1 하드웨어 최적화

- **GPU 메모리**: 최소 8GB 권장
- **CPU**: 멀티코어 프로세서
- **RAM**: 최소 16GB
- **저장소**: SSD 권장

### 10.2 소프트웨어 최적화

```python
# PyTorch 최적화
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# 데이터 로더 최적화
training_args = TrainingArguments(
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    gradient_accumulation_steps=4,
    fp16=True
)
```

## 11. 모니터링 및 로깅

### 11.1 훈련 로그

```bash
# 실시간 로그 확인
tail -f slm_training.log

# 특정 레벨 로그 필터링
grep "ERROR" slm_training.log
grep "WARNING" slm_training.log
```

### 11.2 성능 메트릭

```python
# 훈련 메트릭 수집
metrics = {
    "training_loss": [],
    "validation_loss": [],
    "learning_rate": [],
    "training_time": 0.0
}

# 메트릭 시각화 (선택사항)
import matplotlib.pyplot as plt

plt.plot(metrics["training_loss"])
plt.plot(metrics["validation_loss"])
plt.title("Training Progress")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend(["Training", "Validation"])
plt.show()
```

## 12. 배포 및 운영

### 12.1 프로덕션 배포

```python
# 프로덕션용 모델 로드
production_model = SLMFoundation(
    model_type="phi35",
    model_name="models/slm_foundation_exported"
)

await production_model.initialize()

# 실시간 미션 처리
async def process_mission(mission: str):
    result = await production_model.process_mission_with_learning(
        mission=mission,
        context={"environment": "production"}
    )
    return result
```

### 12.2 모니터링 및 유지보수

```python
# 정기적인 성능 평가
async def periodic_evaluation():
    while True:
        eval_result = await production_model.evaluate_model()
        
        if eval_result["accuracy"] < 0.8:
            # 재훈련 트리거
            await trigger_retraining()
        
        await asyncio.sleep(3600)  # 1시간마다

# 자동 재훈련
async def trigger_retraining():
    # 새로운 데이터로 재훈련
    await production_model.train_model(resume_from_checkpoint=True)
```

## 결론

이 가이드를 통해 sLM Foundation Model의 개발 및 훈련을 체계적으로 수행할 수 있습니다. 지속적인 학습과 개선을 통해 Physical AI 시스템의 성능을 향상시킬 수 있습니다.

### 주요 포인트

1. **점진적 개발**: 기본 기능부터 시작하여 점진적으로 확장
2. **지속적 학습**: 새로운 데이터와 경험을 통한 지속적 개선
3. **성능 모니터링**: 정기적인 평가와 최적화
4. **안전성 우선**: 모든 동작에서 안전성 고려
5. **확장성**: 다양한 환경과 태스크에 대응 가능한 구조

### 다음 단계

- 더 많은 훈련 데이터 수집
- 고급 최적화 기법 적용
- 실시간 학습 시스템 구축
- 다중 에이전트 협업 시스템 개발

---

*이 문서는 sLM Foundation Model의 개발 및 훈련 과정을 이해하고 활용하는 데 도움이 됩니다. 추가 질문이나 개선 사항이 있으시면 언제든지 문의해 주세요.*
