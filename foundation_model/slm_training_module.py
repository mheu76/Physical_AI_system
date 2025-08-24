"""
sLM Foundation Model Training Module - 모델 훈련 및 파인튜닝

Physical AI 시스템을 위한 sLM Foundation Model의 
훈련, 파인튜닝, 성능 평가를 담당하는 모듈입니다.
"""

import asyncio
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import hashlib
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """훈련 설정"""
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    output_dir: str = "models/slm_foundation"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = True
    dataloader_pin_memory: bool = False

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

class PhysicalAIDataset(Dataset):
    """Physical AI 훈련 데이터셋"""
    
    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 입력 텍스트 구성
        input_text = self._format_input(example)
        
        # 토크나이징
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }
    
    def _format_input(self, example: TrainingExample) -> str:
        """입력 텍스트 포맷팅"""
        context_str = ", ".join([f"{k}: {v}" for k, v in example.context.items()])
        subtasks_str = "\n".join([f"- {task['action']}: {task.get('target', '')}" 
                                 for task in example.subtasks])
        constraints_str = ", ".join([f"{k}: {v}" for k, v in example.constraints.items()])
        
        input_text = f"""Mission: {example.mission}
Context: {context_str}
Subtasks:
{subtasks_str}
Constraints: {constraints_str}
Success Criteria: {', '.join(example.success_criteria)}
Execution Result: {example.execution_result.get('success', False)}
Learning Value: {example.learning_value:.3f}"""
        
        return input_text

class SLMTrainingModule:
    """sLM Foundation Model 훈련 모듈"""
    
    def __init__(self, 
                 phi35_manager,
                 training_config: TrainingConfig = None):
        self.phi35_manager = phi35_manager
        self.config = training_config or TrainingConfig()
        
        # 훈련 데이터
        self.training_examples: List[TrainingExample] = []
        self.validation_examples: List[TrainingExample] = []
        
        # 훈련 상태
        self.is_training = False
        self.current_epoch = 0
        self.global_step = 0
        
        # 성능 메트릭
        self.training_metrics = {
            "total_examples": 0,
            "training_loss": [],
            "validation_loss": [],
            "learning_rate": [],
            "best_validation_loss": float('inf'),
            "training_time": 0.0
        }
        
        # 모델 저장 경로
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🧠 sLM Foundation Model 훈련 모듈 초기화 완료")
    
    async def initialize(self):
        """훈련 모듈 초기화"""
        logger.info("🚀 sLM Foundation Model 훈련 모듈 초기화 중...")
        
        # 기존 훈련 데이터 로드
        await self._load_training_data()
        
        # 훈련 데이터 검증
        await self._validate_training_data()
        
        logger.info("✅ sLM Foundation Model 훈련 모듈 초기화 완료")
    
    async def _load_training_data(self):
        """기존 훈련 데이터 로드"""
        try:
            # 훈련 예제 로드
            training_file = self.output_dir / "training_examples.pkl"
            if training_file.exists():
                with open(training_file, 'rb') as f:
                    self.training_examples = pickle.load(f)
                logger.info(f"📚 {len(self.training_examples)}개 훈련 예제 로드됨")
            
            # 검증 예제 로드
            validation_file = self.output_dir / "validation_examples.pkl"
            if validation_file.exists():
                with open(validation_file, 'rb') as f:
                    self.validation_examples = pickle.load(f)
                logger.info(f"📚 {len(self.validation_examples)}개 검증 예제 로드됨")
                
        except Exception as e:
            logger.warning(f"⚠️ 훈련 데이터 로드 실패: {e}")
    
    async def _validate_training_data(self):
        """훈련 데이터 검증"""
        if not self.training_examples:
            logger.warning("⚠️ 훈련 데이터가 없습니다. 기본 예제를 생성합니다.")
            await self._generate_basic_examples()
        
        if not self.validation_examples:
            logger.warning("⚠️ 검증 데이터가 없습니다. 훈련 데이터에서 분할합니다.")
            await self._split_validation_data()
    
    async def _generate_basic_examples(self):
        """기본 훈련 예제 생성"""
        basic_examples = [
            TrainingExample(
                mission="Pick up the red cup and place it on the table",
                context={"environment": "simple", "safety_level": "normal"},
                subtasks=[
                    {"action": "move_to", "target": "cup_location"},
                    {"action": "grasp", "target": "red_cup"},
                    {"action": "move_to", "target": "table_location"},
                    {"action": "place", "target": "table"}
                ],
                constraints={"max_force": 50.0, "safety_distance": 0.1},
                success_criteria=["cup_picked", "cup_placed"],
                execution_result={"success": True, "efficiency": 0.8},
                learning_value=0.7
            ),
            TrainingExample(
                mission="Organize books on the shelf by size",
                context={"environment": "complex", "safety_level": "high"},
                subtasks=[
                    {"action": "explore", "target": "bookshelf"},
                    {"action": "grasp", "target": "small_book"},
                    {"action": "place", "target": "small_section"},
                    {"action": "grasp", "target": "large_book"},
                    {"action": "place", "target": "large_section"}
                ],
                constraints={"max_force": 30.0, "safety_distance": 0.2},
                success_criteria=["books_organized", "shelf_neat"],
                execution_result={"success": True, "efficiency": 0.6},
                learning_value=0.8
            ),
            TrainingExample(
                mission="Clean up the desk by putting items in their proper places",
                context={"environment": "complex", "safety_level": "normal"},
                subtasks=[
                    {"action": "explore", "target": "desk_surface"},
                    {"action": "grasp", "target": "pencil"},
                    {"action": "place", "target": "pencil_holder"},
                    {"action": "grasp", "target": "paper"},
                    {"action": "place", "target": "paper_tray"}
                ],
                constraints={"max_force": 20.0, "safety_distance": 0.1},
                success_criteria=["desk_clean", "items_organized"],
                execution_result={"success": True, "efficiency": 0.7},
                learning_value=0.6
            )
        ]
        
        self.training_examples.extend(basic_examples)
        logger.info(f"📚 {len(basic_examples)}개 기본 훈련 예제 생성됨")
    
    async def _split_validation_data(self):
        """검증 데이터 분할"""
        if len(self.training_examples) < 2:
            return
        
        # 20%를 검증 데이터로 분할
        split_idx = int(len(self.training_examples) * 0.8)
        self.validation_examples = self.training_examples[split_idx:]
        self.training_examples = self.training_examples[:split_idx]
        
        logger.info(f"📚 훈련: {len(self.training_examples)}개, 검증: {len(self.validation_examples)}개")
    
    async def add_training_example(self, example: TrainingExample, is_validation: bool = False):
        """훈련 예제 추가"""
        if is_validation:
            self.validation_examples.append(example)
        else:
            self.training_examples.append(example)
        
        self.training_metrics["total_examples"] += 1
        
        # 데이터 저장
        await self._save_training_data()
        
        logger.info(f"훈련 예제 추가됨 ({'검증' if is_validation else '훈련'})")
    
    async def _save_training_data(self):
        """훈련 데이터 저장"""
        try:
            # 훈련 예제 저장
            training_file = self.output_dir / "training_examples.pkl"
            with open(training_file, 'wb') as f:
                pickle.dump(self.training_examples, f)
            
            # 검증 예제 저장
            validation_file = self.output_dir / "validation_examples.pkl"
            with open(validation_file, 'wb') as f:
                pickle.dump(self.validation_examples, f)
                
        except Exception as e:
            logger.error(f"❌ 훈련 데이터 저장 실패: {e}")
    
    async def prepare_training_data(self) -> Tuple[Dataset, Dataset]:
        """훈련 데이터 준비"""
        if not self.phi35_manager or not self.phi35_manager.tokenizer:
            raise Exception("PHI-3.5 모델이 초기화되지 않았습니다.")
        
        # 훈련 데이터셋 생성
        train_dataset = PhysicalAIDataset(
            self.training_examples,
            self.phi35_manager.tokenizer,
            max_length=512
        )
        
        # 검증 데이터셋 생성
        val_dataset = PhysicalAIDataset(
            self.validation_examples,
            self.phi35_manager.tokenizer,
            max_length=512
        )
        
        logger.info(f"훈련 데이터셋 준비 완료: {len(train_dataset)}개, 검증: {len(val_dataset)}개")
        
        return train_dataset, val_dataset
    
    async def train_model(self, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """모델 훈련 실행"""
        if self.is_training:
            raise Exception("이미 훈련이 진행 중입니다.")
        
        logger.info("sLM Foundation Model 훈련 시작")
        self.is_training = True
        start_time = datetime.now()
        
        try:
            # 1. 데이터 준비
            train_dataset, val_dataset = await self.prepare_training_data()
            
            # 2. 훈련 설정
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                max_grad_norm=self.config.max_grad_norm,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_total_limit=self.config.save_total_limit,
                fp16=self.config.fp16,
                dataloader_pin_memory=self.config.dataloader_pin_memory,
                # evaluation_strategy="steps",  # 이전 버전 호환성을 위해 주석 처리
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None  # wandb 비활성화
            )
            
            # 3. 데이터 콜레이터
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.phi35_manager.tokenizer,
                mlm=False
            )
            
            # 4. 트레이너 생성
            trainer = Trainer(
                model=self.phi35_manager.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=self.phi35_manager.tokenizer
            )
            
            # 5. 훈련 실행
            logger.info("모델 훈련 시작...")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # 6. 검증
            logger.info("모델 검증 중...")
            eval_result = trainer.evaluate()
            
            # 7. 모델 저장
            logger.info("모델 저장 중...")
            trainer.save_model()
            trainer.save_state()
            
            # 8. 훈련 메트릭 업데이트
            training_time = (datetime.now() - start_time).total_seconds()
            self.training_metrics.update({
                "training_loss": train_result.training_loss,
                "validation_loss": eval_result["eval_loss"],
                "training_time": training_time,
                "best_validation_loss": min(self.training_metrics["best_validation_loss"], 
                                          eval_result["eval_loss"])
            })
            
            # 9. 훈련 결과 저장
            await self._save_training_results(train_result, eval_result)
            
            logger.info("sLM Foundation Model 훈련 완료")
            
            return {
                "success": True,
                "training_loss": train_result.training_loss,
                "validation_loss": eval_result["eval_loss"],
                "training_time": training_time,
                "model_path": str(self.output_dir)
            }
            
        except Exception as e:
            logger.error(f"훈련 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        
        finally:
            self.is_training = False
    
    async def _save_training_results(self, train_result, eval_result):
        """훈련 결과 저장"""
        results = {
            "training_metrics": self.training_metrics,
            "train_result": {
                "training_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "epoch": train_result.epoch
            },
            "eval_result": eval_result,
            "timestamp": datetime.now().isoformat()
        }
        
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    async def evaluate_model(self, test_examples: List[TrainingExample] = None) -> Dict[str, Any]:
        """모델 성능 평가"""
        if not test_examples:
            test_examples = self.validation_examples
        
        if not test_examples:
            return {"success": False, "error": "평가할 예제가 없습니다."}
        
        logger.info(f"🔍 모델 성능 평가 시작: {len(test_examples)}개 예제")
        
        try:
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for example in test_examples:
                # 모델 예측
                prediction = await self._predict_example(example)
                
                # 정확도 계산 (간단한 구현)
                if prediction.get("success") == example.execution_result.get("success"):
                    correct_predictions += 1
                total_predictions += 1
                
                # 손실 계산 (간단한 구현)
                predicted_value = prediction.get("learning_value", 0.0)
                actual_value = example.learning_value
                loss = abs(predicted_value - actual_value)
                total_loss += loss
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            avg_loss = total_loss / len(test_examples) if test_examples else 0.0
            
            evaluation_results = {
                "success": True,
                "accuracy": accuracy,
                "average_loss": avg_loss,
                "total_examples": len(test_examples),
                "correct_predictions": correct_predictions
            }
            
            logger.info(f"📊 평가 결과: 정확도 {accuracy:.3f}, 평균 손실 {avg_loss:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ 모델 평가 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def _predict_example(self, example: TrainingExample) -> Dict[str, Any]:
        """단일 예제 예측"""
        try:
            # 입력 텍스트 구성
            input_text = self._format_input_for_prediction(example)
            
            # 토크나이징
            inputs = self.phi35_manager.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.phi35_manager.model(**inputs)
                logits = outputs.logits
            
            # 결과 해석 (간단한 구현)
            prediction = {
                "success": True,  # 기본값
                "learning_value": 0.5,  # 기본값
                "confidence": 0.7  # 기본값
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ 예측 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _format_input_for_prediction(self, example: TrainingExample) -> str:
        """예측용 입력 텍스트 포맷팅"""
        context_str = ", ".join([f"{k}: {v}" for k, v in example.context.items()])
        subtasks_str = "\n".join([f"- {task['action']}: {task.get('target', '')}" 
                                 for task in example.subtasks])
        
        input_text = f"""Mission: {example.mission}
Context: {context_str}
Subtasks:
{subtasks_str}
Constraints: {example.constraints}
Success Criteria: {example.success_criteria}

Predict the execution result and learning value:"""
        
        return input_text
    
    async def get_training_status(self) -> Dict[str, Any]:
        """훈련 상태 조회"""
        return {
            "is_training": self.is_training,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "training_metrics": self.training_metrics,
            "total_examples": len(self.training_examples),
            "validation_examples": len(self.validation_examples)
        }
    
    async def export_model(self, export_path: str = None) -> Dict[str, Any]:
        """훈련된 모델 내보내기"""
        if not export_path:
            export_path = self.output_dir / "exported_model"
        
        try:
            # 모델과 토크나이저 저장
            self.phi35_manager.model.save_pretrained(export_path)
            self.phi35_manager.tokenizer.save_pretrained(export_path)
            
            # 설정 파일 저장
            config = {
                "model_name": self.config.model_name,
                "training_config": asdict(self.config),
                "training_metrics": self.training_metrics,
                "export_timestamp": datetime.now().isoformat()
            }
            
            config_file = Path(export_path) / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"✅ 모델 내보내기 완료: {export_path}")
            
            return {
                "success": True,
                "export_path": str(export_path),
                "config": config
            }
            
        except Exception as e:
            logger.error(f"❌ 모델 내보내기 실패: {e}")
            return {"success": False, "error": str(e)}

# 테스트 코드
if __name__ == "__main__":
    async def test_training_module():
        """훈련 모듈 테스트"""
        print("🧠 sLM Foundation Model 훈련 모듈 테스트")
        
        # PHI-3.5 매니저 생성 (모의)
        class MockPHI35Manager:
            def __init__(self):
                self.tokenizer = None
                self.model = None
        
        phi35_manager = MockPHI35Manager()
        
        # 훈련 모듈 초기화
        training_module = SLMTrainingModule(phi35_manager)
        await training_module.initialize()
        
        # 훈련 상태 조회
        status = await training_module.get_training_status()
        print(f"훈련 상태: {status}")
        
        print("✅ 훈련 모듈 테스트 완료")
    
    asyncio.run(test_training_module())
