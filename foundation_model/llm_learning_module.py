"""
LLM Foundation Learning Module - 고급 학습 시스템

PHI-3.5 기반 Physical AI 시스템의 고급 학습 기능을 제공합니다.
지속적 학습, 적응적 추론, 지식 증강 등을 통해 시스템이 점진적으로 개선됩니다.
"""

import asyncio
import json
import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import hashlib

from .phi35_integration import PHI35ModelManager

logger = logging.getLogger(__name__)

@dataclass
class LearningExample:
    """학습 예제 데이터 구조"""
    input_mission: str
    context: Dict[str, Any]
    generated_plan: Dict[str, Any]
    execution_result: Dict[str, Any]
    success: bool
    performance_metrics: Dict[str, float]
    timestamp: datetime
    learning_value: float

@dataclass
class KnowledgePattern:
    """지식 패턴 구조"""
    pattern_id: str
    pattern_type: str  # "mission_pattern", "motion_pattern", "constraint_pattern"
    pattern_data: Dict[str, Any]
    confidence: float
    usage_count: int
    last_used: datetime
    created_at: datetime

@dataclass
class AdaptationMetrics:
    """적응 메트릭 구조"""
    accuracy_improvement: float
    response_time_improvement: float
    success_rate_improvement: float
    knowledge_growth: float
    adaptation_score: float

class LLMLearningModule:
    """LLM Foundation 학습 모듈"""
    
    def __init__(self, 
                 phi35_manager: PHI35ModelManager,
                 learning_config: Dict[str, Any]):
        self.phi35_manager = phi35_manager
        self.config = learning_config
        
        # 학습 데이터 저장소
        self.learning_examples: List[LearningExample] = []
        self.knowledge_patterns: Dict[str, KnowledgePattern] = {}
        self.adaptation_history: List[AdaptationMetrics] = []
        
        # 학습 상태
        self.is_learning_enabled = True
        self.learning_rate = 0.01
        self.min_confidence_threshold = 0.7
        
        # 성능 메트릭
        self.performance_metrics = {
            "total_examples": 0,
            "successful_adaptations": 0,
            "knowledge_patterns_created": 0,
            "average_learning_value": 0.0,
            "last_adaptation_time": None
        }
        
        # 학습 데이터 저장 경로
        self.data_dir = Path("data/llm_learning")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🧠 LLM Foundation 학습 모듈 초기화 완료")
    
    async def initialize(self):
        """학습 모듈 초기화"""
        logger.info("🚀 LLM Foundation 학습 모듈 초기화 중...")
        
        # 기존 학습 데이터 로드
        await self._load_learning_data()
        
        # 지식 패턴 초기화
        await self._initialize_knowledge_patterns()
        
        # 적응 메트릭 계산
        await self._calculate_adaptation_metrics()
        
        logger.info("✅ LLM Foundation 학습 모듈 초기화 완료")
    
    async def _load_learning_data(self):
        """기존 학습 데이터 로드"""
        try:
            # 학습 예제 로드
            examples_file = self.data_dir / "learning_examples.pkl"
            if examples_file.exists():
                with open(examples_file, 'rb') as f:
                    self.learning_examples = pickle.load(f)
                logger.info(f"📚 {len(self.learning_examples)}개 학습 예제 로드됨")
            
            # 지식 패턴 로드
            patterns_file = self.data_dir / "knowledge_patterns.pkl"
            if patterns_file.exists():
                with open(patterns_file, 'rb') as f:
                    self.knowledge_patterns = pickle.load(f)
                logger.info(f"🧩 {len(self.knowledge_patterns)}개 지식 패턴 로드됨")
                
        except Exception as e:
            logger.warning(f"⚠️ 학습 데이터 로드 실패: {e}")
    
    async def _initialize_knowledge_patterns(self):
        """기본 지식 패턴 초기화"""
        if not self.knowledge_patterns:
            # 기본 물리 패턴
            basic_patterns = [
                {
                    "pattern_id": "physics_gravity",
                    "pattern_type": "constraint_pattern",
                    "pattern_data": {
                        "constraint": "gravity_effect",
                        "description": "중력의 영향을 고려한 동작 계획",
                        "applicable_missions": ["pick", "place", "move"],
                        "physics_rule": "gravity = 9.81 m/s²"
                    },
                    "confidence": 0.95,
                    "usage_count": 0
                },
                {
                    "pattern_id": "safety_distance",
                    "pattern_type": "constraint_pattern", 
                    "pattern_data": {
                        "constraint": "safety_margin",
                        "description": "안전 거리 유지",
                        "applicable_missions": ["all"],
                        "safety_rule": "minimum_distance = 0.1m"
                    },
                    "confidence": 0.9,
                    "usage_count": 0
                },
                {
                    "pattern_id": "grasp_sequence",
                    "pattern_type": "motion_pattern",
                    "pattern_data": {
                        "sequence": ["approach", "grasp", "lift", "move", "place"],
                        "description": "기본 잡기 동작 시퀀스",
                        "applicable_missions": ["pick_and_place"],
                        "energy_optimization": True
                    },
                    "confidence": 0.85,
                    "usage_count": 0
                }
            ]
            
            for pattern_data in basic_patterns:
                pattern = KnowledgePattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_type=pattern_data["pattern_type"],
                    pattern_data=pattern_data["pattern_data"],
                    confidence=pattern_data["confidence"],
                    usage_count=pattern_data["usage_count"],
                    last_used=datetime.now(),
                    created_at=datetime.now()
                )
                self.knowledge_patterns[pattern.pattern_id] = pattern
            
            logger.info(f"🧩 {len(basic_patterns)}개 기본 지식 패턴 생성됨")
    
    async def learn_from_experience(self, 
                                  mission: str,
                                  context: Dict[str, Any],
                                  generated_plan: Dict[str, Any],
                                  execution_result: Dict[str, Any]) -> float:
        """경험으로부터 학습"""
        if not self.is_learning_enabled:
            return 0.0
        
        try:
            # 학습 예제 생성
            learning_value = self._calculate_learning_value(execution_result)
            
            example = LearningExample(
                input_mission=mission,
                context=context,
                generated_plan=generated_plan,
                execution_result=execution_result,
                success=execution_result.get("success", False),
                performance_metrics=execution_result.get("performance_metrics", {}),
                timestamp=datetime.now(),
                learning_value=learning_value
            )
            
            # 학습 예제 저장
            self.learning_examples.append(example)
            self.performance_metrics["total_examples"] += 1
            
            # 지식 패턴 업데이트
            await self._update_knowledge_patterns(example)
            
            # 적응적 학습 수행
            adaptation_score = await self._perform_adaptive_learning(example)
            
            # 학습 데이터 저장
            await self._save_learning_data()
            
            logger.info(f"📚 학습 완료: 학습가치={learning_value:.3f}, 적응점수={adaptation_score:.3f}")
            return learning_value
            
        except Exception as e:
            logger.error(f"❌ 학습 중 오류 발생: {e}")
            return 0.0
    
    def _calculate_learning_value(self, execution_result: Dict[str, Any]) -> float:
        """학습 가치 계산"""
        success = execution_result.get("success", False)
        performance = execution_result.get("performance_metrics", {})
        
        # 기본 학습 가치
        base_value = 1.0 if success else 0.3
        
        # 성능 기반 보너스
        efficiency = performance.get("efficiency", 0.5)
        accuracy = performance.get("accuracy", 0.5)
        safety = performance.get("safety_score", 0.5)
        
        # 종합 학습 가치
        learning_value = base_value * (0.4 + 0.2 * efficiency + 0.2 * accuracy + 0.2 * safety)
        
        return min(learning_value, 1.0)  # 최대 1.0으로 제한
    
    async def _update_knowledge_patterns(self, example: LearningExample):
        """지식 패턴 업데이트"""
        try:
            # 미션 패턴 분석
            mission_pattern = self._extract_mission_pattern(example.input_mission)
            if mission_pattern:
                await self._update_or_create_pattern("mission_pattern", mission_pattern, example)
            
            # 동작 패턴 분석
            motion_pattern = self._extract_motion_pattern(example.generated_plan)
            if motion_pattern:
                await self._update_or_create_pattern("motion_pattern", motion_pattern, example)
            
            # 제약 패턴 분석
            constraint_pattern = self._extract_constraint_pattern(example.context, example.execution_result)
            if constraint_pattern:
                await self._update_or_create_pattern("constraint_pattern", constraint_pattern, example)
                
        except Exception as e:
            logger.error(f"❌ 지식 패턴 업데이트 실패: {e}")
    
    def _extract_mission_pattern(self, mission: str) -> Optional[Dict[str, Any]]:
        """미션에서 패턴 추출"""
        # 간단한 키워드 기반 패턴 추출
        keywords = mission.lower().split()
        
        pattern = {
            "action_type": None,
            "object_type": None,
            "location_type": None,
            "complexity": "simple"
        }
        
        # 액션 타입 추출
        action_keywords = {
            "pick": ["pick", "grab", "take", "lift"],
            "place": ["place", "put", "set", "drop"],
            "move": ["move", "carry", "transport"],
            "organize": ["organize", "arrange", "sort"],
            "clean": ["clean", "wipe", "sweep"]
        }
        
        for action_type, keywords_list in action_keywords.items():
            if any(keyword in keywords for keyword in keywords_list):
                pattern["action_type"] = action_type
                break
        
        # 객체 타입 추출
        object_keywords = ["cup", "book", "box", "tool", "item", "object"]
        for keyword in object_keywords:
            if keyword in keywords:
                pattern["object_type"] = keyword
                break
        
        # 복잡성 평가
        if len(keywords) > 8 or "and" in keywords:
            pattern["complexity"] = "complex"
        
        return pattern if pattern["action_type"] else None
    
    def _extract_motion_pattern(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """동작 계획에서 패턴 추출"""
        subtasks = plan.get("subtasks", [])
        
        if not subtasks:
            return None
        
        pattern = {
            "sequence_length": len(subtasks),
            "motion_types": [],
            "energy_optimization": False,
            "safety_considerations": []
        }
        
        for subtask in subtasks:
            motion_type = subtask.get("type", "unknown")
            pattern["motion_types"].append(motion_type)
            
            # 에너지 최적화 확인
            if subtask.get("energy_efficient", False):
                pattern["energy_optimization"] = True
            
            # 안전 고려사항 확인
            safety = subtask.get("safety_checks", [])
            pattern["safety_considerations"].extend(safety)
        
        return pattern
    
    def _extract_constraint_pattern(self, context: Dict[str, Any], result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """제약 조건에서 패턴 추출"""
        constraints = context.get("constraints", {})
        
        if not constraints:
            return None
        
        pattern = {
            "constraint_types": [],
            "safety_level": "normal",
            "physical_limits": {},
            "environmental_factors": []
        }
        
        # 제약 타입 추출
        for constraint_type, value in constraints.items():
            pattern["constraint_types"].append(constraint_type)
            
            if constraint_type == "safety_distance":
                pattern["safety_level"] = "high" if value > 0.2 else "normal"
            elif constraint_type == "max_force":
                pattern["physical_limits"]["max_force"] = value
            elif constraint_type == "environment":
                pattern["environmental_factors"].append(value)
        
        return pattern
    
    async def _update_or_create_pattern(self, pattern_type: str, pattern_data: Dict[str, Any], example: LearningExample):
        """패턴 업데이트 또는 생성"""
        # 패턴 ID 생성
        pattern_hash = hashlib.md5(json.dumps(pattern_data, sort_keys=True).encode()).hexdigest()[:8]
        pattern_id = f"{pattern_type}_{pattern_hash}"
        
        if pattern_id in self.knowledge_patterns:
            # 기존 패턴 업데이트
            pattern = self.knowledge_patterns[pattern_id]
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            
            # 신뢰도 업데이트 (성공한 예제로부터)
            if example.success:
                pattern.confidence = min(pattern.confidence + 0.01, 1.0)
            else:
                pattern.confidence = max(pattern.confidence - 0.005, 0.1)
                
        else:
            # 새 패턴 생성
            pattern = KnowledgePattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                pattern_data=pattern_data,
                confidence=0.7 if example.success else 0.3,
                usage_count=1,
                last_used=datetime.now(),
                created_at=datetime.now()
            )
            self.knowledge_patterns[pattern_id] = pattern
            self.performance_metrics["knowledge_patterns_created"] += 1
    
    async def _perform_adaptive_learning(self, example: LearningExample) -> float:
        """적응적 학습 수행"""
        try:
            # 유사한 예제들 찾기
            similar_examples = self._find_similar_examples(example)
            
            if not similar_examples:
                return 0.0
            
            # 패턴 기반 개선
            improvements = []
            
            for similar_example in similar_examples:
                # 성공 패턴 분석
                if similar_example.success and not example.success:
                    improvement = self._analyze_success_pattern(similar_example, example)
                    improvements.append(improvement)
                
                # 실패 패턴 분석
                elif not similar_example.success and example.success:
                    improvement = self._analyze_improvement_pattern(similar_example, example)
                    improvements.append(improvement)
            
            # 평균 개선도 계산
            adaptation_score = np.mean(improvements) if improvements else 0.0
            
            # 적응 메트릭 기록
            adaptation_metric = AdaptationMetrics(
                accuracy_improvement=adaptation_score,
                response_time_improvement=0.0,  # 추후 구현
                success_rate_improvement=0.0,   # 추후 구현
                knowledge_growth=len(self.knowledge_patterns) / 100.0,
                adaptation_score=adaptation_score
            )
            
            self.adaptation_history.append(adaptation_metric)
            self.performance_metrics["successful_adaptations"] += 1
            self.performance_metrics["last_adaptation_time"] = datetime.now()
            
            return adaptation_score
            
        except Exception as e:
            logger.error(f"❌ 적응적 학습 실패: {e}")
            return 0.0
    
    def _find_similar_examples(self, example: LearningExample, max_examples: int = 5) -> List[LearningExample]:
        """유사한 예제들 찾기"""
        similar_examples = []
        
        for prev_example in reversed(self.learning_examples[:-1]):  # 현재 예제 제외
            similarity = self._calculate_similarity(example, prev_example)
            
            if similarity > 0.6:  # 유사도 임계값
                similar_examples.append(prev_example)
                
                if len(similar_examples) >= max_examples:
                    break
        
        return similar_examples
    
    def _calculate_similarity(self, example1: LearningExample, example2: LearningExample) -> float:
        """두 예제 간 유사도 계산"""
        # 미션 유사도
        mission_similarity = self._calculate_text_similarity(
            example1.input_mission, example2.input_mission
        )
        
        # 컨텍스트 유사도
        context_similarity = self._calculate_context_similarity(
            example1.context, example2.context
        )
        
        # 가중 평균
        similarity = 0.6 * mission_similarity + 0.4 * context_similarity
        return similarity
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (간단한 구현)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """컨텍스트 유사도 계산"""
        # 간단한 키 기반 유사도
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        return len(intersection) / len(union)
    
    def _analyze_success_pattern(self, success_example: LearningExample, current_example: LearningExample) -> float:
        """성공 패턴 분석"""
        # 성공한 예제의 특징 분석
        success_features = self._extract_success_features(success_example)
        current_features = self._extract_success_features(current_example)
        
        # 차이점 분석
        differences = self._analyze_differences(success_features, current_features)
        
        # 개선 가능성 점수
        improvement_score = sum(differences.values()) / len(differences) if differences else 0.0
        
        return improvement_score
    
    def _analyze_improvement_pattern(self, failure_example: LearningExample, current_example: LearningExample) -> float:
        """개선 패턴 분석"""
        # 실패한 예제와 현재 성공한 예제의 차이점 분석
        failure_features = self._extract_success_features(failure_example)
        current_features = self._extract_success_features(current_example)
        
        # 개선된 점들 분석
        improvements = self._analyze_improvements(failure_features, current_features)
        
        # 개선 점수
        improvement_score = sum(improvements.values()) / len(improvements) if improvements else 0.0
        
        return improvement_score
    
    def _extract_success_features(self, example: LearningExample) -> Dict[str, Any]:
        """성공 특징 추출"""
        features = {
            "mission_complexity": len(example.input_mission.split()),
            "plan_steps": len(example.generated_plan.get("subtasks", [])),
            "execution_time": example.execution_result.get("execution_time", 0.0),
            "energy_efficiency": example.performance_metrics.get("efficiency", 0.0),
            "safety_score": example.performance_metrics.get("safety_score", 0.0)
        }
        return features
    
    def _analyze_differences(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, float]:
        """특징 차이점 분석"""
        differences = {}
        
        for key in features1.keys():
            if key in features2:
                diff = abs(features1[key] - features2[key])
                max_val = max(features1[key], features2[key])
                if max_val > 0:
                    differences[key] = diff / max_val
                else:
                    differences[key] = 0.0
        
        return differences
    
    def _analyze_improvements(self, old_features: Dict[str, Any], new_features: Dict[str, Any]) -> Dict[str, float]:
        """개선점 분석"""
        improvements = {}
        
        for key in old_features.keys():
            if key in new_features:
                if key in ["energy_efficiency", "safety_score"]:
                    # 높을수록 좋은 지표
                    improvement = (new_features[key] - old_features[key]) / max(old_features[key], 0.1)
                else:
                    # 낮을수록 좋은 지표 (시간, 복잡성 등)
                    improvement = (old_features[key] - new_features[key]) / max(old_features[key], 0.1)
                
                improvements[key] = max(improvement, 0.0)
        
        return improvements
    
    async def _calculate_adaptation_metrics(self):
        """적응 메트릭 계산"""
        if len(self.adaptation_history) < 2:
            return
        
        recent_metrics = self.adaptation_history[-10:]  # 최근 10개
        
        avg_adaptation = np.mean([m.adaptation_score for m in recent_metrics])
        avg_knowledge_growth = np.mean([m.knowledge_growth for m in recent_metrics])
        
        logger.info(f"📊 적응 메트릭 - 평균 적응점수: {avg_adaptation:.3f}, 지식성장: {avg_knowledge_growth:.3f}")
    
    async def _save_learning_data(self):
        """학습 데이터 저장"""
        try:
            # 학습 예제 저장
            examples_file = self.data_dir / "learning_examples.pkl"
            with open(examples_file, 'wb') as f:
                pickle.dump(self.learning_examples, f)
            
            # 지식 패턴 저장
            patterns_file = self.data_dir / "knowledge_patterns.pkl"
            with open(patterns_file, 'wb') as f:
                pickle.dump(self.knowledge_patterns, f)
            
            # 성능 메트릭 저장
            metrics_file = self.data_dir / "performance_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"❌ 학습 데이터 저장 실패: {e}")
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """학습 인사이트 제공"""
        insights = {
            "total_examples": self.performance_metrics["total_examples"],
            "knowledge_patterns": len(self.knowledge_patterns),
            "successful_adaptations": self.performance_metrics["successful_adaptations"],
            "average_learning_value": self.performance_metrics["average_learning_value"],
            "recent_performance": self._get_recent_performance(),
            "top_patterns": self._get_top_patterns(),
            "learning_trends": self._get_learning_trends()
        }
        
        return insights
    
    def _get_recent_performance(self) -> Dict[str, float]:
        """최근 성능 분석"""
        if not self.learning_examples:
            return {}
        
        recent_examples = self.learning_examples[-20:]  # 최근 20개
        
        success_rate = sum(1 for ex in recent_examples if ex.success) / len(recent_examples)
        avg_learning_value = np.mean([ex.learning_value for ex in recent_examples])
        
        return {
            "success_rate": success_rate,
            "average_learning_value": avg_learning_value,
            "examples_count": len(recent_examples)
        }
    
    def _get_top_patterns(self) -> List[Dict[str, Any]]:
        """상위 패턴들 반환"""
        sorted_patterns = sorted(
            self.knowledge_patterns.values(),
            key=lambda p: (p.confidence, p.usage_count),
            reverse=True
        )
        
        top_patterns = []
        for pattern in sorted_patterns[:5]:  # 상위 5개
            top_patterns.append({
                "pattern_id": pattern.pattern_id,
                "type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "usage_count": pattern.usage_count,
                "description": pattern.pattern_data.get("description", "No description")
            })
        
        return top_patterns
    
    def _get_learning_trends(self) -> Dict[str, Any]:
        """학습 트렌드 분석"""
        if len(self.learning_examples) < 10:
            return {}
        
        # 시간별 성공률 트렌드
        recent_examples = self.learning_examples[-50:]  # 최근 50개
        
        success_trend = []
        learning_value_trend = []
        
        for i in range(0, len(recent_examples), 10):
            batch = recent_examples[i:i+10]
            if batch:
                success_rate = sum(1 for ex in batch if ex.success) / len(batch)
                avg_learning_value = np.mean([ex.learning_value for ex in batch])
                
                success_trend.append(success_rate)
                learning_value_trend.append(avg_learning_value)
        
        return {
            "success_trend": success_trend,
            "learning_value_trend": learning_value_trend,
            "trend_direction": "improving" if success_trend[-1] > success_trend[0] else "declining"
        }
    
    async def optimize_learning_strategy(self) -> Dict[str, Any]:
        """학습 전략 최적화"""
        insights = await self.get_learning_insights()
        
        # 최적화 제안
        recommendations = []
        
        # 성공률이 낮은 경우
        if insights["recent_performance"]["success_rate"] < 0.6:
            recommendations.append({
                "type": "success_rate_improvement",
                "description": "성공률이 낮습니다. 더 많은 기본 패턴 학습이 필요합니다.",
                "action": "increase_basic_pattern_learning"
            })
        
        # 학습 가치가 낮은 경우
        if insights["recent_performance"]["average_learning_value"] < 0.5:
            recommendations.append({
                "type": "learning_value_improvement", 
                "description": "학습 가치가 낮습니다. 더 복잡한 미션에 도전해보세요.",
                "action": "increase_mission_complexity"
            })
        
        # 패턴이 부족한 경우
        if insights["knowledge_patterns"] < 10:
            recommendations.append({
                "type": "pattern_diversity",
                "description": "지식 패턴이 부족합니다. 다양한 미션을 시도해보세요.",
                "action": "increase_mission_variety"
            })
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "optimization_score": len(recommendations) / 3.0  # 최적화 필요도
        }
