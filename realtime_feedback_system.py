"""
실시간 피드백 시스템

사용자의 피드백을 실시간으로 수집하고 분석하여
학습 과정을 즉시 개선하는 시스템입니다.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import deque
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """피드백 유형"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTIVE = "corrective"
    GUIDANCE = "guidance"
    EMERGENCY_STOP = "emergency_stop"

class FeedbackSource(Enum):
    """피드백 소스"""
    USER_VERBAL = "user_verbal"
    USER_GESTURE = "user_gesture"
    USER_TEXT = "user_text"
    SENSOR_DATA = "sensor_data"
    SYSTEM_AUTO = "system_auto"

@dataclass
class FeedbackEvent:
    """피드백 이벤트"""
    timestamp: datetime
    feedback_type: FeedbackType
    source: FeedbackSource
    content: str
    confidence: float
    context: Dict[str, Any]
    response_required: bool
    processed: bool = False

@dataclass
class LearningAdjustment:
    """학습 조정 사항"""
    skill_name: str
    adjustment_type: str  # "increase", "decrease", "modify", "reset"
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    applied_at: datetime

class RealTimeFeedbackProcessor:
    """실시간 피드백 처리기"""
    
    def __init__(self, physical_ai_system=None):
        self.physical_ai = physical_ai_system
        self.feedback_queue = Queue()
        self.feedback_history = deque(maxlen=1000)  # 최근 1000개 피드백 보관
        self.learning_adjustments = []
        
        # 피드백 분석 설정
        self.feedback_patterns = {}
        self.response_templates = self._load_response_templates()
        
        # 실시간 처리 상태
        self.processing_active = False
        self.processing_thread = None
        
        # 콜백 함수들
        self.feedback_callbacks = []
        self.adjustment_callbacks = []
        
        # 성능 메트릭
        self.metrics = {
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "response_time": 0.0,
            "adjustment_success_rate": 0.0
        }
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """응답 템플릿 로드"""
        return {
            FeedbackType.POSITIVE: [
                "잘했습니다! 이 방식으로 계속 진행하겠습니다.",
                "좋은 피드백입니다. 학습에 반영하겠습니다.",
                "훌륭합니다! 성공 패턴으로 기록했습니다."
            ],
            FeedbackType.NEGATIVE: [
                "이해했습니다. 다른 방법을 시도해보겠습니다.",
                "피드백 감사합니다. 접근 방식을 수정하겠습니다.",
                "알겠습니다. 더 나은 방법을 찾아보겠습니다."
            ],
            FeedbackType.CORRECTIVE: [
                "교정 사항을 반영했습니다.",
                "수정 내용을 적용하겠습니다.",
                "지시사항에 따라 조정하겠습니다."
            ],
            FeedbackType.GUIDANCE: [
                "안내해주셔서 감사합니다. 따라해보겠습니다.",
                "새로운 접근법을 학습하겠습니다.",
                "유용한 조언입니다. 적용해보겠습니다."
            ],
            FeedbackType.EMERGENCY_STOP: [
                "즉시 중단합니다!",
                "안전 모드로 전환합니다.",
                "모든 동작을 정지합니다."
            ]
        }
    
    def start_processing(self):
        """실시간 피드백 처리 시작"""
        if self.processing_active:
            return
        
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("실시간 피드백 처리 시작")
    
    def stop_processing(self):
        """실시간 피드백 처리 중지"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("실시간 피드백 처리 중지")
    
    def add_feedback(self, 
                    feedback_type: FeedbackType,
                    source: FeedbackSource,
                    content: str,
                    confidence: float = 1.0,
                    context: Dict[str, Any] = None,
                    response_required: bool = True):
        """피드백 추가"""
        if context is None:
            context = {}
        
        feedback = FeedbackEvent(
            timestamp=datetime.now(),
            feedback_type=feedback_type,
            source=source,
            content=content,
            confidence=confidence,
            context=context,
            response_required=response_required
        )
        
        self.feedback_queue.put(feedback)
        self.metrics["total_feedback"] += 1
        
        if feedback_type == FeedbackType.POSITIVE:
            self.metrics["positive_feedback"] += 1
        elif feedback_type == FeedbackType.NEGATIVE:
            self.metrics["negative_feedback"] += 1
        
        logger.info(f"피드백 추가: {feedback_type.value} - {content[:50]}...")
    
    def _processing_loop(self):
        """피드백 처리 루프"""
        while self.processing_active:
            try:
                # 피드백 대기 (논블로킹)
                try:
                    feedback = self.feedback_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # 피드백 처리
                start_time = time.time()
                self._process_feedback(feedback)
                processing_time = time.time() - start_time
                
                # 성능 메트릭 업데이트
                self.metrics["response_time"] = (
                    self.metrics["response_time"] * 0.9 + processing_time * 0.1
                )
                
                # 피드백 기록
                self.feedback_history.append(feedback)
                
                # 콜백 호출
                for callback in self.feedback_callbacks:
                    try:
                        callback(feedback)
                    except Exception as e:
                        logger.error(f"피드백 콜백 오류: {e}")
                
                self.feedback_queue.task_done()
                
            except Exception as e:
                logger.error(f"피드백 처리 오류: {e}")
                time.sleep(0.1)
    
    def _process_feedback(self, feedback: FeedbackEvent):
        """개별 피드백 처리"""
        try:
            # 긴급 중지 처리
            if feedback.feedback_type == FeedbackType.EMERGENCY_STOP:
                self._handle_emergency_stop(feedback)
                return
            
            # 컨텍스트 분석
            context_analysis = self._analyze_context(feedback)
            
            # 학습 조정 결정
            adjustments = self._determine_learning_adjustments(feedback, context_analysis)
            
            # 조정 적용
            for adjustment in adjustments:
                self._apply_learning_adjustment(adjustment)
            
            # 응답 생성 및 전송
            if feedback.response_required:
                response = self._generate_response(feedback)
                self._send_response(response, feedback)
            
            feedback.processed = True
            logger.info(f"피드백 처리 완료: {feedback.feedback_type.value}")
            
        except Exception as e:
            logger.error(f"피드백 처리 실패: {e}")
    
    def _handle_emergency_stop(self, feedback: FeedbackEvent):
        """긴급 중지 처리"""
        logger.critical("긴급 중지 신호 수신!")
        
        # Physical AI 시스템에 긴급 중지 신호 전송
        if self.physical_ai:
            try:
                # 모든 동작 즉시 중지
                asyncio.run_coroutine_threadsafe(
                    self._emergency_stop_all_actions(),
                    asyncio.get_event_loop()
                )
            except Exception as e:
                logger.error(f"긴급 중지 실행 오류: {e}")
        
        # 즉시 응답
        emergency_response = "🚨 긴급 중지! 모든 동작을 즉시 중단했습니다."
        self._send_response(emergency_response, feedback)
    
    async def _emergency_stop_all_actions(self):
        """모든 동작 긴급 중지"""
        if self.physical_ai and hasattr(self.physical_ai, 'agent_executor'):
            # 안전 모드 활성화
            await self.physical_ai.agent_executor.emergency_stop()
    
    def _analyze_context(self, feedback: FeedbackEvent) -> Dict[str, Any]:
        """컨텍스트 분석"""
        analysis = {
            "current_skill": feedback.context.get("current_skill", "unknown"),
            "action_in_progress": feedback.context.get("action", "none"),
            "success_rate": feedback.context.get("success_rate", 0.0),
            "recent_performance": self._get_recent_performance(),
            "feedback_frequency": self._get_feedback_frequency(),
            "pattern_match": self._find_feedback_pattern(feedback)
        }
        
        return analysis
    
    def _get_recent_performance(self) -> Dict[str, float]:
        """최근 성능 데이터 가져오기"""
        recent_feedback = list(self.feedback_history)[-10:]  # 최근 10개
        
        if not recent_feedback:
            return {"positive_ratio": 0.5, "confidence": 0.0}
        
        positive_count = sum(1 for f in recent_feedback if f.feedback_type == FeedbackType.POSITIVE)
        positive_ratio = positive_count / len(recent_feedback)
        avg_confidence = np.mean([f.confidence for f in recent_feedback])
        
        return {
            "positive_ratio": positive_ratio,
            "confidence": avg_confidence
        }
    
    def _get_feedback_frequency(self) -> float:
        """피드백 빈도 계산"""
        if len(self.feedback_history) < 2:
            return 0.0
        
        recent_feedback = list(self.feedback_history)[-5:]  # 최근 5개
        time_diffs = []
        
        for i in range(1, len(recent_feedback)):
            diff = (recent_feedback[i].timestamp - recent_feedback[i-1].timestamp).total_seconds()
            time_diffs.append(diff)
        
        return np.mean(time_diffs) if time_diffs else 0.0
    
    def _find_feedback_pattern(self, feedback: FeedbackEvent) -> Optional[str]:
        """피드백 패턴 찾기"""
        # 간단한 패턴 매칭 (실제로는 더 복잡한 ML 모델 사용 가능)
        content_lower = feedback.content.lower()
        
        patterns = {
            "speed_too_fast": ["빨라", "느려", "천천히", "속도"],
            "direction_wrong": ["방향", "돌아", "반대", "좌", "우"],
            "force_too_strong": ["세게", "약하게", "힘", "부드럽게"],
            "position_incorrect": ["위치", "자리", "거기", "여기"],
            "good_execution": ["좋다", "잘한다", "완벽", "훌륭"]
        }
        
        for pattern_name, keywords in patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                return pattern_name
        
        return None
    
    def _determine_learning_adjustments(self, 
                                       feedback: FeedbackEvent, 
                                       context: Dict[str, Any]) -> List[LearningAdjustment]:
        """학습 조정 사항 결정"""
        adjustments = []
        current_skill = context.get("current_skill", "unknown")
        pattern = context.get("pattern_match")
        
        # 패턴 기반 조정
        if pattern == "speed_too_fast" and feedback.feedback_type == FeedbackType.NEGATIVE:
            adjustments.append(LearningAdjustment(
                skill_name=current_skill,
                adjustment_type="decrease",
                parameter="speed_factor",
                old_value=1.0,
                new_value=0.8,
                reason=f"사용자 피드백: {feedback.content}",
                applied_at=datetime.now()
            ))
        
        elif pattern == "good_execution" and feedback.feedback_type == FeedbackType.POSITIVE:
            adjustments.append(LearningAdjustment(
                skill_name=current_skill,
                adjustment_type="increase",
                parameter="success_weight",
                old_value=1.0,
                new_value=1.2,
                reason="긍정적 피드백으로 가중치 증가",
                applied_at=datetime.now()
            ))
        
        # 성능 기반 조정
        recent_perf = context.get("recent_performance", {})
        if recent_perf.get("positive_ratio", 0.5) < 0.3:  # 30% 미만 성공률
            adjustments.append(LearningAdjustment(
                skill_name=current_skill,
                adjustment_type="modify",
                parameter="learning_rate",
                old_value=0.01,
                new_value=0.005,  # 학습률 감소
                reason="낮은 성공률로 인한 보수적 학습",
                applied_at=datetime.now()
            ))
        
        return adjustments
    
    def _apply_learning_adjustment(self, adjustment: LearningAdjustment):
        """학습 조정 적용"""
        try:
            if not self.physical_ai:
                logger.warning("Physical AI 시스템이 없어 조정을 적용할 수 없습니다")
                return
            
            # 발달적 학습 엔진에 조정 적용
            dev_engine = self.physical_ai.dev_engine
            if hasattr(dev_engine, 'skill_engine'):
                skill_engine = dev_engine.skill_engine
                
                if adjustment.skill_name in skill_engine.skills_db:
                    skill = skill_engine.skills_db[adjustment.skill_name]
                    
                    # 파라미터별 조정 적용
                    if adjustment.parameter == "success_weight":
                        # 성공 가중치 조정 (가상의 파라미터)
                        pass
                    elif adjustment.parameter == "speed_factor":
                        # 속도 팩터 조정 (가상의 파라미터)
                        pass
                    
                    self.learning_adjustments.append(adjustment)
                    
                    # 콜백 호출
                    for callback in self.adjustment_callbacks:
                        try:
                            callback(adjustment)
                        except Exception as e:
                            logger.error(f"조정 콜백 오류: {e}")
                    
                    logger.info(f"학습 조정 적용: {adjustment.skill_name} - {adjustment.parameter}")
        
        except Exception as e:
            logger.error(f"학습 조정 적용 실패: {e}")
    
    def _generate_response(self, feedback: FeedbackEvent) -> str:
        """응답 생성"""
        templates = self.response_templates.get(feedback.feedback_type, ["알겠습니다."])
        
        # 템플릿 중 랜덤 선택
        import random
        base_response = random.choice(templates)
        
        # 컨텍스트에 따른 추가 정보
        additional_info = ""
        if feedback.context.get("current_skill"):
            skill_name = feedback.context["current_skill"].replace("_", " ").title()
            additional_info = f" ({skill_name} 스킬 학습 중)"
        
        return base_response + additional_info
    
    def _send_response(self, response: str, feedback: FeedbackEvent):
        """응답 전송"""
        # 실제 구현에서는 사용자 인터페이스로 응답 전송
        logger.info(f"AI 응답: {response}")
        
        # 콜백 함수를 통해 UI에 알림
        response_data = {
            "response": response,
            "original_feedback": asdict(feedback),
            "timestamp": datetime.now().isoformat()
        }
        
        # 응답 콜백 호출
        for callback in self.feedback_callbacks:
            try:
                callback("response", response_data)
            except Exception as e:
                logger.error(f"응답 콜백 오류: {e}")
    
    def register_feedback_callback(self, callback: Callable):
        """피드백 콜백 등록"""
        self.feedback_callbacks.append(callback)
    
    def register_adjustment_callback(self, callback: Callable):
        """조정 콜백 등록"""
        self.adjustment_callbacks.append(callback)
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """피드백 분석 데이터 반환"""
        recent_feedback = list(self.feedback_history)[-50:]  # 최근 50개
        
        if not recent_feedback:
            return {
                "total_feedback": 0,
                "feedback_types": {},
                "sources": {},
                "avg_confidence": 0.0,
                "recent_trend": "no_data"
            }
        
        # 피드백 유형별 분석
        type_counts = {}
        source_counts = {}
        confidences = []
        
        for feedback in recent_feedback:
            # 유형별 카운트
            fb_type = feedback.feedback_type.value
            type_counts[fb_type] = type_counts.get(fb_type, 0) + 1
            
            # 소스별 카운트
            fb_source = feedback.source.value
            source_counts[fb_source] = source_counts.get(fb_source, 0) + 1
            
            # 신뢰도 수집
            confidences.append(feedback.confidence)
        
        # 트렌드 분석 (최근 10개 vs 이전 10개)
        if len(recent_feedback) >= 20:
            recent_10 = recent_feedback[-10:]
            previous_10 = recent_feedback[-20:-10]
            
            recent_positive = sum(1 for f in recent_10 if f.feedback_type == FeedbackType.POSITIVE)
            previous_positive = sum(1 for f in previous_10 if f.feedback_type == FeedbackType.POSITIVE)
            
            if recent_positive > previous_positive:
                trend = "improving"
            elif recent_positive < previous_positive:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "total_feedback": len(recent_feedback),
            "feedback_types": type_counts,
            "sources": source_counts,
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "recent_trend": trend,
            "metrics": self.metrics.copy(),
            "adjustments_count": len(self.learning_adjustments),
            "last_adjustment": self.learning_adjustments[-1].applied_at.isoformat() if self.learning_adjustments else None
        }
    
    def clear_history(self):
        """히스토리 초기화"""
        self.feedback_history.clear()
        self.learning_adjustments.clear()
        self.metrics = {
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "response_time": 0.0,
            "adjustment_success_rate": 0.0
        }
        logger.info("피드백 히스토리 초기화 완료")

# 사용 예제
class FeedbackSystemDemo:
    """피드백 시스템 데모"""
    
    def __init__(self):
        self.feedback_processor = RealTimeFeedbackProcessor()
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """콜백 설정"""
        def feedback_callback(event_type, data):
            if event_type == "response":
                print(f"AI 응답: {data['response']}")
            else:
                print(f"피드백 처리됨: {data.feedback_type.value}")
        
        def adjustment_callback(adjustment):
            print(f"학습 조정: {adjustment.skill_name} - {adjustment.parameter}: {adjustment.old_value} → {adjustment.new_value}")
        
        self.feedback_processor.register_feedback_callback(feedback_callback)
        self.feedback_processor.register_adjustment_callback(adjustment_callback)
    
    def run_demo(self):
        """데모 실행"""
        print("🔄 실시간 피드백 시스템 데모 시작")
        
        # 피드백 처리 시작
        self.feedback_processor.start_processing()
        
        # 예제 피드백들
        demo_feedbacks = [
            (FeedbackType.POSITIVE, "잘했어요! 완벽합니다."),
            (FeedbackType.NEGATIVE, "너무 빨라요. 천천히 해주세요."),
            (FeedbackType.CORRECTIVE, "조금 더 왼쪽으로 이동해주세요."),
            (FeedbackType.GUIDANCE, "이런 방식으로 해보시겠어요?"),
            (FeedbackType.POSITIVE, "좋습니다! 이대로 계속 진행하세요.")
        ]
        
        # 피드백 순차 추가
        for feedback_type, content in demo_feedbacks:
            self.feedback_processor.add_feedback(
                feedback_type=feedback_type,
                source=FeedbackSource.USER_VERBAL,
                content=content,
                context={"current_skill": "basic_movement", "action": "move_to_target"}
            )
            time.sleep(1)  # 1초 간격
        
        # 처리 대기
        time.sleep(3)
        
        # 분석 결과 출력
        analytics = self.feedback_processor.get_feedback_analytics()
        print("\n📊 피드백 분석 결과:")
        print(json.dumps(analytics, indent=2, ensure_ascii=False))
        
        # 정리
        self.feedback_processor.stop_processing()
        print("\n✅ 피드백 시스템 데모 완료")

if __name__ == "__main__":
    demo = FeedbackSystemDemo()
    demo.run_demo()