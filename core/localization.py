"""
언어 및 로컬라이제이션 지원 모듈
Physical AI System의 다국어 지원 및 메시지 관리
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class LocalizationManager:
    """시스템 언어 및 로컬라이제이션 관리"""
    
    def __init__(self, default_language: str = "ko", locale_dir: str = "locales"):
        self.default_language = default_language
        self.current_language = default_language
        self.locale_dir = Path(locale_dir)
        self.messages: Dict[str, Dict[str, str]] = {}
        self.supported_languages = ["ko", "en"]
        
        # 메시지 초기화
        self._load_default_messages()
        
    def _load_default_messages(self):
        """기본 메시지 로드"""
        self.messages = {
            "ko": {
                # 시스템 메시지
                "system.initializing": "시스템 초기화 중...",
                "system.ready": "시스템 준비 완료",
                "system.error": "시스템 오류",
                "system.shutdown": "시스템 종료 중...",
                
                # Foundation Model 메시지  
                "foundation.initializing": "Foundation Model 초기화 중...",
                "foundation.ready": "Foundation Model 준비 완료",
                "foundation.error": "Foundation Model 오류",
                "foundation.phi35_loading": "PHI-3.5 모델 로드 중...",
                "foundation.phi35_ready": "PHI-3.5 모델 준비 완료",
                "foundation.mission_processing": "미션 처리 중...",
                "foundation.mission_complete": "미션 처리 완료",
                
                # Learning 메시지
                "learning.training_start": "훈련 시작",
                "learning.training_complete": "훈련 완료", 
                "learning.learning_progress": "학습 진행 중...",
                "learning.skill_acquired": "새로운 스킬 습득",
                
                # Execution 메시지
                "execution.start": "실행 시작",
                "execution.complete": "실행 완료",
                "execution.error": "실행 오류",
                "execution.safety_check": "안전 점검 중...",
                
                # Hardware 메시지
                "hardware.connecting": "하드웨어 연결 중...",
                "hardware.connected": "하드웨어 연결 완료",
                "hardware.disconnected": "하드웨어 연결 해제",
                "hardware.error": "하드웨어 오류",
                
                # 일반 메시지
                "success": "성공",
                "error": "오류", 
                "warning": "경고",
                "info": "정보",
                "debug": "디버그",
                "loading": "로드 중...",
                "saving": "저장 중...",
                "complete": "완료",
                "failed": "실패",
                "retry": "재시도",
                "timeout": "시간 초과",
                "cancelled": "취소됨",
                
                # 미션 관련
                "mission.received": "미션 수신",
                "mission.processing": "미션 처리 중",
                "mission.decomposing": "미션 분해 중",
                "mission.planning": "미션 계획 수립 중",
                "mission.executing": "미션 실행 중",
                "mission.completed": "미션 완료",
                "mission.failed": "미션 실패",
                
                # 상태 메시지
                "status.idle": "대기 중",
                "status.busy": "작업 중", 
                "status.error": "오류 상태",
                "status.maintenance": "유지보수",
                "status.offline": "오프라인",
                "status.online": "온라인"
            },
            
            "en": {
                # System messages
                "system.initializing": "System initializing...",
                "system.ready": "System ready",
                "system.error": "System error",
                "system.shutdown": "System shutting down...",
                
                # Foundation Model messages
                "foundation.initializing": "Foundation Model initializing...",
                "foundation.ready": "Foundation Model ready",
                "foundation.error": "Foundation Model error",
                "foundation.phi35_loading": "Loading PHI-3.5 model...",
                "foundation.phi35_ready": "PHI-3.5 model ready",
                "foundation.mission_processing": "Processing mission...",
                "foundation.mission_complete": "Mission processing complete",
                
                # Learning messages
                "learning.training_start": "Training started",
                "learning.training_complete": "Training completed",
                "learning.learning_progress": "Learning in progress...",
                "learning.skill_acquired": "New skill acquired",
                
                # Execution messages
                "execution.start": "Execution started",
                "execution.complete": "Execution completed",
                "execution.error": "Execution error",
                "execution.safety_check": "Safety check in progress...",
                
                # Hardware messages
                "hardware.connecting": "Connecting to hardware...",
                "hardware.connected": "Hardware connected",
                "hardware.disconnected": "Hardware disconnected",
                "hardware.error": "Hardware error",
                
                # General messages
                "success": "Success",
                "error": "Error",
                "warning": "Warning", 
                "info": "Info",
                "debug": "Debug",
                "loading": "Loading...",
                "saving": "Saving...",
                "complete": "Complete",
                "failed": "Failed",
                "retry": "Retry",
                "timeout": "Timeout",
                "cancelled": "Cancelled",
                
                # Mission related
                "mission.received": "Mission received",
                "mission.processing": "Mission processing",
                "mission.decomposing": "Mission decomposing",
                "mission.planning": "Mission planning",
                "mission.executing": "Mission executing",
                "mission.completed": "Mission completed",
                "mission.failed": "Mission failed",
                
                # Status messages
                "status.idle": "Idle",
                "status.busy": "Busy",
                "status.error": "Error",
                "status.maintenance": "Maintenance",
                "status.offline": "Offline", 
                "status.online": "Online"
            }
        }
    
    def set_language(self, language: str) -> bool:
        """언어 설정 변경"""
        if language in self.supported_languages:
            self.current_language = language
            logger.info(f"언어가 {language}로 변경되었습니다" if language == "ko" else f"Language changed to {language}")
            return True
        else:
            logger.warning(f"지원하지 않는 언어: {language}" if self.current_language == "ko" else f"Unsupported language: {language}")
            return False
    
    def get_message(self, key: str, **kwargs) -> str:
        """메시지 조회"""
        try:
            # 현재 언어에서 메시지 조회
            if self.current_language in self.messages:
                message = self.messages[self.current_language].get(key)
                if message:
                    return message.format(**kwargs) if kwargs else message
            
            # 기본 언어에서 메시지 조회
            if self.default_language in self.messages:
                message = self.messages[self.default_language].get(key)
                if message:
                    return message.format(**kwargs) if kwargs else message
            
            # 메시지를 찾을 수 없는 경우
            return f"[{key}]"
            
        except Exception as e:
            logger.error(f"메시지 조회 오류: {e}")
            return f"[{key}]"
    
    def get_language(self) -> str:
        """현재 언어 반환"""
        return self.current_language
    
    def get_supported_languages(self) -> list:
        """지원 언어 목록 반환"""
        return self.supported_languages.copy()
    
    def add_message(self, language: str, key: str, message: str) -> bool:
        """메시지 추가"""
        try:
            if language not in self.messages:
                self.messages[language] = {}
            
            self.messages[language][key] = message
            return True
            
        except Exception as e:
            logger.error(f"메시지 추가 오류: {e}")
            return False
    
    def load_from_file(self, file_path: str) -> bool:
        """파일에서 메시지 로드"""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"메시지 파일을 찾을 수 없음: {file_path}")
                return False
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    logger.error(f"지원하지 않는 파일 형식: {path.suffix}")
                    return False
            
            # 메시지 병합
            for language, messages in data.items():
                if language not in self.messages:
                    self.messages[language] = {}
                self.messages[language].update(messages)
            
            logger.info(f"메시지 파일 로드 완료: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"메시지 파일 로드 오류: {e}")
            return False
    
    def save_to_file(self, file_path: str) -> bool:
        """메시지를 파일로 저장"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            
            logger.info(f"메시지 파일 저장 완료: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"메시지 파일 저장 오류: {e}")
            return False

# 전역 로컬라이제이션 매니저 인스턴스
localization_manager = LocalizationManager()

# 편의 함수들
def set_language(language: str) -> bool:
    """시스템 언어 설정"""
    return localization_manager.set_language(language)

def get_message(key: str, **kwargs) -> str:
    """메시지 조회"""
    return localization_manager.get_message(key, **kwargs)

def get_language() -> str:
    """현재 언어 반환"""
    return localization_manager.get_language()

# 메시지 단축키 함수들
def msg(key: str, **kwargs) -> str:
    """메시지 조회 단축 함수"""
    return get_message(key, **kwargs)

def system_msg(action: str) -> str:
    """시스템 메시지 단축 함수"""
    return get_message(f"system.{action}")

def foundation_msg(action: str) -> str:
    """Foundation Model 메시지 단축 함수"""
    return get_message(f"foundation.{action}")

def learning_msg(action: str) -> str:
    """Learning 메시지 단축 함수"""
    return get_message(f"learning.{action}")

def execution_msg(action: str) -> str:
    """Execution 메시지 단축 함수"""
    return get_message(f"execution.{action}")

def hardware_msg(action: str) -> str:
    """Hardware 메시지 단축 함수"""
    return get_message(f"hardware.{action}")

def mission_msg(action: str) -> str:
    """Mission 메시지 단축 함수"""
    return get_message(f"mission.{action}")

def status_msg(status: str) -> str:
    """Status 메시지 단축 함수"""
    return get_message(f"status.{status}")