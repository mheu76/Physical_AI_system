"""
Physical AI Interface Manager - 메인 통합 인터페이스

Claude Code 스타일의 통합된 AI 인터페이스를 제공합니다.
모든 Physical AI 기능을 하나의 일관된 대화형 환경에서 사용할 수 있습니다.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import sys
from pathlib import Path

# Physical AI 시스템 컴포넌트들 - import 경로 수정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 안전한 import
try:
    from main import PhysicalAI
except ImportError:
    # 폴백: 직접 구현
    PhysicalAI = None
    logger.warning("PhysicalAI 클래스를 main.py에서 가져올 수 없습니다. 기본 구현을 사용합니다.")

from .command_processor import CommandProcessor
from .tool_system import ToolSystem
from .session_manager import SessionManager

logger = logging.getLogger(__name__)

@dataclass
class InterfaceConfig:
    """인터페이스 설정"""
    language: str = "ko"
    interface_mode: str = "interactive"  # interactive, batch, api
    output_format: str = "rich"  # rich, plain, json
    enable_tools: bool = True
    enable_voice: bool = False
    debug_mode: bool = False

class PhysicalAIInterface:
    """
    Physical AI Code - 통합 인터페이스
    
    Claude Code와 같은 방식으로 Physical AI 시스템의 모든 기능을
    하나의 대화형 인터페이스에서 사용할 수 있게 합니다.
    """
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = config_path
        self.interface_config = InterfaceConfig()
        
        # 핵심 컴포넌트 초기화
        self.physical_ai = None
        self.command_processor = CommandProcessor()
        self.tool_system = ToolSystem()
        self.session_manager = SessionManager()
        
        # 상태 관리
        self.is_initialized = False
        self.current_session = None
        self.conversation_history = []
        
        logger.info("🤖 Physical AI Code 인터페이스 초기화 완료")
    
    async def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            # Physical AI 시스템 초기화
            if PhysicalAI:
                self.physical_ai = PhysicalAI(self.config_path)
                if hasattr(self.physical_ai, 'initialize'):
                    await self.physical_ai.initialize()
                else:
                    # 동기 초기화인 경우
                    pass
            else:
                # 폴백: 기본 Physical AI 객체 생성
                self.physical_ai = self._create_fallback_physical_ai()
            
            # 도구 시스템 초기화
            await self.tool_system.initialize(self.physical_ai)
            
            # 세션 시작
            self.current_session = await self.session_manager.create_session()
            
            self.is_initialized = True
            logger.info("✅ Physical AI Code 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}")
            # 오류가 발생해도 기본 기능은 동작하도록 함
            try:
                await self.tool_system.initialize(None)
                self.current_session = await self.session_manager.create_session()
                self.is_initialized = True
                logger.info("⚠️ 제한된 모드로 초기화 완료")
                return True
            except Exception as e2:
                logger.error(f"❌ 제한된 모드 초기화도 실패: {e2}")
                return False
    
    async def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        사용자 입력 처리 - Claude Code 스타일
        
        Args:
            user_input: 사용자의 자연어 입력
            
        Returns:
            처리 결과 딕셔너리
        """
        if not self.is_initialized:
            await self.initialize()
        
        # 대화 히스토리에 추가
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "type": "user",
            "content": user_input
        })
        
        try:
            # 1. 명령 분석 및 분류
            command_analysis = await self.command_processor.analyze_command(user_input)
            
            # 2. 도구 실행 필요 여부 판단
            if command_analysis.get("requires_tools"):
                tool_result = await self.tool_system.execute_tools(
                    command_analysis["tools"], 
                    user_input
                )
            else:
                tool_result = None
            
            # 3. PHI-3.5 모델로 응답 생성
            response = await self._generate_response(
                user_input, 
                command_analysis, 
                tool_result
            )
            
            # 4. 응답 히스토리에 추가
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "type": "assistant", 
                "content": response["content"],
                "tools_used": tool_result.get("tools_used", []) if tool_result else []
            })
            
            return {
                "success": True,
                "response": response,
                "tool_results": tool_result,
                "session_id": self.current_session.id
            }
            
        except Exception as e:
            logger.error(f"입력 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": {"content": f"죄송합니다. 처리 중 오류가 발생했습니다: {e}"}
            }
    
    async def _generate_response(
        self, 
        user_input: str, 
        command_analysis: Dict[str, Any],
        tool_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """응답 생성"""
        
        # 컨텍스트 구성
        context = {
            "user_input": user_input,
            "command_type": command_analysis.get("type", "general"),
            "conversation_history": self.conversation_history[-5:],  # 최근 5개
            "tool_results": tool_result,
            "system_status": await self._get_system_status()
        }
        
        # PHI-3.5 모델로 응답 생성
        if self.physical_ai and self.physical_ai.slm_foundation:
            ai_response = await self.physical_ai.slm_foundation.generate_response(
                user_input, 
                context
            )
        else:
            # 폴백 응답
            ai_response = await self._generate_fallback_response(user_input, context)
        
        return {
            "content": ai_response,
            "type": "text",
            "metadata": {
                "command_type": command_analysis.get("type"),
                "confidence": command_analysis.get("confidence", 0.8),
                "processing_time": datetime.now()
            }
        }
    
    async def _generate_fallback_response(self, user_input: str, context: Dict) -> str:
        """폴백 응답 생성"""
        command_type = context.get("command_type", "general")
        
        if command_type == "mission":
            return f"미션 '{user_input}'을 분석하고 실행 계획을 수립하겠습니다."
        elif command_type == "hardware":
            return "하드웨어 상태를 확인하고 요청된 작업을 수행하겠습니다."
        elif command_type == "learning":
            return "학습 시스템을 통해 새로운 기술을 습득하거나 개선하겠습니다."
        else:
            return f"'{user_input}' 요청을 처리하겠습니다. 더 구체적인 지시가 필요하시면 말씀해 주세요."
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            "initialized": self.is_initialized,
            "physical_ai_ready": bool(self.physical_ai and self.physical_ai.system_ready),
            "tools_available": len(self.tool_system.available_tools),
            "session_active": bool(self.current_session),
            "timestamp": datetime.now()
        }
    
    async def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """명령 직접 실행"""
        return await self.command_processor.execute_command(command, **kwargs)
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 목록"""
        return await self.tool_system.list_tools()
    
    async def get_help(self, topic: str = None) -> str:
        """도움말 생성"""
        if topic:
            return await self._get_topic_help(topic)
        
        return """
🤖 Physical AI Code - 통합 인터페이스

사용 가능한 명령어:
• /mission <작업> - 물리적 미션 실행
• /learn <기술> - 새로운 기술 학습
• /hardware - 하드웨어 상태 확인
• /simulate - 시뮬레이션 실행
• /status - 시스템 상태 확인
• /tools - 사용 가능한 도구 목록
• /help <주제> - 특정 주제 도움말

자연어로도 명령할 수 있습니다:
"로봇아, 빨간 컵을 테이블로 옮겨줘"
"새로운 잡기 동작을 학습해줘"
"시뮬레이션에서 테스트해보자"
        """
    
    async def _get_topic_help(self, topic: str) -> str:
        """특정 주제 도움말"""
        help_topics = {
            "mission": "미션 실행: /mission '빨간 공을 상자에 넣어줘'",
            "learning": "학습 시스템: /learn 'grasp_skill' 또는 '잡기 동작을 연습해줘'",
            "hardware": "하드웨어 제어: /hardware status 또는 '로봇 상태 확인해줘'",
            "simulation": "시뮬레이션: /simulate physics 또는 '시뮬레이션 시작'",
        }
        
        return help_topics.get(topic, f"'{topic}' 주제에 대한 도움말을 찾을 수 없습니다.")
    
    async def shutdown(self):
        """시스템 종료"""
        if self.current_session:
            await self.session_manager.close_session(self.current_session.id)
        
        if self.physical_ai:
            await self.physical_ai.shutdown()
        
        logger.info("👋 Physical AI Code 종료됨")
    
    def _create_fallback_physical_ai(self):
        """폴백 Physical AI 객체 생성"""
        class FallbackPhysicalAI:
            def __init__(self):
                self.system_ready = True
                self.slm_foundation = None
                
            async def execute_mission(self, mission):
                return {
                    "success": True,
                    "message": f"시뮬레이션 모드에서 미션 '{mission}' 실행됨"
                }
                
            async def shutdown(self):
                pass
        
        return FallbackPhysicalAI()