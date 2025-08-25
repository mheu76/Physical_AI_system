"""
Physical AI Session Manager - 세션 관리

Claude Code 스타일의 세션 관리 시스템
"""

import asyncio
import uuid
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import weakref

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """대화 턴"""
    timestamp: datetime
    type: str  # user, assistant, system
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)

@dataclass
class Session:
    """세션 정보"""
    id: str
    created_at: datetime
    last_activity: datetime
    status: str = "active"  # active, paused, closed
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "status": self.status,
            "conversation_history": [
                {
                    "timestamp": turn.timestamp.isoformat(),
                    "type": turn.type,
                    "content": turn.content,
                    "metadata": turn.metadata,
                    "tools_used": turn.tools_used
                }
                for turn in self.conversation_history
            ],
            "context": self.context,
            "metadata": self.metadata
        }

class SessionManager:
    """세션 관리자"""
    
    def __init__(self, max_sessions: int = 100, session_timeout: int = 3600):
        self.sessions: Dict[str, Session] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout  # seconds
        self._cleanup_task = None
        
    async def create_session(self, context: Optional[Dict[str, Any]] = None) -> Session:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = Session(
            id=session_id,
            created_at=now,
            last_activity=now,
            context=context or {},
            metadata={
                "user_agent": "Physical AI Code",
                "version": "1.0.0"
            }
        )
        
        self.sessions[session_id] = session
        
        # 세션 수 제한
        if len(self.sessions) > self.max_sessions:
            await self._cleanup_old_sessions()
        
        # 자동 정리 작업 시작
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        logger.info(f"New session created: {session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """세션 조회"""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = datetime.now()
            return session
        return None
    
    async def update_session_context(self, session_id: str, context_update: Dict[str, Any]):
        """세션 컨텍스트 업데이트"""
        session = await self.get_session(session_id)
        if session:
            session.context.update(context_update)
            session.last_activity = datetime.now()
    
    async def add_conversation_turn(
        self, 
        session_id: str, 
        turn_type: str, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tools_used: Optional[List[str]] = None
    ):
        """대화 턴 추가"""
        session = await self.get_session(session_id)
        if session:
            turn = ConversationTurn(
                timestamp=datetime.now(),
                type=turn_type,
                content=content,
                metadata=metadata or {},
                tools_used=tools_used or []
            )
            
            session.conversation_history.append(turn)
            session.last_activity = datetime.now()
            
            # 대화 히스토리 길이 제한
            max_history = 100
            if len(session.conversation_history) > max_history:
                session.conversation_history = session.conversation_history[-max_history:]
    
    async def get_conversation_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None
    ) -> List[ConversationTurn]:
        """대화 히스토리 조회"""
        session = await self.get_session(session_id)
        if session:
            history = session.conversation_history
            if limit:
                history = history[-limit:]
            return history
        return []
    
    async def pause_session(self, session_id: str):
        """세션 일시정지"""
        session = await self.get_session(session_id)
        if session:
            session.status = "paused"
            session.last_activity = datetime.now()
            logger.info(f"Session paused: {session_id}")
    
    async def resume_session(self, session_id: str):
        """세션 재개"""
        session = await self.get_session(session_id)
        if session:
            session.status = "active"
            session.last_activity = datetime.now()
            logger.info(f"Session resumed: {session_id}")
    
    async def close_session(self, session_id: str):
        """세션 종료"""
        session = self.sessions.get(session_id)
        if session:
            session.status = "closed"
            session.last_activity = datetime.now()
            
            # 세션 데이터 저장 (필요한 경우)
            await self._save_session_data(session)
            
            # 메모리에서 제거
            del self.sessions[session_id]
            logger.info(f"Session closed: {session_id}")
    
    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """활성 세션 목록"""
        active_sessions = []
        
        for session in self.sessions.values():
            if session.status == "active":
                active_sessions.append({
                    "id": session.id,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "conversation_turns": len(session.conversation_history),
                    "context_keys": list(session.context.keys())
                })
        
        return active_sessions
    
    async def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 통계"""
        session = await self.get_session(session_id)
        if session:
            user_turns = sum(1 for turn in session.conversation_history if turn.type == "user")
            assistant_turns = sum(1 for turn in session.conversation_history if turn.type == "assistant")
            
            tools_used = set()
            for turn in session.conversation_history:
                tools_used.update(turn.tools_used)
            
            duration = session.last_activity - session.created_at
            
            return {
                "session_id": session.id,
                "status": session.status,
                "duration": duration.total_seconds(),
                "total_turns": len(session.conversation_history),
                "user_turns": user_turns,
                "assistant_turns": assistant_turns,
                "unique_tools_used": len(tools_used),
                "tools_used": list(tools_used),
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat()
            }
        return None
    
    async def search_sessions(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """세션 검색"""
        results = []
        query_lower = query.lower()
        
        for session in self.sessions.values():
            # 대화 내용에서 검색
            matches = []
            for turn in session.conversation_history:
                if query_lower in turn.content.lower():
                    matches.append({
                        "timestamp": turn.timestamp.isoformat(),
                        "type": turn.type,
                        "content": turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
                    })
            
            if matches:
                results.append({
                    "session_id": session.id,
                    "created_at": session.created_at.isoformat(),
                    "matches": matches[:3]  # 최대 3개 매치만 반환
                })
            
            if len(results) >= limit:
                break
        
        return results
    
    async def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 내보내기"""
        session = await self.get_session(session_id)
        if session:
            return session.to_dict()
        return None
    
    async def import_session(self, session_data: Dict[str, Any]) -> str:
        """세션 가져오기"""
        session_id = session_data["id"]
        
        session = Session(
            id=session_id,
            created_at=datetime.fromisoformat(session_data["created_at"]),
            last_activity=datetime.fromisoformat(session_data["last_activity"]),
            status=session_data["status"],
            context=session_data["context"],
            metadata=session_data["metadata"]
        )
        
        # 대화 히스토리 복원
        for turn_data in session_data["conversation_history"]:
            turn = ConversationTurn(
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                type=turn_data["type"],
                content=turn_data["content"],
                metadata=turn_data["metadata"],
                tools_used=turn_data["tools_used"]
            )
            session.conversation_history.append(turn)
        
        self.sessions[session_id] = session
        logger.info(f"Session imported: {session_id}")
        return session_id
    
    async def _cleanup_old_sessions(self):
        """오래된 세션 정리"""
        now = datetime.now()
        timeout_threshold = now - timedelta(seconds=self.session_timeout)
        
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if session.last_activity < timeout_threshold and session.status != "active":
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.close_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _periodic_cleanup(self):
        """주기적 정리 작업"""
        while True:
            try:
                await asyncio.sleep(300)  # 5분마다 실행
                await self._cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")
    
    async def _save_session_data(self, session: Session):
        """세션 데이터 저장 (필요한 경우 구현)"""
        # 파일, 데이터베이스 등에 세션 데이터 저장
        # 현재는 로그만 남김
        logger.debug(f"Session data saved for session: {session.id}")
    
    async def shutdown(self):
        """세션 매니저 종료"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 모든 활성 세션 저장
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)
        
        logger.info("Session manager shut down")
    
    def __len__(self) -> int:
        """활성 세션 수"""
        return len(self.sessions)