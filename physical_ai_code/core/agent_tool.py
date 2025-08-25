"""
Agent Management Tool - /agent 명령어 처리

PHI-3.5 기반 동적 AI Agent 생성 및 관리를 위한 도구
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .tool_system import PhysicalAITool, ToolResult, ToolParameter
from .agent_system import AgentSystem

logger = logging.getLogger(__name__)

class AgentTool(PhysicalAITool):
    """AI Agent 관리 도구"""
    
    def __init__(self):
        super().__init__(
            name="agent_manager",
            description="PHI-3.5를 사용하여 AI Agent를 동적으로 생성, 업데이트, 관리합니다"
        )
        
        # 매개변수 정의
        self.add_parameter(ToolParameter(
            name="action",
            type="string",
            description="실행할 동작",
            enum_values=["create", "update", "execute", "list", "info", "delete"]
        ))
        
        self.add_parameter(ToolParameter(
            name="instruction",
            type="string", 
            description="에이전트 생성/업데이트 지시사항",
            required=False
        ))
        
        self.add_parameter(ToolParameter(
            name="agent_id",
            type="string",
            description="대상 에이전트 ID",
            required=False
        ))
        
        self.add_parameter(ToolParameter(
            name="agent_name",
            type="string",
            description="에이전트 이름 (검색용)",
            required=False
        ))
        
        self.add_parameter(ToolParameter(
            name="context",
            type="object",
            description="실행 컨텍스트",
            required=False,
            default={}
        ))
        
        self.agent_system: Optional[AgentSystem] = None
    
    def set_physical_ai(self, physical_ai):
        """Physical AI 인스턴스 설정"""
        super().set_physical_ai(physical_ai)
        
        # Agent System 초기화
        if not self.agent_system:
            self.agent_system = AgentSystem(physical_ai)
            asyncio.create_task(self.agent_system.initialize())
    
    async def execute(self, **kwargs) -> ToolResult:
        """도구 실행"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            params = self.validate_parameters(kwargs)
            action = params["action"]
            
            if not self.agent_system:
                return ToolResult(
                    success=False,
                    error="Agent system not initialized"
                )
            
            # 액션별 처리
            if action == "create":
                result = await self._create_agent(params)
            elif action == "update":
                result = await self._update_agent(params)
            elif action == "execute":
                result = await self._execute_agent(params)
            elif action == "list":
                result = await self._list_agents(params)
            elif action == "info":
                result = await self._get_agent_info(params)
            elif action == "delete":
                result = await self._delete_agent(params)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ToolResult(
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"action": action, "params": params}
            )
            
        except Exception as e:
            logger.error(f"Agent tool execution failed: {e}")
            return ToolResult(
                success=False,
                error=f"Agent operation failed: {str(e)}"
            )
    
    async def _create_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 생성"""
        instruction = params.get("instruction")
        if not instruction:
            return {
                "success": False,
                "error": "Agent creation instruction is required"
            }
        
        try:
            agent = await self.agent_system.create_agent(instruction)
            
            return {
                "success": True,
                "action": "create",
                "agent": {
                    "id": agent.id,
                    "name": agent.name,
                    "description": agent.description,
                    "capabilities": [cap.name for cap in agent.capabilities],
                    "behaviors_count": len(agent.behaviors),
                    "personality": agent.personality,
                    "tags": agent.tags,
                    "created_at": agent.created_at.isoformat()
                },
                "message": f"새로운 에이전트 '{agent.name}'가 성공적으로 생성되었습니다!"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Agent creation failed: {str(e)}"
            }
    
    async def _update_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 업데이트"""
        agent_id = params.get("agent_id")
        agent_name = params.get("agent_name")
        instruction = params.get("instruction")
        
        if not instruction:
            return {
                "success": False,
                "error": "Update instruction is required"
            }
        
        # 에이전트 ID 찾기
        if not agent_id and agent_name:
            agent_id = await self._find_agent_by_name(agent_name)
            if not agent_id:
                return {
                    "success": False,
                    "error": f"Agent named '{agent_name}' not found"
                }
        
        if not agent_id:
            return {
                "success": False,
                "error": "Agent ID or name is required"
            }
        
        try:
            updated_agent = await self.agent_system.update_agent(agent_id, instruction)
            
            return {
                "success": True,
                "action": "update",
                "agent": {
                    "id": updated_agent.id,
                    "name": updated_agent.name,
                    "description": updated_agent.description,
                    "capabilities": [cap.name for cap in updated_agent.capabilities],
                    "updated_at": updated_agent.updated_at.isoformat(),
                    "update_history": updated_agent.metadata.get("update_history", [])[-3:]  # 최근 3개
                },
                "message": f"에이전트 '{updated_agent.name}'가 성공적으로 업데이트되었습니다!"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Agent update failed: {str(e)}"
            }
    
    async def _execute_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 실행"""
        agent_id = params.get("agent_id")
        agent_name = params.get("agent_name")
        context = params.get("context", {})
        
        # 에이전트 ID 찾기
        if not agent_id and agent_name:
            agent_id = await self._find_agent_by_name(agent_name)
            if not agent_id:
                return {
                    "success": False,
                    "error": f"Agent named '{agent_name}' not found"
                }
        
        if not agent_id:
            return {
                "success": False,
                "error": "Agent ID or name is required"
            }
        
        execution_result = await self.agent_system.execute_agent(agent_id, context)
        
        if execution_result["success"]:
            return {
                "success": True,
                "action": "execute",
                "agent_name": execution_result["agent_name"],
                "execution_count": execution_result["execution_count"],
                "success_rate": round(execution_result["success_rate"], 2),
                "results": execution_result["results"],
                "message": f"에이전트 '{execution_result['agent_name']}'를 성공적으로 실행했습니다!"
            }
        else:
            return {
                "success": False,
                "error": execution_result.get("error", "Unknown execution error")
            }
    
    async def _list_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 목록"""
        agents = await self.agent_system.list_agents()
        
        return {
            "success": True,
            "action": "list",
            "agents": agents,
            "count": len(agents),
            "message": f"총 {len(agents)}개의 에이전트가 등록되어 있습니다."
        }
    
    async def _get_agent_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 상세 정보"""
        agent_id = params.get("agent_id")
        agent_name = params.get("agent_name")
        
        # 에이전트 ID 찾기
        if not agent_id and agent_name:
            agent_id = await self._find_agent_by_name(agent_name)
            if not agent_id:
                return {
                    "success": False,
                    "error": f"Agent named '{agent_name}' not found"
                }
        
        if not agent_id:
            return {
                "success": False,
                "error": "Agent ID or name is required"
            }
        
        agent = await self.agent_system.get_agent(agent_id)
        if not agent:
            return {
                "success": False,
                "error": f"Agent {agent_id} not found"
            }
        
        return {
            "success": True,
            "action": "info",
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat(),
                "creator_instruction": agent.creator_instruction,
                "status": agent.status,
                "execution_count": agent.execution_count,
                "success_rate": round(agent.success_rate, 2),
                "last_executed": agent.last_executed.isoformat() if agent.last_executed else None,
                "capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "tools_required": cap.tools_required
                    }
                    for cap in agent.capabilities
                ],
                "behaviors": [
                    {
                        "trigger": beh.trigger,
                        "priority": beh.priority,
                        "action_count": len(beh.action_sequence)
                    }
                    for beh in agent.behaviors
                ],
                "personality": agent.personality,
                "tags": agent.tags,
                "update_history": agent.metadata.get("update_history", [])
            },
            "message": f"에이전트 '{agent.name}'의 상세 정보입니다."
        }
    
    async def _delete_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 삭제"""
        agent_id = params.get("agent_id")
        agent_name = params.get("agent_name")
        
        # 에이전트 ID 찾기
        if not agent_id and agent_name:
            agent_id = await self._find_agent_by_name(agent_name)
            if not agent_id:
                return {
                    "success": False,
                    "error": f"Agent named '{agent_name}' not found"
                }
        
        if not agent_id:
            return {
                "success": False,
                "error": "Agent ID or name is required"
            }
        
        # 에이전트 정보 조회 (삭제 전)
        agent = await self.agent_system.get_agent(agent_id)
        if not agent:
            return {
                "success": False,
                "error": f"Agent {agent_id} not found"
            }
        
        deleted = await self.agent_system.delete_agent(agent_id)
        
        if deleted:
            return {
                "success": True,
                "action": "delete",
                "agent_name": agent.name,
                "message": f"에이전트 '{agent.name}'가 성공적으로 삭제되었습니다."
            }
        else:
            return {
                "success": False,
                "error": "Failed to delete agent"
            }
    
    async def _find_agent_by_name(self, agent_name: str) -> Optional[str]:
        """이름으로 에이전트 ID 찾기"""
        agents = await self.agent_system.list_agents()
        
        for agent in agents:
            if agent["name"].lower() == agent_name.lower():
                return agent["id"]
        
        return None