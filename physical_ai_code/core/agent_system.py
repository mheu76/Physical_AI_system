"""
Dynamic AI Agent System - PHI-3.5 기반 동적 에이전트 생성

사용자의 자연어 지시를 받아 PHI-3.5를 통해 
맞춤형 AI Agent를 동적으로 생성하고 관리합니다.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """에이전트 능력 정의"""
    name: str
    description: str
    tools_required: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

@dataclass  
class AgentBehavior:
    """에이전트 행동 정의"""
    trigger: str  # 활성화 조건
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 5  # 1(최고) ~ 10(최저)
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DynamicAgent:
    """동적 생성된 AI 에이전트"""
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    creator_instruction: str  # 원본 사용자 지시
    
    # 에이전트 정의
    capabilities: List[AgentCapability] = field(default_factory=list)
    behaviors: List[AgentBehavior] = field(default_factory=list)
    personality: Dict[str, Any] = field(default_factory=dict)
    
    # 실행 상태
    status: str = "active"  # active, paused, archived
    execution_count: int = 0
    success_rate: float = 0.0
    last_executed: Optional[datetime] = None
    
    # 메타데이터
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "creator_instruction": self.creator_instruction,
            "capabilities": [asdict(cap) for cap in self.capabilities],
            "behaviors": [asdict(beh) for beh in self.behaviors],
            "personality": self.personality,
            "status": self.status,
            "execution_count": self.execution_count,
            "success_rate": self.success_rate,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "tags": self.tags,
            "metadata": self.metadata
        }

class AgentTemplate:
    """에이전트 템플릿"""
    
    # 기본 에이전트 템플릿들
    TEMPLATES = {
        "assistant": {
            "name_pattern": "{task} Assistant",
            "description_pattern": "{task}를 도와주는 전문 어시스턴트",
            "base_capabilities": ["communication", "task_planning"],
            "personality_traits": {"helpful": 0.9, "proactive": 0.7, "careful": 0.8}
        },
        "specialist": {
            "name_pattern": "{domain} Specialist", 
            "description_pattern": "{domain} 분야의 전문가 에이전트",
            "base_capabilities": ["analysis", "expert_knowledge"],
            "personality_traits": {"analytical": 0.9, "precise": 0.9, "methodical": 0.8}
        },
        "automation": {
            "name_pattern": "{process} Automator",
            "description_pattern": "{process} 과정을 자동화하는 에이전트",
            "base_capabilities": ["process_execution", "monitoring"],
            "personality_traits": {"efficient": 0.9, "consistent": 0.9, "reliable": 0.8}
        },
        "learning": {
            "name_pattern": "{skill} Learner",
            "description_pattern": "{skill} 학습을 담당하는 에이전트",
            "base_capabilities": ["skill_acquisition", "practice_management"],
            "personality_traits": {"curious": 0.9, "persistent": 0.8, "adaptive": 0.9}
        }
    }
    
    @classmethod
    def get_template(cls, template_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """템플릿 가져오기"""
        if template_type not in cls.TEMPLATES:
            template_type = "assistant"
        
        template = cls.TEMPLATES[template_type].copy()
        
        # 컨텍스트로 템플릿 채우기
        for key, value in template.items():
            if isinstance(value, str) and "{" in value:
                try:
                    template[key] = value.format(**context)
                except KeyError:
                    pass  # 매칭되지 않는 키는 그대로 두기
        
        return template

class AgentSystem:
    """동적 AI 에이전트 시스템"""
    
    def __init__(self, physical_ai_interface):
        self.physical_ai = physical_ai_interface
        self.agents: Dict[str, DynamicAgent] = {}
        self.agent_storage_path = Path("agents")
        self.agent_storage_path.mkdir(exist_ok=True)
        
        # PHI-3.5 에이전트 생성 프롬프트 템플릿
        self.agent_generation_prompt = """
당신은 Physical AI System의 전문 에이전트 아키텍트입니다.
사용자의 요청을 분석하여 최적의 AI 에이전트를 설계해주세요.

사용자 요청: {user_instruction}

다음 JSON 형식으로 에이전트를 설계해주세요:

{
    "agent_type": "assistant|specialist|automation|learning",
    "name": "에이전트 이름",
    "description": "에이전트 설명",
    "capabilities": [
        {
            "name": "능력명",
            "description": "능력 설명", 
            "tools_required": ["필요한 도구들"],
            "parameters": {"매개변수": "값"}
        }
    ],
    "behaviors": [
        {
            "trigger": "활성화 조건",
            "action_sequence": [
                {"action": "실행할 동작", "parameters": {}}
            ],
            "priority": 5,
            "conditions": {"조건": "값"}
        }
    ],
    "personality": {
        "trait1": 0.8,
        "trait2": 0.7
    },
    "tags": ["태그1", "태그2"]
}

Physical AI System에서 사용 가능한 도구들:
- mission_executor: 물리적 미션 실행
- learning_system: 기술 학습
- hardware_status: 하드웨어 모니터링  
- physics_simulation: 물리 시뮬레이션
- vision_system: 컴퓨터 비전

실용적이고 구체적인 에이전트를 설계해주세요.
        """
    
    async def initialize(self):
        """에이전트 시스템 초기화"""
        await self._load_existing_agents()
        logger.info(f"Agent system initialized with {len(self.agents)} agents")
    
    async def create_agent(self, user_instruction: str) -> DynamicAgent:
        """사용자 지시로부터 새 에이전트 생성"""
        try:
            # PHI-3.5를 통한 에이전트 설계 생성
            agent_design = await self._generate_agent_design(user_instruction)
            
            # 에이전트 객체 생성
            agent = await self._create_agent_from_design(agent_design, user_instruction)
            
            # 저장 및 등록
            self.agents[agent.id] = agent
            await self._save_agent(agent)
            
            logger.info(f"Created new agent: {agent.name} ({agent.id})")
            return agent
            
        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            # 폴백: 기본 에이전트 생성
            return await self._create_fallback_agent(user_instruction)
    
    async def update_agent(self, agent_id: str, update_instruction: str) -> DynamicAgent:
        """기존 에이전트 업데이트"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        try:
            # PHI-3.5를 통한 업데이트 설계 생성
            update_design = await self._generate_update_design(agent, update_instruction)
            
            # 에이전트 업데이트 적용
            updated_agent = await self._apply_agent_updates(agent, update_design, update_instruction)
            
            # 저장
            self.agents[agent_id] = updated_agent
            await self._save_agent(updated_agent)
            
            logger.info(f"Updated agent: {updated_agent.name} ({agent_id})")
            return updated_agent
            
        except Exception as e:
            logger.error(f"Agent update failed: {e}")
            raise
    
    async def _generate_agent_design(self, user_instruction: str) -> Dict[str, Any]:
        """PHI-3.5를 통한 에이전트 설계 생성"""
        
        # PHI-3.5 모델에 설계 요청
        if self.physical_ai and self.physical_ai.slm_foundation:
            prompt = self.agent_generation_prompt.format(user_instruction=user_instruction)
            
            try:
                response = await self.physical_ai.slm_foundation.generate_response(
                    prompt, 
                    {"temperature": 0.3, "max_length": 1024}
                )
                
                # JSON 파싱 시도
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    design_json = json_match.group(0)
                    return json.loads(design_json)
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except Exception as e:
                logger.warning(f"PHI-3.5 agent design generation failed: {e}")
                return self._generate_fallback_design(user_instruction)
        else:
            return self._generate_fallback_design(user_instruction)
    
    async def _generate_update_design(self, agent: DynamicAgent, update_instruction: str) -> Dict[str, Any]:
        """에이전트 업데이트 설계 생성"""
        
        update_prompt = f"""
기존 에이전트 정보:
이름: {agent.name}
설명: {agent.description}
현재 능력: {[cap.name for cap in agent.capabilities]}

업데이트 요청: {update_instruction}

다음 형식으로 업데이트 사항을 제안해주세요:

{{
    "action": "add|modify|remove",
    "target": "capability|behavior|personality",
    "changes": {{
        "추가/수정/제거할 내용"
    }}
}}
        """
        
        if self.physical_ai and self.physical_ai.slm_foundation:
            try:
                response = await self.physical_ai.slm_foundation.generate_response(
                    update_prompt,
                    {"temperature": 0.3, "max_length": 512}
                )
                
                # JSON 파싱
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                    
            except Exception as e:
                logger.warning(f"Update design generation failed: {e}")
        
        # 폴백: 기본 업데이트 설계
        return {
            "action": "modify",
            "target": "capability", 
            "changes": {
                "name": f"Updated capability from: {update_instruction}",
                "description": update_instruction
            }
        }
    
    async def _create_agent_from_design(self, design: Dict[str, Any], user_instruction: str) -> DynamicAgent:
        """설계로부터 에이전트 생성"""
        
        agent_id = str(uuid.uuid4())
        now = datetime.now()
        
        # 템플릿 적용
        template_context = {
            "task": design.get("name", "Assistant"),
            "domain": design.get("description", "General"),
            "process": design.get("name", "Process"),
            "skill": design.get("name", "Skill")
        }
        
        template = AgentTemplate.get_template(
            design.get("agent_type", "assistant"),
            template_context
        )
        
        # 능력 생성
        capabilities = []
        for cap_data in design.get("capabilities", []):
            capability = AgentCapability(
                name=cap_data.get("name", "General Capability"),
                description=cap_data.get("description", ""),
                tools_required=cap_data.get("tools_required", []),
                parameters=cap_data.get("parameters", {})
            )
            capabilities.append(capability)
        
        # 행동 생성
        behaviors = []
        for beh_data in design.get("behaviors", []):
            behavior = AgentBehavior(
                trigger=beh_data.get("trigger", "user_request"),
                action_sequence=beh_data.get("action_sequence", []),
                priority=beh_data.get("priority", 5),
                conditions=beh_data.get("conditions", {})
            )
            behaviors.append(behavior)
        
        # 에이전트 생성
        agent = DynamicAgent(
            id=agent_id,
            name=design.get("name", template.get("name_pattern", "AI Agent")),
            description=design.get("description", template.get("description_pattern", "AI Assistant")),
            created_at=now,
            updated_at=now,
            creator_instruction=user_instruction,
            capabilities=capabilities,
            behaviors=behaviors,
            personality=design.get("personality", template.get("personality_traits", {})),
            tags=design.get("tags", [])
        )
        
        return agent
    
    async def _apply_agent_updates(
        self, 
        agent: DynamicAgent, 
        update_design: Dict[str, Any], 
        update_instruction: str
    ) -> DynamicAgent:
        """에이전트에 업데이트 적용"""
        
        action = update_design.get("action", "modify")
        target = update_design.get("target", "capability")
        changes = update_design.get("changes", {})
        
        # 업데이트 시간 갱신
        agent.updated_at = datetime.now()
        
        if target == "capability":
            if action == "add":
                new_capability = AgentCapability(
                    name=changes.get("name", "New Capability"),
                    description=changes.get("description", update_instruction),
                    tools_required=changes.get("tools_required", []),
                    parameters=changes.get("parameters", {})
                )
                agent.capabilities.append(new_capability)
                
            elif action == "modify" and agent.capabilities:
                # 첫 번째 능력 수정
                agent.capabilities[0].description = f"{agent.capabilities[0].description} | Updated: {update_instruction}"
        
        elif target == "behavior":
            if action == "add":
                new_behavior = AgentBehavior(
                    trigger=changes.get("trigger", "user_request"),
                    action_sequence=changes.get("action_sequence", [{"action": update_instruction}]),
                    priority=changes.get("priority", 5),
                    conditions=changes.get("conditions", {})
                )
                agent.behaviors.append(new_behavior)
        
        elif target == "personality":
            if action == "modify":
                agent.personality.update(changes)
        
        # 메타데이터에 업데이트 히스토리 추가
        if "update_history" not in agent.metadata:
            agent.metadata["update_history"] = []
        
        agent.metadata["update_history"].append({
            "timestamp": datetime.now().isoformat(),
            "instruction": update_instruction,
            "action": action,
            "target": target
        })
        
        return agent
    
    def _generate_fallback_design(self, user_instruction: str) -> Dict[str, Any]:
        """폴백 에이전트 설계"""
        return {
            "agent_type": "assistant",
            "name": f"Custom Assistant",
            "description": f"사용자 요청 '{user_instruction}'을 처리하는 어시스턴트",
            "capabilities": [
                {
                    "name": "Task Execution",
                    "description": user_instruction,
                    "tools_required": ["mission_executor"],
                    "parameters": {"task": user_instruction}
                }
            ],
            "behaviors": [
                {
                    "trigger": "user_request",
                    "action_sequence": [
                        {"action": "analyze_request", "parameters": {}},
                        {"action": "execute_task", "parameters": {"task": user_instruction}}
                    ],
                    "priority": 5,
                    "conditions": {}
                }
            ],
            "personality": {"helpful": 0.9, "reliable": 0.8},
            "tags": ["custom", "assistant"]
        }
    
    async def _create_fallback_agent(self, user_instruction: str) -> DynamicAgent:
        """폴백 에이전트 생성"""
        design = self._generate_fallback_design(user_instruction)
        return await self._create_agent_from_design(design, user_instruction)
    
    async def execute_agent(self, agent_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """에이전트 실행"""
        if agent_id not in self.agents:
            return {"success": False, "error": f"Agent {agent_id} not found"}
        
        agent = self.agents[agent_id]
        context = context or {}
        
        try:
            # 실행 전 상태 업데이트
            agent.last_executed = datetime.now()
            agent.execution_count += 1
            
            # 에이전트 행동 실행
            results = []
            for behavior in sorted(agent.behaviors, key=lambda b: b.priority):
                if self._check_behavior_conditions(behavior, context):
                    behavior_result = await self._execute_behavior(behavior, context)
                    results.append({
                        "behavior": behavior.trigger,
                        "result": behavior_result
                    })
            
            # 성공률 업데이트 (단순화)
            success_count = sum(1 for r in results if r["result"].get("success"))
            if results:
                agent.success_rate = (agent.success_rate * (agent.execution_count - 1) + 
                                    (success_count / len(results))) / agent.execution_count
            
            await self._save_agent(agent)
            
            return {
                "success": True,
                "agent_name": agent.name,
                "results": results,
                "execution_count": agent.execution_count,
                "success_rate": agent.success_rate
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _check_behavior_conditions(self, behavior: AgentBehavior, context: Dict[str, Any]) -> bool:
        """행동 실행 조건 확인"""
        # 단순한 조건 확인 (향후 확장 가능)
        if not behavior.conditions:
            return True
        
        for key, expected_value in behavior.conditions.items():
            if context.get(key) != expected_value:
                return False
        
        return True
    
    async def _execute_behavior(self, behavior: AgentBehavior, context: Dict[str, Any]) -> Dict[str, Any]:
        """행동 시퀀스 실행"""
        results = []
        
        for action in behavior.action_sequence:
            action_type = action.get("action")
            parameters = action.get("parameters", {})
            
            # 실제 도구 실행
            if action_type == "execute_task" and self.physical_ai:
                result = await self.physical_ai.tool_system.execute_tool(
                    "mission_executor", 
                    mission=parameters.get("task", ""),
                    **parameters
                )
                results.append({"action": action_type, "result": result})
            else:
                # 기본 동작
                results.append({
                    "action": action_type,
                    "result": {"success": True, "message": f"Executed {action_type}"}
                })
        
        return {"success": True, "actions": results}
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """에이전트 목록"""
        agent_list = []
        
        for agent in self.agents.values():
            agent_info = {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "status": agent.status,
                "created_at": agent.created_at.isoformat(),
                "execution_count": agent.execution_count,
                "success_rate": round(agent.success_rate, 2),
                "capabilities_count": len(agent.capabilities),
                "tags": agent.tags
            }
            agent_list.append(agent_info)
        
        return agent_list
    
    async def get_agent(self, agent_id: str) -> Optional[DynamicAgent]:
        """에이전트 조회"""
        return self.agents.get(agent_id)
    
    async def delete_agent(self, agent_id: str) -> bool:
        """에이전트 삭제"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = "archived"
            await self._save_agent(agent)
            del self.agents[agent_id]
            logger.info(f"Deleted agent: {agent.name} ({agent_id})")
            return True
        return False
    
    async def _save_agent(self, agent: DynamicAgent):
        """에이전트 저장"""
        agent_file = self.agent_storage_path / f"{agent.id}.json"
        with open(agent_file, 'w', encoding='utf-8') as f:
            json.dump(agent.to_dict(), f, ensure_ascii=False, indent=2)
    
    async def _load_existing_agents(self):
        """기존 에이전트들 로드"""
        if not self.agent_storage_path.exists():
            return
        
        for agent_file in self.agent_storage_path.glob("*.json"):
            try:
                with open(agent_file, 'r', encoding='utf-8') as f:
                    agent_data = json.load(f)
                
                agent = self._agent_from_dict(agent_data)
                if agent.status == "active":
                    self.agents[agent.id] = agent
                    
            except Exception as e:
                logger.error(f"Failed to load agent from {agent_file}: {e}")
    
    def _agent_from_dict(self, data: Dict[str, Any]) -> DynamicAgent:
        """딕셔너리로부터 에이전트 복원"""
        capabilities = [
            AgentCapability(**cap_data) 
            for cap_data in data.get("capabilities", [])
        ]
        
        behaviors = [
            AgentBehavior(**beh_data)
            for beh_data in data.get("behaviors", [])
        ]
        
        return DynamicAgent(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            creator_instruction=data["creator_instruction"],
            capabilities=capabilities,
            behaviors=behaviors,
            personality=data.get("personality", {}),
            status=data.get("status", "active"),
            execution_count=data.get("execution_count", 0),
            success_rate=data.get("success_rate", 0.0),
            last_executed=datetime.fromisoformat(data["last_executed"]) if data.get("last_executed") else None,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )