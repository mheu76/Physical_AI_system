"""
Physical AI Command Processor - 명령 분석 및 처리

Claude Code 스타일의 자연어 명령을 분석하고 
적절한 도구 실행으로 변환합니다.
"""

import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass 
class CommandAnalysis:
    """명령 분석 결과"""
    type: str  # mission, learning, hardware, simulation, general
    intent: str  # 사용자의 의도
    entities: Dict[str, Any]  # 추출된 개체
    confidence: float  # 분석 신뢰도
    requires_tools: bool  # 도구 실행 필요 여부
    tools: List[Dict[str, Any]]  # 실행할 도구 목록
    parameters: Dict[str, Any]  # 추가 매개변수

class CommandProcessor:
    """명령 처리기"""
    
    def __init__(self):
        self.command_patterns = self._init_command_patterns()
        self.entity_extractors = self._init_entity_extractors()
    
    def _init_command_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """명령 패턴 초기화"""
        return {
            "mission": [
                {
                    "pattern": r"(?:로봇아|로봇|ai)?\s*(.+?)(?:해줘|하자|해보자|실행해|수행해)",
                    "intent": "execute_task",
                    "confidence": 0.9
                },
                {
                    "pattern": r"(?:미션|작업|task)\s*[:\-]?\s*(.+)",
                    "intent": "execute_mission", 
                    "confidence": 0.95
                },
                {
                    "pattern": r"(.+?)(?:을|를)\s*(.+?)(?:에|로)\s*(?:옮겨|이동|놓아|두어)(?:줘|라)?",
                    "intent": "move_object",
                    "confidence": 0.85
                }
            ],
            "learning": [
                {
                    "pattern": r"(?:학습|배워|연습|훈련)(?:해줘|하자|시작)",
                    "intent": "start_learning",
                    "confidence": 0.9
                },
                {
                    "pattern": r"(.+?)\s*(?:스킬|기술|동작)(?:을|를)?\s*(?:학습|배워|연습)",
                    "intent": "learn_skill",
                    "confidence": 0.85
                },
                {
                    "pattern": r"(?:더\s*)?(?:잘|정확히|정밀하게)\s*(.+?)(?:하도록|할 수 있도록)\s*(?:학습|개선)",
                    "intent": "improve_skill",
                    "confidence": 0.8
                }
            ],
            "hardware": [
                {
                    "pattern": r"(?:하드웨어|로봇|시스템)\s*(?:상태|상황|점검)(?:을|를)?\s*(?:확인|체크|보여줘)",
                    "intent": "check_hardware",
                    "confidence": 0.9
                },
                {
                    "pattern": r"(?:센서|모터|관절|actuator|sensor)\s*(?:상태|정보)",
                    "intent": "check_component",
                    "confidence": 0.85
                }
            ],
            "simulation": [
                {
                    "pattern": r"(?:시뮬레이션|simulation|시뮬|테스트)(?:을|를)?\s*(?:시작|실행|해보자)",
                    "intent": "run_simulation",
                    "confidence": 0.9
                },
                {
                    "pattern": r"(?:가상|virtual|mock)\s*(?:환경|세계)에서\s*(.+)",
                    "intent": "simulate_task",
                    "confidence": 0.85
                }
            ],
            "vision": [
                {
                    "pattern": r"(?:무엇이|뭐가|객체|물체)(?:를|을)?\s*(?:보이나|있나|감지)",
                    "intent": "object_detection",
                    "confidence": 0.85
                },
                {
                    "pattern": r"(?:카메라|camera|비전|vision)(?:로|으로)?\s*(.+?)(?:분석|확인|인식)",
                    "intent": "vision_analysis",
                    "confidence": 0.8
                }
            ]
        }
    
    def _init_entity_extractors(self) -> Dict[str, List[Dict[str, Any]]]:
        """개체 추출기 초기화"""
        return {
            "objects": [
                {
                    "pattern": r"(빨간|파란|노란|초록|검은|하얀|투명한)?\s*(컵|cup|병|bottle|공|ball|박스|box|상자)",
                    "type": "physical_object"
                },
                {
                    "pattern": r"(테이블|table|책상|desk|바닥|floor|선반|shelf)",
                    "type": "surface"
                }
            ],
            "actions": [
                {
                    "pattern": r"(집어|잡아|grasp|pick|옮겨|move|놓아|place|두어|put)",
                    "type": "manipulation"
                },
                {
                    "pattern": r"(이동|move|회전|rotate|돌려|turn)",
                    "type": "motion"
                }
            ],
            "skills": [
                {
                    "pattern": r"(잡기|grasp|집기|pick|조작|manipulation|이동|movement)",
                    "type": "motor_skill"
                }
            ],
            "components": [
                {
                    "pattern": r"(관절|joint|모터|motor|센서|sensor|카메라|camera|그리퍼|gripper)",
                    "type": "hardware_component"
                }
            ]
        }
    
    async def analyze_command(self, user_input: str) -> Dict[str, Any]:
        """명령 분석"""
        user_input = user_input.strip()
        
        # 1. 슬래시 명령어 처리
        if user_input.startswith('/'):
            return await self._analyze_slash_command(user_input)
        
        # 2. 자연어 명령 분석
        return await self._analyze_natural_language(user_input)
    
    async def _analyze_slash_command(self, command: str) -> Dict[str, Any]:
        """슬래시 명령어 분석"""
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        command_map = {
            "mission": {
                "type": "mission",
                "intent": "execute_mission",
                "tools": [{"name": "mission_executor", "parameters": {"mission": args}}],
                "confidence": 1.0
            },
            "learn": {
                "type": "learning", 
                "intent": "learn_skill",
                "tools": [{"name": "learning_system", "parameters": {"skill_name": args}}],
                "confidence": 1.0
            },
            "hardware": {
                "type": "hardware",
                "intent": "check_hardware",
                "tools": [{"name": "hardware_status", "parameters": {"component": args or "all"}}],
                "confidence": 1.0
            },
            "simulate": {
                "type": "simulation",
                "intent": "run_simulation", 
                "tools": [{"name": "physics_simulation", "parameters": {"scenario": args or "basic_environment"}}],
                "confidence": 1.0
            },
            "vision": {
                "type": "vision",
                "intent": "object_detection",
                "tools": [{"name": "vision_system", "parameters": {"task": args or "object_detection"}}],
                "confidence": 1.0
            },
            "agent": {
                "type": "agent_management",
                "intent": "manage_agents",
                "tools": [{"name": "agent_manager", "parameters": self._parse_agent_command(args)}],
                "confidence": 1.0
            }
        }
        
        if cmd in command_map:
            analysis = command_map[cmd]
            analysis["requires_tools"] = True
            analysis["entities"] = {}
            analysis["parameters"] = {"raw_args": args}
            return analysis
        else:
            return {
                "type": "unknown",
                "intent": "unknown_command",
                "entities": {},
                "confidence": 0.0,
                "requires_tools": False,
                "tools": [],
                "parameters": {"command": cmd, "args": args}
            }
    
    async def _analyze_natural_language(self, text: str) -> Dict[str, Any]:
        """자연어 명령 분석"""
        best_match = None
        best_score = 0.0
        
        # 각 카테고리별 패턴 매칭
        for category, patterns in self.command_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                match = re.search(pattern, text, re.IGNORECASE)
                
                if match:
                    score = pattern_info["confidence"]
                    if score > best_score:
                        best_score = score
                        best_match = {
                            "type": category,
                            "intent": pattern_info["intent"],
                            "confidence": score,
                            "match": match,
                            "groups": match.groups() if match.groups() else []
                        }
        
        if not best_match:
            # 매칭되지 않은 경우 일반 명령으로 처리
            return {
                "type": "general",
                "intent": "general_query",
                "entities": {},
                "confidence": 0.5,
                "requires_tools": False,
                "tools": [],
                "parameters": {"raw_text": text}
            }
        
        # 개체 추출
        entities = await self._extract_entities(text)
        
        # 도구 매핑
        tools = await self._map_to_tools(best_match, entities, text)
        
        return {
            "type": best_match["type"],
            "intent": best_match["intent"],
            "entities": entities,
            "confidence": best_match["confidence"],
            "requires_tools": len(tools) > 0,
            "tools": tools,
            "parameters": {
                "raw_text": text,
                "matched_groups": best_match["groups"]
            }
        }
    
    async def _extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """개체 추출"""
        entities = {}
        
        for entity_type, extractors in self.entity_extractors.items():
            entities[entity_type] = []
            
            for extractor in extractors:
                pattern = extractor["pattern"]
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity = {
                        "text": match.group(0),
                        "type": extractor["type"],
                        "start": match.start(),
                        "end": match.end(),
                        "groups": match.groups() if match.groups() else []
                    }
                    entities[entity_type].append(entity)
        
        return entities
    
    async def _map_to_tools(self, match_info: Dict, entities: Dict, text: str) -> List[Dict[str, Any]]:
        """의도와 개체를 도구로 매핑"""
        tools = []
        
        if match_info["type"] == "mission":
            # 미션 실행 도구
            mission_text = text
            if match_info["groups"]:
                mission_text = match_info["groups"][0]
            
            tools.append({
                "name": "mission_executor",
                "parameters": {
                    "mission": mission_text,
                    "timeout": 300
                }
            })
            
        elif match_info["type"] == "learning":
            # 학습 도구
            skill_name = "general_skill"
            if match_info["groups"]:
                skill_name = match_info["groups"][0]
            elif entities.get("skills"):
                skill_name = entities["skills"][0]["text"]
            
            learning_type = "new_skill"
            if "개선" in text or "더" in text:
                learning_type = "improve_existing"
            
            tools.append({
                "name": "learning_system",
                "parameters": {
                    "skill_name": skill_name,
                    "learning_type": learning_type,
                    "practice_iterations": 10
                }
            })
            
        elif match_info["type"] == "hardware":
            # 하드웨어 상태 확인 도구
            component = "all"
            if entities.get("components"):
                component = entities["components"][0]["text"]
            
            tools.append({
                "name": "hardware_status",
                "parameters": {
                    "component": component
                }
            })
            
        elif match_info["type"] == "simulation":
            # 시뮬레이션 도구
            scenario = "basic_environment"
            if match_info["groups"]:
                scenario = match_info["groups"][0]
            
            tools.append({
                "name": "physics_simulation", 
                "parameters": {
                    "scenario": scenario,
                    "duration": 60.0,
                    "gui_mode": True
                }
            })
            
        elif match_info["type"] == "vision":
            # 비전 시스템 도구
            task = "object_detection"
            if "분석" in text:
                task = "scene_analysis"
            elif "자세" in text or "pose" in text.lower():
                task = "pose_estimation"
            
            tools.append({
                "name": "vision_system",
                "parameters": {
                    "task": task,
                    "camera_id": 0
                }
            })
        
        return tools
    
    def _parse_agent_command(self, args: str) -> Dict[str, Any]:
        """에이전트 명령어 파싱"""
        if not args:
            return {"action": "list"}
        
        # 기본 패턴들
        if args.startswith("create "):
            instruction = args[7:].strip()  # "create " 제거
            return {
                "action": "create",
                "instruction": instruction
            }
        elif args.startswith("update "):
            # "/agent update agent_name 업데이트 내용" 형식
            parts = args[7:].split(maxsplit=1)  # "update " 제거
            if len(parts) >= 2:
                return {
                    "action": "update",
                    "agent_name": parts[0],
                    "instruction": parts[1]
                }
            else:
                return {
                    "action": "update",
                    "instruction": parts[0] if parts else ""
                }
        elif args.startswith("run ") or args.startswith("execute "):
            # "/agent run agent_name" 형식
            prefix_len = 4 if args.startswith("run ") else 8
            agent_name = args[prefix_len:].strip()
            return {
                "action": "execute",
                "agent_name": agent_name
            }
        elif args.startswith("info ") or args.startswith("show "):
            # "/agent info agent_name" 형식
            prefix_len = 5 if args.startswith("info ") else 5
            agent_name = args[prefix_len:].strip()
            return {
                "action": "info",
                "agent_name": agent_name
            }
        elif args.startswith("delete ") or args.startswith("remove "):
            # "/agent delete agent_name" 형식
            prefix_len = 7 if args.startswith("delete ") else 7
            agent_name = args[prefix_len:].strip()
            return {
                "action": "delete",
                "agent_name": agent_name
            }
        elif args in ["list", "ls"]:
            return {"action": "list"}
        else:
            # 기본적으로는 생성 명령으로 간주
            return {
                "action": "create",
                "instruction": args
            }
    
    async def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """명령 직접 실행"""
        try:
            # 명령 분석
            analysis = await self.analyze_command(command)
            
            # 명령 유형별 처리
            if analysis["type"] == "mission":
                return await self._execute_mission_command(analysis, **kwargs)
            elif analysis["type"] == "learning":
                return await self._execute_learning_command(analysis, **kwargs)
            elif analysis["type"] == "hardware":
                return await self._execute_hardware_command(analysis, **kwargs)
            elif analysis["type"] == "simulation":
                return await self._execute_simulation_command(analysis, **kwargs)
            elif analysis["type"] == "agent_management":
                return await self._execute_agent_command(analysis, **kwargs)
            else:
                return {
                    "success": True,
                    "message": f"Command '{command}' analyzed but no direct execution available",
                    "analysis": analysis
                }
                
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_mission_command(self, analysis: Dict, **kwargs) -> Dict[str, Any]:
        """미션 명령 실행"""
        mission = kwargs.get("mission", analysis["parameters"].get("raw_text", ""))
        
        return {
            "success": True,
            "message": f"Mission '{mission}' queued for execution",
            "details": {
                "mission": mission,
                "analysis": analysis,
                "status": "queued"
            }
        }
    
    async def _execute_learning_command(self, analysis: Dict, **kwargs) -> Dict[str, Any]:
        """학습 명령 실행"""
        skill = kwargs.get("skill", "general_skill")
        
        return {
            "success": True, 
            "message": f"Learning session for '{skill}' initiated",
            "details": {
                "skill": skill,
                "analysis": analysis,
                "status": "learning_started"
            }
        }
    
    async def _execute_hardware_command(self, analysis: Dict, **kwargs) -> Dict[str, Any]:
        """하드웨어 명령 실행"""
        component = kwargs.get("component", "all")
        
        # 시뮬레이션된 하드웨어 상태
        status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "operational",
            "components": {
                "sensors": {"status": "online", "count": 5},
                "actuators": {"status": "online", "count": 8},
                "joints": {"status": "calibrated", "count": 6},
                "gripper": {"status": "ready", "force": 0.0}
            }
        }
        
        return {
            "success": True,
            "message": f"Hardware status retrieved for '{component}'",
            "details": status
        }
    
    async def _execute_simulation_command(self, analysis: Dict, **kwargs) -> Dict[str, Any]:
        """시뮬레이션 명령 실행"""
        scenario = kwargs.get("scenario", "basic_environment")
        
        return {
            "success": True,
            "message": f"Simulation '{scenario}' started",
            "details": {
                "scenario": scenario,
                "status": "running",
                "estimated_duration": 60.0
            }
        }
    
    async def _execute_agent_command(self, analysis: Dict, **kwargs) -> Dict[str, Any]:
        """에이전트 명령 실행"""
        agent_args = kwargs.get("agent_args", "")
        
        # agent_args 파싱
        parsed_params = self._parse_agent_command(agent_args)
        
        return {
            "success": True,
            "message": f"Agent command executed: {parsed_params.get('action', 'unknown')}",
            "details": parsed_params
        }