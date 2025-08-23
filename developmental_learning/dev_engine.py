"""
Developmental Learning Engine - 발달적 학습 시스템

아기가 자라듯이 점진적으로 스킬을 습득하고 개선하는 
발달적 학습 엔진입니다.
"""

import asyncio
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import random

@dataclass
class Skill:
    """스킬 정의"""
    name: str
    difficulty_level: int  # 1(쉬움) ~ 10(매우 어려움)
    prerequisites: List[str]
    success_rate: float
    practice_count: int
    last_practiced: datetime
    energy_efficiency: float
    
@dataclass
class Experience:
    """경험 데이터"""
    timestamp: datetime
    skill_used: str
    context: Dict[str, Any]
    action_taken: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    learning_value: float

class SkillAcquisitionEngine:
    """스킬 습득 엔진"""
    
    def __init__(self):
        self.skills_db = self._initialize_basic_skills()
        self.learning_progress = {}
        self.curriculum = CurriculumLearning()
        
    def _initialize_basic_skills(self) -> Dict[str, Skill]:
        """기본 스킬 초기화"""
        now = datetime.now()
        return {
            "basic_movement": Skill(
                name="basic_movement",
                difficulty_level=1,
                prerequisites=[],
                success_rate=0.9,
                practice_count=0,
                last_practiced=now,
                energy_efficiency=0.8
            ),
            "object_recognition": Skill(
                name="object_recognition",
                difficulty_level=2,
                prerequisites=["basic_movement"],
                success_rate=0.7,
                practice_count=0,
                last_practiced=now,
                energy_efficiency=0.9
            ),
            "simple_grasp": Skill(
                name="simple_grasp",
                difficulty_level=3,
                prerequisites=["basic_movement", "object_recognition"],
                success_rate=0.5,
                practice_count=0,
                last_practiced=now,
                energy_efficiency=0.6
            ),
            "precise_manipulation": Skill(
                name="precise_manipulation",
                difficulty_level=7,
                prerequisites=["simple_grasp"],
                success_rate=0.2,
                practice_count=0,
                last_practiced=now,
                energy_efficiency=0.4
            ),
            "collaborative_task": Skill(
                name="collaborative_task",
                difficulty_level=9,
                prerequisites=["precise_manipulation", "object_recognition"],
                success_rate=0.1,
                practice_count=0,
                last_practiced=now,
                energy_efficiency=0.5
            )
        }
    
    async def assess_skill_requirements(self, task_plan) -> List[str]:
        """태스크에 필요한 스킬 평가"""
        required_skills = []
        
        for subtask in task_plan.subtasks:
            action = subtask.get("action", "")
            
            if action == "move_to":
                required_skills.append("basic_movement")
            elif action == "grasp":
                required_skills.append("simple_grasp")
                required_skills.append("object_recognition")
            elif action == "place":
                required_skills.append("precise_manipulation")
            elif action == "explore":
                required_skills.append("basic_movement")
                required_skills.append("object_recognition")
        
        return list(set(required_skills))  # 중복 제거
    
    async def check_skill_readiness(self, skill_names: List[str]) -> Dict[str, bool]:
        """스킬 준비도 확인"""
        readiness = {}
        
        for skill_name in skill_names:
            if skill_name not in self.skills_db:
                readiness[skill_name] = False
                continue
                
            skill = self.skills_db[skill_name]
            
            # 전제조건 확인
            prerequisites_met = all(
                prereq in self.skills_db and self.skills_db[prereq].success_rate > 0.7
                for prereq in skill.prerequisites
            )
            
            # 성공률 기준 (70% 이상이면 준비됨)
            success_ready = skill.success_rate >= 0.7
            
            readiness[skill_name] = prerequisites_met and success_ready
        
        return readiness
    
    async def practice_skill(self, skill_name: str, context: Dict[str, Any]) -> bool:
        """스킬 연습"""
        if skill_name not in self.skills_db:
            print(f"스킬 {skill_name}을 찾을 수 없습니다.")
            return False
        
        skill = self.skills_db[skill_name]
        
        # 연습 시도
        success = random.random() < skill.success_rate
        
        # 스킬 개선 (연습할수록 향상)
        if success:
            # 성공시 성공률 소폭 증가
            skill.success_rate = min(0.95, skill.success_rate + 0.02)
            skill.energy_efficiency = min(1.0, skill.energy_efficiency + 0.01)
        else:
            # 실패시에도 약간의 학습 효과
            skill.success_rate = min(0.95, skill.success_rate + 0.005)
        
        skill.practice_count += 1
        skill.last_practiced = datetime.now()
        
        print(f"스킬 '{skill_name}' 연습 {'성공' if success else '실패'} "
              f"(성공률: {skill.success_rate:.2f})")
        
        return success

class MemoryManagement:
    """메모리 및 경험 관리"""
    
    def __init__(self):
        self.episodic_memory: List[Experience] = []
        self.semantic_memory: Dict[str, Any] = {}
        self.working_memory: Dict[str, Any] = {}
        self.max_episodic_memories = 10000
    
    def store_experience(self, experience: Experience):
        """경험 저장"""
        self.episodic_memory.append(experience)
        
        # 메모리 크기 제한
        if len(self.episodic_memory) > self.max_episodic_memories:
            # 오래된 경험 중 학습 가치가 낮은 것부터 제거
            self.episodic_memory.sort(key=lambda x: x.learning_value)
            self.episodic_memory = self.episodic_memory[100:]  # 100개 제거
    
    def retrieve_similar_experiences(self, context: Dict[str, Any], 
                                   limit: int = 10) -> List[Experience]:
        """유사한 경험 검색"""
        # 단순한 유사도 매칭 (실제로는 더 정교한 임베딩 기반 검색)
        relevant_experiences = []
        
        for exp in self.episodic_memory:
            similarity = self._calculate_context_similarity(context, exp.context)
            if similarity > 0.5:  # 임계값
                relevant_experiences.append((exp, similarity))
        
        # 유사도 순으로 정렬
        relevant_experiences.sort(key=lambda x: x[1], reverse=True)
        
        return [exp for exp, _ in relevant_experiences[:limit]]
    
    def _calculate_context_similarity(self, ctx1: Dict[str, Any], 
                                    ctx2: Dict[str, Any]) -> float:
        """컨텍스트 유사도 계산"""
        # 간단한 키 기반 유사도
        common_keys = set(ctx1.keys()) & set(ctx2.keys())
        total_keys = set(ctx1.keys()) | set(ctx2.keys())
        
        if not total_keys:
            return 0.0
            
        return len(common_keys) / len(total_keys)
    
    def update_semantic_knowledge(self, skill_name: str, insights: Dict[str, Any]):
        """의미적 지식 업데이트"""
        if skill_name not in self.semantic_memory:
            self.semantic_memory[skill_name] = {}
        
        # 새로운 인사이트 통합
        self.semantic_memory[skill_name].update(insights)

class CurriculumLearning:
    """커리큘럼 학습 - 단순한 것부터 복잡한 것으로"""
    
    def __init__(self):
        self.learning_stages = [
            {"stage": 1, "max_difficulty": 2, "focus": "기본 동작"},
            {"stage": 2, "max_difficulty": 4, "focus": "객체 인식 및 조작"},
            {"stage": 3, "max_difficulty": 6, "focus": "정밀 조작"},
            {"stage": 4, "max_difficulty": 8, "focus": "복합 태스크"},
            {"stage": 5, "max_difficulty": 10, "focus": "고급 협업"}
        ]
        self.current_stage = 1
    
    def get_appropriate_skills(self, available_skills: Dict[str, Skill]) -> List[str]:
        """현재 단계에 적합한 스킬 반환"""
        current_max_difficulty = self.learning_stages[self.current_stage - 1]["max_difficulty"]
        
        appropriate = []
        for name, skill in available_skills.items():
            if skill.difficulty_level <= current_max_difficulty:
                # 전제조건도 확인
                if all(prereq in available_skills and 
                      available_skills[prereq].success_rate > 0.7 
                      for prereq in skill.prerequisites):
                    appropriate.append(name)
        
        return appropriate
    
    def should_advance_stage(self, skills: Dict[str, Skill]) -> bool:
        """다음 단계로 진행해야 하는지 판단"""
        if self.current_stage >= len(self.learning_stages):
            return False
        
        current_max_difficulty = self.learning_stages[self.current_stage - 1]["max_difficulty"]
        
        # 현재 단계의 모든 스킬이 70% 이상 성공률을 갖는지 확인
        stage_skills = [
            skill for skill in skills.values() 
            if skill.difficulty_level <= current_max_difficulty
        ]
        
        if not stage_skills:
            return True
        
        avg_success_rate = sum(skill.success_rate for skill in stage_skills) / len(stage_skills)
        return avg_success_rate >= 0.7
    
    def advance_stage(self):
        """다음 단계로 진행"""
        if self.current_stage < len(self.learning_stages):
            self.current_stage += 1
            print(f"학습 단계 {self.current_stage}로 진행: "
                  f"{self.learning_stages[self.current_stage - 1]['focus']}")

class DevelopmentalEngine:
    """발달적 학습 메인 엔진"""
    
    def __init__(self):
        self.skill_engine = SkillAcquisitionEngine()
        self.memory = MemoryManagement()
        self.autonomous_learning_active = True
        
    async def initialize(self):
        """발달적 학습 엔진 초기화"""
        print("Developmental Learning Engine 초기화 중...")
        
        # 기본 스킬들로 연습 시작
        basic_skills = ["basic_movement", "object_recognition"]
        for skill in basic_skills:
            await self.skill_engine.practice_skill(skill, {"context": "initialization"})
        
        print("Developmental Learning Engine 초기화 완료")
    
    async def analyze_required_skills(self, task_plan) -> Dict[str, Any]:
        """태스크에 필요한 스킬 분석 및 준비"""
        print("필요한 스킬 분석 중...")
        
        # 1. 필요한 스킬 식별
        required_skills = await self.skill_engine.assess_skill_requirements(task_plan)
        print(f"필요한 스킬: {required_skills}")
        
        # 2. 스킬 준비도 확인
        readiness = await self.skill_engine.check_skill_readiness(required_skills)
        
        # 3. 부족한 스킬 연습
        for skill_name, is_ready in readiness.items():
            if not is_ready:
                print(f"스킬 '{skill_name}' 집중 연습 중...")
                for _ in range(5):  # 5회 연습
                    await self.skill_engine.practice_skill(
                        skill_name, 
                        {"context": "preparation", "task": task_plan.mission}
                    )
        
        # 4. 최종 스킬 상태 반환
        skill_states = {}
        for skill_name in required_skills:
            if skill_name in self.skill_engine.skills_db:
                skill = self.skill_engine.skills_db[skill_name]
                skill_states[skill_name] = {
                    "success_rate": skill.success_rate,
                    "energy_efficiency": skill.energy_efficiency,
                    "practice_count": skill.practice_count
                }
        
        return {
            "required_skills": required_skills,
            "readiness": readiness,
            "skill_states": skill_states
        }
    
    async def learn_from_experience(self, execution_result):
        """실행 결과로부터 학습"""
        print("실행 결과로부터 학습 중...")
        
        # ExecutionResult 객체에서 데이터 추출
        if hasattr(execution_result, 'success'):
            # ExecutionResult 객체인 경우
            success = execution_result.success
            learning_value = execution_result.learning_value
            actions = execution_result.actions_performed
            errors = execution_result.errors
        else:
            # 딕셔너리인 경우
            success = execution_result.get("success", False)
            learning_value = execution_result.get("learning_value", 0.5)
            actions = execution_result.get("actions", {})
            errors = execution_result.get("errors", [])
        
        # 경험 데이터 생성
        experience = Experience(
            timestamp=datetime.now(),
            skill_used="task_execution",  # 기본값
            context={"actions_count": len(actions) if isinstance(actions, list) else 0},
            action_taken={"actions": actions},
            result={"success": success, "errors": errors},
            success=success,
            learning_value=learning_value
        )
        
        # 메모리에 저장
        self.memory.store_experience(experience)
        
        # 관련 스킬 업데이트
        if experience.skill_used in self.skill_engine.skills_db:
            skill = self.skill_engine.skills_db[experience.skill_used]
            
            if experience.success:
                skill.success_rate = min(0.95, skill.success_rate + 0.01)
                skill.energy_efficiency = min(1.0, skill.energy_efficiency + 0.005)
            else:
                # 실패했어도 경험은 쌓임
                skill.success_rate = max(0.05, skill.success_rate - 0.005)
        
        print(f"경험 학습 완료: {experience.skill_used} "
              f"({'성공' if experience.success else '실패'})")
    
    async def autonomous_exploration(self):
        """자율적 탐색 및 학습"""
        if not self.autonomous_learning_active:
            return
            
        print("자율적 탐색 학습 시작...")
        
        # 현재 단계에 적합한 스킬들 선택
        appropriate_skills = self.skill_engine.curriculum.get_appropriate_skills(
            self.skill_engine.skills_db
        )
        
        if not appropriate_skills:
            return
        
        # 가장 개선이 필요한 스킬 선택
        target_skill = min(appropriate_skills, 
                          key=lambda s: self.skill_engine.skills_db[s].success_rate)
        
        # 연습
        context = {"context": "autonomous_exploration", "timestamp": datetime.now()}
        success = await self.skill_engine.practice_skill(target_skill, context)
        
        # 단계 진행 여부 확인
        if self.skill_engine.curriculum.should_advance_stage(self.skill_engine.skills_db):
            self.skill_engine.curriculum.advance_stage()
        
        print(f"자율 탐색 완료: {target_skill} ({'성공' if success else '실패'})")

# 테스트 코드
if __name__ == "__main__":
    async def test():
        dev_engine = DevelopmentalEngine()
        await dev_engine.initialize()
        
        # 몇 번의 자율 학습 시뮬레이션
        for i in range(10):
            await dev_engine.autonomous_exploration()
            await asyncio.sleep(0.1)  # 짧은 대기
        
        # 스킬 상태 출력
        print("\n=== 최종 스킬 상태 ===")
        for name, skill in dev_engine.skill_engine.skills_db.items():
            print(f"{name}: 성공률 {skill.success_rate:.2f}, "
                  f"연습횟수 {skill.practice_count}, "
                  f"효율성 {skill.energy_efficiency:.2f}")
    
    asyncio.run(test())
