"""
Hugging Face Gradio Demo for Physical AI System

이 데모는 Physical AI System의 핵심 기능들을 
웹 인터페이스에서 체험할 수 있게 해줍니다.
"""

import gradio as gr
import asyncio
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
from PIL import Image

# Physical AI System imports
from main import PhysicalAI
from foundation_model.slm_foundation import SLMFoundation
from developmental_learning.dev_engine import DevelopmentalEngine


class PhysicalAIDemo:
    """Physical AI System Gradio Demo"""
    
    def __init__(self):
        self.ai_system = None
        self.demo_initialized = False
        
    async def initialize_demo(self):
        """데모용 AI 시스템 초기화"""
        if not self.demo_initialized:
            # 데모 모드로 시스템 초기화
            self.ai_system = PhysicalAI("configs/default.yaml")
            await self.ai_system.initialize()
            self.demo_initialized = True
            
    async def mission_planner_demo(self, mission_text):
        """미션 계획 수립 데모 - 실제 LLM 사용"""
        if not mission_text.strip():
            return "미션을 입력해주세요!", "", ""
        
        try:
            # 데모용 AI 시스템 초기화 (한 번만)
            await self.initialize_demo()
            
            # 실제 Foundation Model을 통한 미션 해석
            task_plan = await self.ai_system.slm_foundation.interpret_mission(mission_text)
            
            # 결과 포맷팅
            subtasks_formatted = []
            for i, subtask in enumerate(task_plan.subtasks):
                action_icons = {
                    "move_to": "🎯",
                    "grasp": "🤏", 
                    "place": "📦",
                    "explore": "🔍",
                    "explore_environment": "🔍"
                }
                icon = action_icons.get(subtask.get("action", ""), "🤖")
                subtasks_formatted.append(f"{icon} {subtask.get('action', 'unknown').title()} → {subtask.get('target', 'unknown')}")
            
            planning_result = f"""
## 🧠 미션 분석 결과 (실제 AI 생성)

**입력 미션**: {mission_text}

### 📋 서브태스크 분해:
""" + "\n".join([f"{i+1}. {task}" for i, task in enumerate(subtasks_formatted)])

            # 안전 제약사항 (LLM이 있다면 LLM으로 분석)
            constraints = f"""
### ⚠️ 안전 제약사항:
- 최대 속도: {task_plan.constraints.get('max_velocity', 2.0)} m/s
- 최대 가속도: {task_plan.constraints.get('max_acceleration', 5.0)} m/s²
- 안전 거리: {task_plan.constraints.get('safety_distance', 0.1)} m
- 최대 힘: {task_plan.constraints.get('max_force', 50.0)} N
- 제한 시간: {task_plan.constraints.get('timeout', 300)} 초
"""

            # 성공 기준
            success_criteria = f"""
### ✅ 성공 기준:
""" + "\n".join([f"- {criterion}" for criterion in task_plan.success_criteria])
            
            success_criteria += f"""
- 예상 실행 시간: {task_plan.expected_duration:.1f} 초
- 서브태스크 수: {len(task_plan.subtasks)}개
"""
            
            return planning_result, constraints, success_criteria
            
        except Exception as e:
            print(f"실제 AI 호출 실패: {e}")
            # 폴백 데모 응답
            return self._fallback_mission_demo(mission_text)
    
    def _fallback_mission_demo(self, mission_text):
        """LLM 실패시 폴백 데모"""
        subtasks = [
            "🎯 객체 위치 탐색 및 식별",
            "🤖 목표 위치로 이동", 
            "🤏 안전한 객체 그립",
            "📦 목표 지점으로 운반",
            "✅ 정확한 위치에 배치"
        ]
        
        planning_result = f"""
## 🧠 미션 분석 결과 (폴백 모드)

**입력 미션**: {mission_text}

### 📋 서브태스크 분해:
""" + "\n".join([f"{i+1}. {task}" for i, task in enumerate(subtasks)])

        constraints = """
### ⚠️ 안전 제약사항:
- 최대 속도: 2.0 m/s
- 최대 가속도: 5.0 m/s²
- 안전 거리: 0.1 m
- 그립 강도: 최대 50N
"""

        success_criteria = """
### ✅ 성공 기준:
- 태스크 완료율 > 90%
- 충돌 발생: 0회
- 에너지 효율성 > 80%
- 실행 시간 < 60초
"""
        
        return planning_result, constraints, success_criteria

    def skill_learning_demo(self, skill_name, practice_rounds):
        """스킬 학습 데모"""
        if not skill_name:
            return "스킬 이름을 입력해주세요!", None
            
        try:
            # 스킬 학습 시뮬레이션
            skill_progress = []
            success_rates = []
            
            # 초기 성공률 (스킬에 따라 다름)
            initial_success = {
                "basic_movement": 0.8,
                "object_recognition": 0.6, 
                "simple_grasp": 0.4,
                "precise_manipulation": 0.2,
                "collaborative_task": 0.1
            }.get(skill_name.lower().replace(" ", "_"), 0.3)
            
            current_success = initial_success
            
            for round_num in range(practice_rounds):
                # 학습 곡선 시뮬레이션
                improvement = np.random.uniform(0.01, 0.05)
                noise = np.random.uniform(-0.02, 0.02)
                current_success = min(0.95, current_success + improvement + noise)
                
                skill_progress.append({
                    "round": round_num + 1,
                    "success_rate": current_success,
                    "improvement": improvement
                })
                success_rates.append(current_success)
            
            # 학습 곡선 그래프 생성
            plt.figure(figsize=(10, 6))
            rounds = list(range(1, practice_rounds + 1))
            plt.plot(rounds, success_rates, 'b-o', linewidth=2, markersize=6)
            plt.fill_between(rounds, success_rates, alpha=0.3)
            plt.xlabel('연습 라운드')
            plt.ylabel('성공률')
            plt.title(f'🌱 {skill_name} 스킬 학습 진행률')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # 성과 지표 추가
            plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='목표 성공률 (70%)')
            plt.legend()
            
            # 그래프를 이미지로 변환
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            learning_summary = f"""
## 🌱 스킬 학습 결과

**스킬**: {skill_name}
**연습 횟수**: {practice_rounds}회

### 📊 학습 성과:
- 초기 성공률: {initial_success:.1%}
- 최종 성공률: {current_success:.1%}
- 개선도: {current_success - initial_success:.1%}
- 학습 효율: {'🟢 우수' if current_success > 0.7 else '🟡 보통' if current_success > 0.5 else '🔴 개선 필요'}

### 💡 학습 인사이트:
- 점진적 개선을 통한 안정적 학습 확인
- {'목표 성공률(70%) 달성!' if current_success > 0.7 else '추가 연습 권장'}
- 실전 적용 준비도: {'높음' if current_success > 0.8 else '보통' if current_success > 0.6 else '낮음'}
"""
            
            return learning_summary, buf.getvalue()
            
        except Exception as e:
            return f"❌ 학습 오류: {str(e)}", None

    def safety_monitor_demo(self):
        """안전 모니터링 데모"""
        try:
            # 안전 상태 시뮬레이션
            safety_metrics = {
                "collision_risk": np.random.uniform(0, 0.1),
                "human_proximity": np.random.uniform(0, 0.3), 
                "joint_limits": np.random.uniform(0, 0.8),
                "energy_level": np.random.uniform(0.6, 1.0),
                "system_temperature": np.random.uniform(25, 45)
            }
            
            # 안전 상태 판단
            safety_status = "🟢 안전" 
            warnings = []
            
            if safety_metrics["collision_risk"] > 0.05:
                safety_status = "🔴 위험"
                warnings.append("충돌 위험 감지")
                
            if safety_metrics["human_proximity"] > 0.2:
                safety_status = "🟡 주의" if safety_status == "🟢 안전" else safety_status
                warnings.append("인간 근접 감지")
                
            if safety_metrics["energy_level"] < 0.2:
                warnings.append("배터리 부족")
                
            if safety_metrics["system_temperature"] > 40:
                warnings.append("시스템 과열")
            
            safety_report = f"""
## 🛡️ 실시간 안전 모니터링

### 현재 상태: {safety_status}

### 📊 안전 지표:
- 충돌 위험도: {safety_metrics["collision_risk"]:.1%} {'🟢' if safety_metrics["collision_risk"] < 0.05 else '🔴'}
- 인간 근접도: {safety_metrics["human_proximity"]:.1%} {'🟢' if safety_metrics["human_proximity"] < 0.2 else '🟡'}
- 관절 부하율: {safety_metrics["joint_limits"]:.1%} {'🟢' if safety_metrics["joint_limits"] < 0.8 else '🟡'}
- 에너지 잔량: {safety_metrics["energy_level"]:.1%} {'🟢' if safety_metrics["energy_level"] > 0.2 else '🔴'}
- 시스템 온도: {safety_metrics["system_temperature"]:.1f}°C {'🟢' if safety_metrics["system_temperature"] < 40 else '🟡'}

### ⚠️ 안전 경고:
""" + ("\n".join([f"- {warning}" for warning in warnings]) if warnings else "- 현재 안전 경고 없음")

            safety_actions = """
### 🚨 비상 대응:
- 비상 정지 버튼: 준비됨
- 안전 모드 전환: 자동 활성화
- 충돌 회피 알고리즘: 작동 중
- 인간 안전 우선순위: 최고
"""
            
            return safety_report, safety_actions
            
        except Exception as e:
            return f"❌ 안전 모니터링 오류: {str(e)}", ""

    def hardware_status_demo(self):
        """하드웨어 상태 데모"""
        try:
            # 하드웨어 상태 시뮬레이션
            sensors = {
                "main_camera": {"status": "정상", "fps": 30, "resolution": "640x480"},
                "tactile_sensors": {"status": "정상", "sensitivity": 0.1, "contacts": 0},
                "imu_sensor": {"status": "정상", "calibrated": True, "drift": 0.01}
            }
            
            actuators = {
                "joint_1": {"position": 0.5, "velocity": 0.1, "temperature": 35, "status": "정상"},
                "joint_2": {"position": -0.3, "velocity": 0.0, "temperature": 32, "status": "정상"},
                "gripper": {"is_open": True, "force": 0.0, "object_held": None, "status": "정상"}
            }
            
            hardware_report = f"""
## 🔌 하드웨어 상태 리포트

### 📷 센서 상태:
""" + "\n".join([f"- **{sensor_id}**: {info['status']} " + 
                f"({'FPS: ' + str(info['fps']) if 'fps' in info else ''}" +
                f"{'해상도: ' + info['resolution'] if 'resolution' in info else ''}" +
                f"{'민감도: ' + str(info['sensitivity']) if 'sensitivity' in info else ''}"
                f")"
                for sensor_id, info in sensors.items()])

            hardware_report += f"""

### 🤖 액추에이터 상태:
""" + "\n".join([f"- **{actuator_id}**: {info['status']} " +
                f"(위치: {info.get('position', 'N/A')}, " +
                f"온도: {info.get('temperature', 'N/A')}°C)"
                for actuator_id, info in actuators.items()])

            system_health = """

### 💚 시스템 건강도:
- 전체 시스템: 🟢 정상
- 통신 상태: 🟢 안정
- 캘리브레이션: 🟢 완료
- 성능 지표: 🟢 최적
"""
            
            return hardware_report, system_health
            
        except Exception as e:
            return f"❌ 하드웨어 상태 오류: {str(e)}", ""


# 글로벌 데모 인스턴스
demo_instance = PhysicalAIDemo()


def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # CSS 스타일링
    custom_css = """
    .gradio-container {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .title {
        text-align: center;
        color: #2563eb;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Physical AI System Demo") as demo:
        
        # 헤더
        gr.HTML("""
        <div class="title">🤖 Physical AI System Demo</div>
        <div class="subtitle">Developmental Robotics & Embodied AI Framework</div>
        """)
        
        # 탭 구성
        with gr.Tabs():
            
            # 미션 계획 탭
            with gr.TabItem("🎯 Mission Planning"):
                gr.Markdown("### 자연어 미션을 물리적 동작 계획으로 변환합니다")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        mission_input = gr.Textbox(
                            label="미션 입력",
                            placeholder="예: Pick up the red cup and place it on the table",
                            lines=3
                        )
                        plan_button = gr.Button("🧠 미션 분석", variant="primary")
                        
                    with gr.Column(scale=2):
                        planning_output = gr.Markdown(label="계획 결과")
                        
                with gr.Row():
                    constraints_output = gr.Markdown(label="제약사항")
                    success_criteria_output = gr.Markdown(label="성공기준")
                
                def mission_planner_wrapper(mission_text):
                    """비동기 함수를 동기로 래핑"""
                    import asyncio
                    try:
                        # 이벤트 루프가 있는지 확인
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 이미 실행 중인 루프에서는 새 태스크 생성
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(
                                    lambda: asyncio.run(demo_instance.mission_planner_demo(mission_text))
                                )
                                return future.result()
                        else:
                            return asyncio.run(demo_instance.mission_planner_demo(mission_text))
                    except Exception as e:
                        print(f"Async wrapper error: {e}")
                        return demo_instance._fallback_mission_demo(mission_text)
                
                plan_button.click(
                    fn=mission_planner_wrapper,
                    inputs=[mission_input],
                    outputs=[planning_output, constraints_output, success_criteria_output]
                )
            
            # 스킬 학습 탭  
            with gr.TabItem("🌱 Skill Learning"):
                gr.Markdown("### AI가 스스로 스킬을 학습하는 과정을 관찰합니다")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        skill_input = gr.Textbox(
                            label="학습할 스킬",
                            placeholder="예: object_recognition, simple_grasp",
                            value="simple_grasp"
                        )
                        practice_slider = gr.Slider(
                            label="연습 횟수",
                            minimum=5,
                            maximum=50,
                            value=20,
                            step=5
                        )
                        learn_button = gr.Button("🌱 스킬 학습", variant="primary")
                        
                    with gr.Column(scale=2):
                        learning_output = gr.Markdown(label="학습 결과")
                        learning_plot = gr.Image(label="학습 곡선")
                
                learn_button.click(
                    fn=demo_instance.skill_learning_demo,
                    inputs=[skill_input, practice_slider],
                    outputs=[learning_output, learning_plot]
                )
            
            # 안전 모니터링 탭
            with gr.TabItem("🛡️ Safety Monitor"):
                gr.Markdown("### 실시간 안전 상태를 모니터링합니다")
                
                with gr.Row():
                    with gr.Column():
                        safety_button = gr.Button("🔍 안전 상태 확인", variant="primary")
                        gr.Markdown("*실제 로봇에서는 10Hz 주기로 자동 모니터링*")
                        
                with gr.Row():
                    safety_status = gr.Markdown(label="안전 상태")
                    safety_actions = gr.Markdown(label="안전 조치")
                
                safety_button.click(
                    fn=demo_instance.safety_monitor_demo,
                    outputs=[safety_status, safety_actions]
                )
            
            # 하드웨어 상태 탭
            with gr.TabItem("🔌 Hardware Status"):
                gr.Markdown("### 센서와 액추에이터 상태를 확인합니다")
                
                with gr.Row():
                    with gr.Column():
                        hardware_button = gr.Button("🔧 하드웨어 점검", variant="primary") 
                        gr.Markdown("*센서 융합 및 액추에이터 제어 상태*")
                        
                with gr.Row():
                    hardware_status = gr.Markdown(label="하드웨어 상태")
                    system_health = gr.Markdown(label="시스템 건강도")
                
                hardware_button.click(
                    fn=demo_instance.hardware_status_demo,
                    outputs=[hardware_status, system_health]
                )
        
        # 푸터
        gr.HTML("""
        <div style="text-align: center; margin-top: 2em; padding: 1em; background-color: #f8fafc; border-radius: 8px;">
            <p><strong>🚀 Physical AI System</strong> - The future of AI is not just digital, it's physical.</p>
            <p>
                <a href="https://github.com/your-username/physical-ai-system" target="_blank">📖 Documentation</a> |
                <a href="https://github.com/your-username/physical-ai-system" target="_blank">💻 GitHub</a> |
                <a href="#" target="_blank">🤝 Community</a>
            </p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    # 데모 실행
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )