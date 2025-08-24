"""
대화형 동작학습 인터페이스 - 통합 시스템

사용자와 PHI-3.5가 자연어로 대화하면서 실시간으로 동작을 학습하고
피드백을 주고받을 수 있는 고급 인터페이스입니다.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import speech_recognition as sr
import pyttsx3
from queue import Queue, Empty
import cv2
import numpy as np

# 프로젝트 모듈
from main import PhysicalAI
from foundation_model.phi35_integration import PHI35ModelManager

logger = logging.getLogger(__name__)

@dataclass
class LearningSession:
    """학습 세션 정보"""
    session_id: str
    user_name: str
    started_at: datetime
    total_interactions: int
    successful_demonstrations: int
    failed_attempts: int
    learned_skills: List[str]
    current_skill: str
    session_notes: str

@dataclass
class InteractionRecord:
    """상호작용 기록"""
    timestamp: datetime
    user_input: str
    input_type: str  # "text", "voice", "demonstration"
    ai_response: str
    action_performed: Optional[Dict[str, Any]]
    feedback_given: str
    learning_value: float
    success: bool

class VoiceInterface:
    """음성 인터페이스"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.listening = False
        
        # TTS 설정
        voices = self.tts_engine.getProperty('voices')
        if voices and len(voices) > 1:
            self.tts_engine.setProperty('voice', voices[1].id)  # 여성 음성
        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.8)
    
    def calibrate_microphone(self):
        """마이크 보정"""
        with self.microphone as source:
            logger.info("마이크 보정 중...")
            self.recognizer.adjust_for_ambient_noise(source)
            logger.info("마이크 보정 완료")
    
    async def listen_for_speech(self) -> Optional[str]:
        """음성 인식"""
        try:
            with self.microphone as source:
                logger.info("음성 입력 대기 중...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # 음성을 텍스트로 변환
            text = self.recognizer.recognize_google(audio, language='ko-KR')
            logger.info(f"인식된 텍스트: {text}")
            return text
            
        except sr.WaitTimeoutError:
            logger.warning("음성 입력 시간 초과")
            return None
        except sr.UnknownValueError:
            logger.warning("음성을 인식할 수 없습니다")
            return None
        except Exception as e:
            logger.error(f"음성 인식 오류: {e}")
            return None
    
    def speak(self, text: str):
        """TTS 음성 출력"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS 오류: {e}")

class VisualFeedbackSystem:
    """시각적 피드백 시스템"""
    
    def __init__(self, canvas_widget):
        self.canvas = canvas_widget
        self.canvas_width = 800
        self.canvas_height = 600
        self.robot_position = (400, 300)  # 중앙
        self.target_position = None
        self.trajectory_points = []
        self.skill_progress = {}
        
    def update_robot_position(self, x: int, y: int):
        """로봇 위치 업데이트"""
        self.robot_position = (x, y)
        self.redraw_scene()
    
    def set_target_position(self, x: int, y: int):
        """목표 위치 설정"""
        self.target_position = (x, y)
        self.redraw_scene()
    
    def add_trajectory_point(self, x: int, y: int):
        """궤적 점 추가"""
        self.trajectory_points.append((x, y))
        if len(self.trajectory_points) > 50:  # 최대 50개 점 유지
            self.trajectory_points.pop(0)
        self.redraw_scene()
    
    def update_skill_progress(self, skill_name: str, progress: float):
        """스킬 진행도 업데이트"""
        self.skill_progress[skill_name] = progress
    
    def redraw_scene(self):
        """화면 다시 그리기"""
        self.canvas.delete("all")
        
        # 배경 그리드
        for i in range(0, self.canvas_width, 50):
            self.canvas.create_line(i, 0, i, self.canvas_height, fill="lightgray", width=1)
        for i in range(0, self.canvas_height, 50):
            self.canvas.create_line(0, i, self.canvas_width, i, fill="lightgray", width=1)
        
        # 궤적 그리기
        if len(self.trajectory_points) > 1:
            for i in range(1, len(self.trajectory_points)):
                x1, y1 = self.trajectory_points[i-1]
                x2, y2 = self.trajectory_points[i]
                alpha = i / len(self.trajectory_points)
                color_intensity = int(255 * alpha)
                color = f"#{color_intensity:02x}00{255-color_intensity:02x}"
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
        
        # 목표 위치 그리기
        if self.target_position:
            x, y = self.target_position
            self.canvas.create_oval(x-15, y-15, x+15, y+15, fill="green", outline="darkgreen", width=2)
            self.canvas.create_text(x, y-25, text="목표", font=("Arial", 10))
        
        # 로봇 그리기
        x, y = self.robot_position
        self.canvas.create_oval(x-20, y-20, x+20, y+20, fill="blue", outline="darkblue", width=3)
        self.canvas.create_text(x, y, text="🤖", font=("Arial", 16))

class InteractiveLearningInterface:
    """대화형 동작학습 인터페이스"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 대화형 동작학습 시스템")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f5f5f5')
        
        # 시스템 컴포넌트
        self.physical_ai = None
        self.voice_interface = VoiceInterface()
        self.visual_feedback = None
        
        # 학습 세션
        self.current_session = None
        self.interaction_history = []
        self.feedback_queue = Queue()
        
        # GUI 상태
        self.is_listening = False
        self.is_demonstrating = False
        self.auto_speech = True
        
        self.setup_gui()
        
        # 비동기 태스크 관리
        self.background_tasks = []
        
    def setup_gui(self):
        """GUI 설정"""
        # 메인 컨테이너
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 제목
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="🎯 대화형 동작학습 시스템", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        # 상태 표시
        self.status_label = ttk.Label(title_frame, text="● 대기 중", 
                                     font=('Arial', 12), foreground='gray')
        self.status_label.pack(side=tk.RIGHT)
        
        # 메인 레이아웃 (3분할)
        paned_main = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned_main.pack(fill=tk.BOTH, expand=True)
        
        # 좌측: 대화 영역
        self.setup_conversation_panel(paned_main)
        
        # 중앙: 시각적 피드백
        self.setup_visual_panel(paned_main)
        
        # 우측: 제어 및 상태
        self.setup_control_panel(paned_main)
    
    def setup_conversation_panel(self, parent):
        """대화 패널 설정"""
        conv_frame = ttk.Frame(parent)
        parent.add(conv_frame, weight=2)
        
        # 대화 영역 제목
        conv_title = ttk.Label(conv_frame, text="💬 AI와 대화", font=('Arial', 14, 'bold'))
        conv_title.pack(pady=(0, 10))
        
        # 대화 기록
        history_frame = ttk.LabelFrame(conv_frame, text="대화 기록")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.conversation_text = scrolledtext.ScrolledText(
            history_frame,
            height=25,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='white',
            state=tk.DISABLED
        )
        self.conversation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 입력 영역
        input_frame = ttk.Frame(conv_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.text_input = ttk.Entry(input_frame, font=('Arial', 11))
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.text_input.bind('<Return>', self.send_text_message)
        
        # 버튼들
        ttk.Button(input_frame, text="전송", command=self.send_text_message).pack(side=tk.RIGHT, padx=(2, 0))
        
        # 음성 제어
        voice_frame = ttk.Frame(conv_frame)
        voice_frame.pack(fill=tk.X)
        
        self.voice_button = ttk.Button(voice_frame, text="🎤 음성 입력", command=self.toggle_voice_input)
        self.voice_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.speech_var = tk.BooleanVar(value=True)
        speech_check = ttk.Checkbutton(voice_frame, text="음성 응답", variable=self.speech_var)
        speech_check.pack(side=tk.LEFT)
    
    def setup_visual_panel(self, parent):
        """시각적 피드백 패널 설정"""
        visual_frame = ttk.Frame(parent)
        parent.add(visual_frame, weight=3)
        
        # 시각화 제목
        visual_title = ttk.Label(visual_frame, text="👁️ 실시간 피드백", font=('Arial', 14, 'bold'))
        visual_title.pack(pady=(0, 10))
        
        # 캔버스 (시뮬레이션 화면)
        canvas_frame = ttk.LabelFrame(visual_frame, text="로봇 시뮬레이션")
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.simulation_canvas = tk.Canvas(canvas_frame, bg='white', width=800, height=600)
        self.simulation_canvas.pack(padx=5, pady=5)
        
        self.visual_feedback = VisualFeedbackSystem(self.simulation_canvas)
        
        # 마우스 이벤트 바인딩 (데모용 목표 위치 설정)
        self.simulation_canvas.bind('<Button-1>', self.on_canvas_click)
    
    def setup_control_panel(self, parent):
        """제어 패널 설정"""
        control_frame = ttk.Frame(parent)
        parent.add(control_frame, weight=1)
        
        # 제어 패널 제목
        control_title = ttk.Label(control_frame, text="⚙️ 제어 및 상태", font=('Arial', 14, 'bold'))
        control_title.pack(pady=(0, 20))
        
        # 시스템 상태
        status_frame = ttk.LabelFrame(control_frame, text="시스템 상태")
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.system_status_text = scrolledtext.ScrolledText(status_frame, height=6, font=('Consolas', 9))
        self.system_status_text.pack(fill=tk.X, padx=5, pady=5)
        
        # 학습 제어
        learning_frame = ttk.LabelFrame(control_frame, text="학습 제어")
        learning_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(learning_frame, text="새 학습 세션", command=self.start_new_session).pack(fill=tk.X, pady=2)
        ttk.Button(learning_frame, text="시연 시작", command=self.start_demonstration).pack(fill=tk.X, pady=2)
        ttk.Button(learning_frame, text="피드백 제공", command=self.provide_feedback).pack(fill=tk.X, pady=2)
        
        # 스킬 진행도
        skills_frame = ttk.LabelFrame(control_frame, text="스킬 진행도")
        skills_frame.pack(fill=tk.BOTH, expand=True)
        
        # 프로그레스 바들
        self.skill_progress_bars = {}
        skills = ['basic_movement', 'object_recognition', 'simple_grasp', 'precise_manipulation']
        
        for i, skill in enumerate(skills):
            skill_label = ttk.Label(skills_frame, text=skill.replace('_', ' ').title())
            skill_label.grid(row=i*2, column=0, sticky='w', padx=5, pady=(5, 2))
            
            progress_bar = ttk.Progressbar(skills_frame, mode='determinate')
            progress_bar.grid(row=i*2+1, column=0, sticky='ew', padx=5, pady=(0, 5))
            
            self.skill_progress_bars[skill] = progress_bar
        
        skills_frame.columnconfigure(0, weight=1)
    
    def add_conversation_message(self, sender: str, message: str, message_type: str = "normal"):
        """대화에 메시지 추가"""
        self.conversation_text.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 색상 설정
        if sender == "사용자":
            color = "blue"
            prefix = "👤"
        elif sender == "AI":
            color = "green"
            prefix = "🤖"
        else:
            color = "gray"
            prefix = "ℹ️"
        
        # 메시지 추가
        self.conversation_text.insert(tk.END, f"[{timestamp}] {prefix} {sender}: {message}\n\n")
        
        # 자동 스크롤
        self.conversation_text.see(tk.END)
        self.conversation_text.config(state=tk.DISABLED)
    
    def update_system_status(self, status: str):
        """시스템 상태 업데이트"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.system_status_text.insert(tk.END, f"[{timestamp}] {status}\n")
        self.system_status_text.see(tk.END)
        
        # 상태 라벨 업데이트
        if "초기화" in status:
            self.status_label.config(text="● 초기화 중", foreground='orange')
        elif "준비" in status:
            self.status_label.config(text="● 준비됨", foreground='green')
        elif "학습" in status:
            self.status_label.config(text="● 학습 중", foreground='blue')
        elif "오류" in status:
            self.status_label.config(text="● 오류", foreground='red')
    
    def send_text_message(self, event=None):
        """텍스트 메시지 전송"""
        message = self.text_input.get().strip()
        if not message:
            return
        
        self.text_input.delete(0, tk.END)
        self.add_conversation_message("사용자", message)
        
        # AI 응답 처리 (비동기)
        threading.Thread(target=self.process_user_input, args=(message, "text"), daemon=True).start()
    
    def toggle_voice_input(self):
        """음성 입력 토글"""
        if self.is_listening:
            self.is_listening = False
            self.voice_button.config(text="🎤 음성 입력")
            self.update_system_status("음성 입력 중지")
        else:
            self.is_listening = True
            self.voice_button.config(text="🛑 중지")
            self.update_system_status("음성 입력 시작")
            threading.Thread(target=self.voice_input_loop, daemon=True).start()
    
    def voice_input_loop(self):
        """음성 입력 루프"""
        while self.is_listening:
            try:
                # 음성 인식 시도
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                text = loop.run_until_complete(self.voice_interface.listen_for_speech())
                loop.close()
                
                if text:
                    # GUI 스레드에서 메시지 추가
                    self.root.after(0, lambda: self.add_conversation_message("사용자", f"[음성] {text}"))
                    
                    # AI 응답 처리
                    self.process_user_input(text, "voice")
                
                time.sleep(0.1)  # CPU 사용량 조절
                
            except Exception as e:
                self.root.after(0, lambda: self.update_system_status(f"음성 입력 오류: {e}"))
                break
    
    def process_user_input(self, user_input: str, input_type: str):
        """사용자 입력 처리"""
        try:
            # 새 이벤트 루프에서 처리
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # AI 응답 생성
            response = loop.run_until_complete(self._generate_ai_response(user_input, input_type))
            
            # GUI 업데이트
            self.root.after(0, lambda: self.add_conversation_message("AI", response))
            
            # 음성 응답 (옵션)
            if self.speech_var.get() and input_type == "voice":
                self.voice_interface.speak(response)
            
            # 동작 실행 시뮬레이션
            self.root.after(0, lambda: self.simulate_robot_action(user_input))
            
            loop.close()
            
        except Exception as e:
            error_msg = f"입력 처리 오류: {e}"
            self.root.after(0, lambda: self.add_conversation_message("시스템", error_msg))
            logger.error(error_msg)
    
    async def _generate_ai_response(self, user_input: str, input_type: str) -> str:
        """AI 응답 생성"""
        try:
            if not self.physical_ai:
                return "시스템이 아직 초기화되지 않았습니다. 초기화를 진행해주세요."
            
            # 컨텍스트 생성
            context = f"""당신은 대화형 로봇 학습 시스템의 AI 튜터입니다.
            
현재 상황:
- 입력 방식: {input_type}
- 학습 세션: {'활성' if self.current_session else '비활성'}
- 현재 스킬: {self.current_session.current_skill if self.current_session else '없음'}

사용자의 입력에 대해:
1. 자연스럽고 친근한 응답을 해주세요
2. 학습 관련 질문에는 구체적인 안내를 제공하세요
3. 로봇 동작이 필요한 경우 어떤 동작을 수행할지 설명하세요
4. 피드백이나 교정이 필요한 경우 건설적인 조언을 해주세요

사용자 입력: {user_input}"""
            
            # PHI-3.5 응답 생성
            if hasattr(self.physical_ai, 'slm_foundation') and self.physical_ai.slm_foundation.phi35_ai:
                response = await self.physical_ai.slm_foundation.phi35_ai.model_manager.generate_response(
                    context, max_new_tokens=256, temperature=0.7
                )
                return response
            else:
                return "죄송합니다. AI 모델이 준비되지 않았습니다."
                
        except Exception as e:
            logger.error(f"AI 응답 생성 실패: {e}")
            return f"응답 생성 중 오류가 발생했습니다: {e}"
    
    def simulate_robot_action(self, user_input: str):
        """로봇 동작 시뮬레이션"""
        # 간단한 키워드 기반 동작 시뮬레이션
        if any(word in user_input.lower() for word in ["이동", "움직", "가다", "move"]):
            # 랜덤 위치로 이동 시뮬레이션
            import random
            new_x = random.randint(50, 750)
            new_y = random.randint(50, 550)
            
            # 애니메이션 효과
            self.animate_robot_movement(new_x, new_y)
            
        elif any(word in user_input.lower() for word in ["잡다", "집다", "grasp", "pick"]):
            # 잡기 동작 시뮬레이션
            self.simulate_grasp_action()
    
    def animate_robot_movement(self, target_x: int, target_y: int):
        """로봇 이동 애니메이션"""
        current_x, current_y = self.visual_feedback.robot_position
        
        steps = 20
        dx = (target_x - current_x) / steps
        dy = (target_y - current_y) / steps
        
        def move_step(step):
            if step <= steps:
                new_x = current_x + dx * step
                new_y = current_y + dy * step
                
                self.visual_feedback.update_robot_position(int(new_x), int(new_y))
                self.visual_feedback.add_trajectory_point(int(new_x), int(new_y))
                
                # 다음 스텝 예약
                self.root.after(50, lambda: move_step(step + 1))
            else:
                self.update_system_status(f"로봇이 ({target_x}, {target_y})로 이동 완료")
        
        move_step(1)
    
    def simulate_grasp_action(self):
        """잡기 동작 시뮬레이션"""
        # 간단한 시각적 효과
        x, y = self.visual_feedback.robot_position
        
        # 그리퍼 원 그리기
        def draw_gripper(size):
            self.visual_feedback.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                outline="red", width=3, tags="gripper"
            )
        
        def animate_grasp(step):
            if step <= 10:
                self.visual_feedback.canvas.delete("gripper")
                draw_gripper(30 - step * 2)
                self.root.after(100, lambda: animate_grasp(step + 1))
            else:
                self.visual_feedback.canvas.delete("gripper")
                self.update_system_status("잡기 동작 완료")
        
        animate_grasp(1)
    
    def on_canvas_click(self, event):
        """캔버스 클릭 이벤트"""
        x, y = event.x, event.y
        self.visual_feedback.set_target_position(x, y)
        self.add_conversation_message("시스템", f"목표 위치 설정: ({x}, {y})")
    
    def start_new_session(self):
        """새 학습 세션 시작"""
        session_id = str(uuid.uuid4())[:8]
        user_name = "User"  # 추후 사용자 입력으로 변경 가능
        
        self.current_session = LearningSession(
            session_id=session_id,
            user_name=user_name,
            started_at=datetime.now(),
            total_interactions=0,
            successful_demonstrations=0,
            failed_attempts=0,
            learned_skills=[],
            current_skill="basic_movement",
            session_notes=""
        )
        
        self.update_system_status(f"새 학습 세션 시작: {session_id}")
        self.add_conversation_message("시스템", f"🎯 새로운 학습 세션이 시작되었습니다! (ID: {session_id})")
        self.add_conversation_message("AI", "안녕하세요! 오늘 어떤 동작을 학습해볼까요? 간단한 움직임부터 시작해보시겠어요?")
    
    def start_demonstration(self):
        """시연 시작"""
        if not self.current_session:
            messagebox.showwarning("경고", "먼저 학습 세션을 시작해주세요.")
            return
        
        self.is_demonstrating = True
        self.update_system_status("시연 모드 시작")
        self.add_conversation_message("AI", "시연을 시작합니다. 캔버스를 클릭해서 로봇의 움직임을 지시해주세요.")
    
    def provide_feedback(self):
        """피드백 제공"""
        if not self.current_session:
            messagebox.showwarning("경고", "활성 학습 세션이 없습니다.")
            return
        
        # 간단한 피드백 다이얼로그
        feedback_window = tk.Toplevel(self.root)
        feedback_window.title("피드백 제공")
        feedback_window.geometry("400x300")
        
        ttk.Label(feedback_window, text="로봇의 동작에 대해 피드백을 주세요:", font=('Arial', 11)).pack(pady=10)
        
        feedback_text = scrolledtext.ScrolledText(feedback_window, height=10)
        feedback_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def submit_feedback():
            feedback = feedback_text.get(1.0, tk.END).strip()
            if feedback:
                self.add_conversation_message("사용자", f"[피드백] {feedback}")
                self.process_user_input(f"피드백: {feedback}", "text")
                feedback_window.destroy()
        
        ttk.Button(feedback_window, text="피드백 제출", command=submit_feedback).pack(pady=10)
    
    async def initialize_system(self):
        """시스템 초기화"""
        try:
            self.update_system_status("Physical AI 시스템 초기화 중...")
            
            # Physical AI 시스템 초기화
            self.physical_ai = PhysicalAI("configs/default.yaml")
            await self.physical_ai.initialize()
            
            # 음성 인터페이스 보정
            self.voice_interface.calibrate_microphone()
            
            self.update_system_status("✅ 시스템 초기화 완료")
            self.add_conversation_message("시스템", "🎉 대화형 학습 시스템이 준비되었습니다!")
            
            # 스킬 진행도 업데이트 시작
            self.start_skill_monitoring()
            
        except Exception as e:
            error_msg = f"시스템 초기화 실패: {e}"
            self.update_system_status(error_msg)
            logger.error(error_msg)
    
    def start_skill_monitoring(self):
        """스킬 모니터링 시작"""
        def update_skills():
            try:
                if self.physical_ai and hasattr(self.physical_ai, 'dev_engine'):
                    dev_engine = self.physical_ai.dev_engine
                    if hasattr(dev_engine, 'skill_engine') and hasattr(dev_engine.skill_engine, 'skills_db'):
                        for skill_name, skill in dev_engine.skill_engine.skills_db.items():
                            if skill_name in self.skill_progress_bars:
                                progress = skill.success_rate * 100
                                self.skill_progress_bars[skill_name]['value'] = progress
            except Exception as e:
                logger.error(f"스킬 업데이트 오류: {e}")
            
            # 5초 후 다시 실행
            self.root.after(5000, update_skills)
        
        # 첫 실행
        update_skills()
    
    def run(self):
        """인터페이스 실행"""
        # 시스템 초기화 (별도 스레드)
        def init_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.initialize_system())
            loop.close()
        
        threading.Thread(target=init_thread, daemon=True).start()
        
        # GUI 시작
        self.root.mainloop()

def main():
    """메인 함수"""
    interface = InteractiveLearningInterface()
    interface.run()

if __name__ == "__main__":
    main()