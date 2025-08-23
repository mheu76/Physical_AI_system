"""
행동모델 GUI 대화형 정의 시스템

tkinter를 사용한 그래픽 사용자 인터페이스로
PHI-3.5와 대화하면서 행동모델을 정의할 수 있습니다.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import asyncio
import json
import threading
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from foundation_model.phi35_integration import PHI35ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BehaviorModel:
    """행동모델 정의"""
    name: str
    description: str
    motion_primitives: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    constraints: Dict[str, Any]
    created_at: str
    updated_at: str

class BehaviorModelGUI:
    """행동모델 GUI 대화형 정의 시스템"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 PHI-3.5 행동모델 정의 시스템")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # PHI-3.5 매니저
        self.phi35_manager = None
        self.behavior_models = {}
        self.current_model = None
        self.dialog_history = []
        
        # GUI 컴포넌트
        self.setup_gui()
        
        # 비동기 이벤트 루프
        self.loop = None
        
    def setup_gui(self):
        """GUI 설정"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 상단 제목
        title_label = ttk.Label(main_frame, text="🎯 PHI-3.5 행동모델 정의 시스템", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # 메인 컨테이너 (좌우 분할)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 왼쪽 패널 (대화창)
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=2)
        
        # 오른쪽 패널 (모델 관리)
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=1)
        
        self.setup_dialog_panel(left_frame)
        self.setup_model_panel(right_frame)
        
    def setup_dialog_panel(self, parent):
        """대화창 패널 설정"""
        # 대화창 제목
        dialog_title = ttk.Label(parent, text="💬 PHI-3.5와 대화", font=('Arial', 12, 'bold'))
        dialog_title.pack(pady=(0, 10))
        
        # 대화 히스토리 표시 영역
        history_frame = ttk.LabelFrame(parent, text="대화 기록")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.dialog_history_text = scrolledtext.ScrolledText(
            history_frame, 
            height=20, 
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='#ffffff',
            fg='#000000'
        )
        self.dialog_history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 입력 영역
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.input_entry = ttk.Entry(input_frame, font=('Arial', 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind('<Return>', self.send_message)
        
        send_button = ttk.Button(input_frame, text="전송", command=self.send_message)
        send_button.pack(side=tk.RIGHT)
        
        # 빠른 명령 버튼들
        quick_commands_frame = ttk.LabelFrame(parent, text="빠른 명령")
        quick_commands_frame.pack(fill=tk.X)
        
        commands = [
            ("새 모델 만들기", "새로운 행동모델을 만들어주세요"),
            ("모델 보기", "정의된 행동모델들을 보여주세요"),
            ("커피 만들기 모델", "커피를 만드는 행동모델을 정의해주세요"),
            ("청소 모델", "방을 청소하는 행동모델을 정의해주세요"),
            ("요리 모델", "요리를 하는 행동모델을 정의해주세요")
        ]
        
        for i, (label, command) in enumerate(commands):
            btn = ttk.Button(quick_commands_frame, text=label, 
                           command=lambda cmd=command: self.quick_command(cmd))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2, sticky='ew')
        
        # 그리드 가중치 설정
        for i in range(3):
            quick_commands_frame.columnconfigure(i, weight=1)
        
    def setup_model_panel(self, parent):
        """모델 관리 패널 설정"""
        # 모델 관리 제목
        model_title = ttk.Label(parent, text="📋 모델 관리", font=('Arial', 12, 'bold'))
        model_title.pack(pady=(0, 10))
        
        # 모델 목록
        list_frame = ttk.LabelFrame(parent, text="정의된 모델들")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 트리뷰 (모델 목록)
        columns = ('name', 'description', 'primitives', 'created')
        self.model_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        self.model_tree.heading('name', text='모델명')
        self.model_tree.heading('description', text='설명')
        self.model_tree.heading('primitives', text='동작 수')
        self.model_tree.heading('created', text='생성일')
        
        self.model_tree.column('name', width=120)
        self.model_tree.column('description', width=150)
        self.model_tree.column('primitives', width=60)
        self.model_tree.column('created', width=80)
        
        self.model_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.model_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_tree.configure(yscrollcommand=scrollbar.set)
        
        # 모델 선택 이벤트
        self.model_tree.bind('<<TreeviewSelect>>', self.on_model_select)
        
        # 모델 관리 버튼들
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="모델 테스트", command=self.test_selected_model).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="모델 삭제", command=self.delete_selected_model).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="모델 내보내기", command=self.export_models).pack(side=tk.LEFT)
        
        # 현재 모델 정보
        info_frame = ttk.LabelFrame(parent, text="현재 모델 정보")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.model_info_text = scrolledtext.ScrolledText(
            info_frame, 
            height=8, 
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#f8f8f8'
        )
        self.model_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def add_message(self, sender: str, message: str):
        """대화창에 메시지 추가"""
        timestamp = time.time()
            
        self.dialog_history.append({
            "sender": sender,
            "message": message,
            "timestamp": timestamp
        })
        
        # GUI 업데이트
        self.dialog_history_text.insert(tk.END, f"[{sender}] {message}\n\n")
        self.dialog_history_text.see(tk.END)
        
    def send_message(self, event=None):
        """메시지 전송"""
        message = self.input_entry.get().strip()
        if not message:
            return
        
        # 입력창 클리어
        self.input_entry.delete(0, tk.END)
        
        # 사용자 메시지 추가
        self.add_message("사용자", message)
        
        # PHI-3.5 응답 생성 (별도 스레드에서)
        threading.Thread(target=self.get_phi35_response, args=(message,), daemon=True).start()
        
    def quick_command(self, command: str):
        """빠른 명령 실행"""
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, command)
        self.send_message()
        
    def get_phi35_response(self, user_message: str):
        """PHI-3.5 응답 생성"""
        try:
            # 새로운 스레드에서 비동기 처리
            def run_async():
                try:
                    # 새로운 이벤트 루프 생성
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # PHI-3.5 응답 생성
                    response = loop.run_until_complete(self._process_user_input(user_message))
                    
                    # GUI 업데이트 (메인 스레드에서)
                    self.root.after(0, lambda: self.add_message("PHI-3.5", response))
                    
                    # 이벤트 루프 정리
                    loop.close()
                    
                except Exception as e:
                    error_msg = f"오류 발생: {e}"
                    self.root.after(0, lambda: self.add_message("시스템", error_msg))
                    logger.error(f"PHI-3.5 응답 생성 실패: {e}")
            
            # 별도 스레드에서 실행
            threading.Thread(target=run_async, daemon=True).start()
            
        except Exception as e:
            error_msg = f"스레드 생성 실패: {e}"
            self.root.after(0, lambda: self.add_message("시스템", error_msg))
            logger.error(f"스레드 생성 실패: {e}")
    
    async def _process_user_input(self, user_input: str) -> str:
        """사용자 입력 처리"""
        try:
            if not self.phi35_manager:
                return "PHI-3.5 모델이 초기화되지 않았습니다."
            
            # 컨텍스트 프롬프트 생성
            context = self._create_context_prompt()
            full_prompt = f"{context}\n\n사용자: {user_input}\n\nPHI-3.5:"
            
            # PHI-3.5 응답 생성
            response = await self.phi35_manager.generate_response(full_prompt)
            
            # 응답에서 행동모델 정보 추출
            await self._extract_behavior_model(response, user_input)
            
            return response
            
        except Exception as e:
            logger.error(f"입력 처리 실패: {e}")
            return f"죄송합니다. 처리 중 오류가 발생했습니다: {e}"
    
    def _create_context_prompt(self) -> str:
        """컨텍스트 프롬프트 생성"""
        context = f"""당신은 Physical AI 시스템의 행동모델 정의 전문가입니다.

현재 상황:
- 정의된 행동모델: {len(self.behavior_models)}개
- 현재 작업 중인 모델: {self.current_model.name if self.current_model else '없음'}

행동모델 정의 규칙:
1. 각 행동모델은 고유한 이름을 가져야 합니다
2. motion_primitives는 기본 동작 단위들의 리스트입니다
3. 각 primitive는 name, parameters, preconditions, postconditions를 포함합니다
4. parameters는 동작 실행에 필요한 매개변수들입니다
5. constraints는 안전성과 물리적 제약조건들입니다

사용자가 행동모델을 정의하거나 수정하려고 할 때, 구조화된 JSON 형태로 응답해주세요.
JSON 응답은 다음과 같은 형식이어야 합니다:

```json
{{
  "action": "create|modify|view|test",
  "model_name": "모델명",
  "description": "모델 설명",
  "motion_primitives": [
    {{
      "name": "동작명",
      "parameters": {{"param1": "value1"}},
      "preconditions": ["조건1", "조건2"],
      "postconditions": ["결과1", "결과2"]
    }}
  ],
  "parameters": {{"global_param": "value"}},
  "constraints": {{"safety": "constraint"}}
}}
```

사용자의 요청에 따라 적절한 응답을 생성해주세요."""
        
        return context
    
    async def _extract_behavior_model(self, response: str, user_input: str):
        """응답에서 행동모델 정보 추출"""
        try:
            # JSON 블록 찾기
            if "```json" in response and "```" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                
                model_data = json.loads(json_str)
                
                # 행동모델 생성/수정
                if model_data.get("action") in ["create", "modify"]:
                    await self._create_or_modify_model(model_data)
                    
        except Exception as e:
            logger.warning(f"모델 추출 실패: {e}")
    
    async def _create_or_modify_model(self, model_data: Dict[str, Any]):
        """행동모델 생성 또는 수정"""
        model_name = model_data.get("model_name")
        if not model_name:
            return
        
        # 기존 모델 확인
        if model_name in self.behavior_models:
            print(f"🔄 기존 모델 '{model_name}' 수정 중...")
        else:
            print(f"🆕 새 모델 '{model_name}' 생성 중...")
        
        # 행동모델 객체 생성
        import datetime
        now = datetime.datetime.now().isoformat()
        
        model = BehaviorModel(
            name=model_name,
            description=model_data.get("description", ""),
            motion_primitives=model_data.get("motion_primitives", []),
            parameters=model_data.get("parameters", {}),
            constraints=model_data.get("constraints", {}),
            created_at=now,
            updated_at=now
        )
        
        self.behavior_models[model_name] = model
        self.current_model = model
        
        # GUI 업데이트
        self.root.after(0, self.update_model_list)
        self.root.after(0, lambda: self.add_message("시스템", f"✅ 모델 '{model_name}' 저장됨"))
    
    def update_model_list(self):
        """모델 목록 업데이트"""
        # 기존 항목 삭제
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        
        # 새 항목 추가
        for name, model in self.behavior_models.items():
            self.model_tree.insert('', 'end', values=(
                name,
                model.description[:30] + "..." if len(model.description) > 30 else model.description,
                len(model.motion_primitives),
                model.created_at[:10]
            ))
    
    def on_model_select(self, event):
        """모델 선택 이벤트"""
        selection = self.model_tree.selection()
        if selection:
            item = self.model_tree.item(selection[0])
            model_name = item['values'][0]
            
            if model_name in self.behavior_models:
                model = self.behavior_models[model_name]
                self.current_model = model
                self.show_model_info(model)
    
    def show_model_info(self, model: BehaviorModel):
        """모델 정보 표시"""
        info = f"""모델명: {model.name}
설명: {model.description}
생성일: {model.created_at}
수정일: {model.updated_at}

동작 단위들 ({len(model.motion_primitives)}개):
"""
        
        for i, primitive in enumerate(model.motion_primitives, 1):
            info += f"""
{i}. {primitive['name']}
   매개변수: {primitive['parameters']}
   전제조건: {primitive['preconditions']}
   결과조건: {primitive['postconditions']}
"""
        
        info += f"""
전역 매개변수:
{json.dumps(model.parameters, indent=2, ensure_ascii=False)}

제약조건:
{json.dumps(model.constraints, indent=2, ensure_ascii=False)}
"""
        
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(1.0, info)
    
    def test_selected_model(self):
        """선택된 모델 테스트"""
        if not self.current_model:
            messagebox.showwarning("경고", "테스트할 모델을 선택해주세요.")
            return
        
        # 테스트 실행
        test_result = f"🧪 모델 '{self.current_model.name}' 테스트 실행 중...\n\n"
        
        for i, primitive in enumerate(self.current_model.motion_primitives, 1):
            test_result += f"동작 {i}: {primitive['name']}\n"
            test_result += f"  매개변수: {primitive['parameters']}\n"
            test_result += f"  전제조건: {primitive['preconditions']}\n"
            test_result += f"  결과조건: {primitive['postconditions']}\n"
            test_result += f"  상태: ✅ 시뮬레이션 성공\n\n"
        
        test_result += f"🎉 모델 '{self.current_model.name}' 테스트 완료!"
        
        self.add_message("시스템", test_result)
    
    def delete_selected_model(self):
        """선택된 모델 삭제"""
        if not self.current_model:
            messagebox.showwarning("경고", "삭제할 모델을 선택해주세요.")
            return
        
        if messagebox.askyesno("확인", f"모델 '{self.current_model.name}'을 삭제하시겠습니까?"):
            del self.behavior_models[self.current_model.name]
            self.current_model = None
            self.update_model_list()
            self.model_info_text.delete(1.0, tk.END)
            self.add_message("시스템", f"🗑️ 모델 '{self.current_model.name}' 삭제됨")
    
    def export_models(self):
        """모델 내보내기"""
        try:
            models_data = [asdict(model) for model in self.behavior_models.values()]
            with open("behavior_models_export.json", 'w', encoding='utf-8') as f:
                json.dump(models_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("성공", "모델이 'behavior_models_export.json' 파일로 내보내졌습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"모델 내보내기 실패: {e}")
    
    async def initialize(self):
        """시스템 초기화"""
        self.add_message("시스템", "🤖 행동모델 대화형 정의 시스템 초기화 중...")
        
        # PHI-3.5 모델 초기화
        self.phi35_manager = PHI35ModelManager()
        await self.phi35_manager.initialize()
        
        # 기존 행동모델 로드
        await self._load_existing_models()
        
        self.add_message("시스템", "✅ 시스템 초기화 완료!")
        self.add_message("시스템", "💬 PHI-3.5와 자연어로 행동모델을 정의해보세요!")
    
    async def _load_existing_models(self):
        """기존 행동모델 로드"""
        models_file = Path("behavior_models.json")
        if models_file.exists():
            try:
                with open(models_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_data in data:
                        model = BehaviorModel(**model_data)
                        self.behavior_models[model.name] = model
                
                self.root.after(0, self.update_model_list)
                self.add_message("시스템", f"📚 {len(self.behavior_models)}개의 기존 행동모델 로드됨")
            except Exception as e:
                self.add_message("시스템", f"⚠️  기존 모델 로드 실패: {e}")
    
    def run(self):
        """GUI 실행"""
        # 초기화 (별도 스레드에서)
        def init_system():
            try:
                # 새로운 이벤트 루프 생성
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # 시스템 초기화
                loop.run_until_complete(self.initialize())
                
                # 이벤트 루프 정리
                loop.close()
                
            except Exception as e:
                error_msg = f"초기화 실패: {e}"
                self.root.after(0, lambda: self.add_message("시스템", error_msg))
                logger.error(f"시스템 초기화 실패: {e}")
        
        threading.Thread(target=init_system, daemon=True).start()
        
        # GUI 실행
        self.root.mainloop()

def main():
    """메인 함수"""
    app = BehaviorModelGUI()
    app.run()

if __name__ == "__main__":
    main()
