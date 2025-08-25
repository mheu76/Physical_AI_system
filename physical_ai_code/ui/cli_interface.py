"""
Physical AI Code - CLI Interface

Claude Code 스타일의 커맨드라인 인터페이스
"""

import asyncio
import sys
import os
from typing import Dict, Any, Optional
import argparse
from datetime import datetime
import json

# Rich를 사용한 예쁜 터미널 UI
try:
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.progress import Progress, TaskID
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
    logger = print  # Rich 사용시에는 print를 logger로 사용
except ImportError:
    RICH_AVAILABLE = False
    print("Rich를 사용할 수 없습니다. 기본 터미널 인터페이스를 사용합니다.")
    print("Rich를 설치하려면: pip install rich")
    
    # 로깅 설정
    import logging
    logger = logging.getLogger(__name__)
    
    # Rich 대체 클래스들
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
        def status(self, message):
            print(f"[진행중] {message}")
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    class Prompt:
        @staticmethod
        def ask(prompt):
            return input(f"{prompt}> ")
    
    class Panel:
        def __init__(self, content, title="", border_style=""):
            self.content = content
            self.title = title
    
    class Table:
        def __init__(self, title="", **kwargs):
            self.title = title
            self.rows = []
        def add_column(self, *args, **kwargs):
            pass
        def add_row(self, *args, **kwargs):
            self.rows.append(args)
    
    class Markdown:
        def __init__(self, text):
            self.text = text

from ..core.interface_manager import PhysicalAIInterface

class CLIInterface:
    """커맨드라인 인터페이스"""
    
    def __init__(self):
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        self.interface = PhysicalAIInterface()
        self.running = False
        self.banner_shown = False
    
    def print_banner(self):
        """시작 배너 출력"""
        if not self.banner_shown:
            banner_text = """
🤖 Physical AI Code - 통합 개발 환경

Claude Code 스타일의 Physical AI 시스템 인터페이스
자연어로 로봇을 제어하고, 학습하고, 시뮬레이션하세요!

명령어:
• /help - 도움말
• /mission <작업> - 미션 실행  
• /learn <기술> - 학습 시작
• /hardware - 하드웨어 상태
• /agent <액션> - AI 에이전트 관리 (PHI-3.5 기반)
• /quit - 종료

또는 자연어로 대화하세요:
"로봇아, 빨간 공을 집어서 상자에 넣어줘"
            """
            
            if self.console:
                self.console.print(Panel(banner_text, title="Physical AI Code", border_style="blue"))
            else:
                print(banner_text)
            
            self.banner_shown = True
    
    async def start_interactive_session(self):
        """대화형 세션 시작"""
        self.running = True
        self.print_banner()
        
        # 시스템 초기화
        if self.console:
            with self.console.status("[bold green]시스템 초기화 중..."):
                await self.interface.initialize()
        else:
            print("시스템 초기화 중...")
            await self.interface.initialize()
        
        if self.console:
            self.console.print("✅ [bold green]시스템 준비 완료![/bold green]")
        else:
            print("✅ 시스템 준비 완료!")
        
        # 메인 대화 루프
        while self.running:
            try:
                # 사용자 입력 받기
                if self.console:
                    user_input = Prompt.ask("\n[bold cyan]Physical AI[/bold cyan]")
                else:
                    user_input = input("\nPhysical AI> ")
                
                if not user_input.strip():
                    continue
                
                # 종료 명령 확인
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    await self._handle_quit()
                    break
                
                # 도움말 명령
                elif user_input.lower().startswith('/help'):
                    await self._handle_help(user_input)
                
                # 기타 명령어들
                elif user_input.startswith('/'):
                    await self._handle_slash_command(user_input)
                
                # 자연어 대화
                else:
                    await self._handle_natural_language(user_input)
                    
            except KeyboardInterrupt:
                await self._handle_quit()
                break
            except Exception as e:
                if self.console:
                    self.console.print(f"[bold red]오류: {e}[/bold red]")
                else:
                    print(f"오류: {e}")
    
    async def _handle_natural_language(self, user_input: str):
        """자연어 입력 처리"""
        if self.console:
            with self.console.status("[bold yellow]처리 중...") as status:
                result = await self.interface.process_input(user_input)
        else:
            print("처리 중...")
            result = await self.interface.process_input(user_input)
        
        if result["success"]:
            response = result["response"]["content"]
            
            if self.console:
                # 도구 사용 정보 표시
                if result.get("tool_results") and result["tool_results"].get("tools_used"):
                    tools_table = Table(title="사용된 도구", show_header=True, header_style="bold magenta")
                    tools_table.add_column("도구명", style="cyan")
                    tools_table.add_column("결과", style="green")
                    
                    for tool in result["tool_results"]["tools_used"]:
                        tools_table.add_row(tool.get("name", "Unknown"), tool.get("status", "Complete"))
                    
                    self.console.print(tools_table)
                
                # AI 응답 출력
                self.console.print(Panel(response, title="🤖 AI 응답", border_style="green"))
            else:
                print(f"\n🤖 AI 응답:\n{response}")
        else:
            error_msg = result.get("error", "알 수 없는 오류")
            if self.console:
                self.console.print(f"[bold red]❌ 오류: {error_msg}[/bold red]")
            else:
                print(f"❌ 오류: {error_msg}")
    
    async def _handle_slash_command(self, command: str):
        """슬래시 명령어 처리"""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/mission":
            await self._execute_mission(args)
        elif cmd == "/learn":
            await self._start_learning(args)
        elif cmd == "/hardware":
            await self._show_hardware_status()
        elif cmd == "/simulate":
            await self._run_simulation(args)
        elif cmd == "/status":
            await self._show_system_status()
        elif cmd == "/tools":
            await self._list_tools()
        elif cmd == "/agent":
            await self._handle_agent_command(args)
        else:
            if self.console:
                self.console.print(f"[bold red]알 수 없는 명령어: {cmd}[/bold red]")
            else:
                print(f"알 수 없는 명령어: {cmd}")
    
    async def _execute_mission(self, mission: str):
        """미션 실행"""
        if not mission:
            if self.console:
                self.console.print("[bold yellow]미션을 입력해주세요. 예: /mission '빨간 컵을 테이블로 옮기기'[/bold yellow]")
            else:
                print("미션을 입력해주세요. 예: /mission '빨간 컵을 테이블로 옮기기'")
            return
        
        if self.console:
            with self.console.status(f"[bold blue]미션 실행 중: {mission}"):
                result = await self.interface.execute_command("mission", mission=mission)
        else:
            print(f"미션 실행 중: {mission}")
            result = await self.interface.execute_command("mission", mission=mission)
        
        self._display_result(result, f"미션 '{mission}' 실행 결과")
    
    async def _start_learning(self, skill: str):
        """학습 시작"""
        if not skill:
            if self.console:
                self.console.print("[bold yellow]학습할 기술을 입력해주세요. 예: /learn grasp_skill[/bold yellow]")
            else:
                print("학습할 기술을 입력해주세요. 예: /learn grasp_skill")
            return
        
        if self.console:
            with self.console.status(f"[bold blue]학습 중: {skill}"):
                result = await self.interface.execute_command("learn", skill=skill)
        else:
            print(f"학습 중: {skill}")
            result = await self.interface.execute_command("learn", skill=skill)
        
        self._display_result(result, f"'{skill}' 학습 결과")
    
    async def _show_hardware_status(self):
        """하드웨어 상태 표시"""
        if self.console:
            with self.console.status("[bold blue]하드웨어 상태 확인 중..."):
                result = await self.interface.execute_command("hardware_status")
        else:
            print("하드웨어 상태 확인 중...")
            result = await self.interface.execute_command("hardware_status")
        
        self._display_result(result, "하드웨어 상태")
    
    async def _run_simulation(self, scenario: str):
        """시뮬레이션 실행"""
        if self.console:
            with self.console.status(f"[bold blue]시뮬레이션 실행 중: {scenario or 'default'}"):
                result = await self.interface.execute_command("simulate", scenario=scenario)
        else:
            print(f"시뮬레이션 실행 중: {scenario or 'default'}")
            result = await self.interface.execute_command("simulate", scenario=scenario)
        
        self._display_result(result, "시뮬레이션 결과")
    
    async def _show_system_status(self):
        """시스템 상태 표시"""
        status = await self.interface._get_system_status()
        
        if self.console:
            status_table = Table(title="시스템 상태", show_header=True, header_style="bold blue")
            status_table.add_column("항목", style="cyan")
            status_table.add_column("상태", style="green")
            
            status_table.add_row("초기화 완료", "✅" if status["initialized"] else "❌")
            status_table.add_row("Physical AI", "✅" if status["physical_ai_ready"] else "❌")
            status_table.add_row("사용 가능한 도구", str(status["tools_available"]))
            status_table.add_row("세션 활성", "✅" if status["session_active"] else "❌")
            status_table.add_row("업데이트 시간", status["timestamp"].strftime("%Y-%m-%d %H:%M:%S"))
            
            self.console.print(status_table)
        else:
            print("\n📊 시스템 상태:")
            print(f"  초기화 완료: {'✅' if status['initialized'] else '❌'}")
            print(f"  Physical AI: {'✅' if status['physical_ai_ready'] else '❌'}")
            print(f"  사용 가능한 도구: {status['tools_available']}")
            print(f"  세션 활성: {'✅' if status['session_active'] else '❌'}")
            print(f"  업데이트 시간: {status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    async def _list_tools(self):
        """도구 목록 표시"""
        tools = await self.interface.list_available_tools()
        
        if self.console:
            tools_table = Table(title="사용 가능한 도구", show_header=True, header_style="bold magenta")
            tools_table.add_column("도구명", style="cyan")
            tools_table.add_column("설명", style="green")
            tools_table.add_column("상태", style="yellow")
            
            for tool in tools:
                tools_table.add_row(
                    tool.get("name", "Unknown"),
                    tool.get("description", "No description"),
                    tool.get("status", "Available")
                )
            
            self.console.print(tools_table)
        else:
            print("\n🛠️  사용 가능한 도구:")
            for tool in tools:
                print(f"  • {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
    
    async def _handle_help(self, command: str):
        """도움말 처리"""
        parts = command.strip().split(maxsplit=1)
        topic = parts[1] if len(parts) > 1 else None
        
        help_text = await self.interface.get_help(topic)
        
        if self.console:
            self.console.print(Panel(Markdown(help_text), title="📚 도움말", border_style="blue"))
        else:
            print(f"\n📚 도움말:\n{help_text}")
    
    def _display_result(self, result: Dict[str, Any], title: str):
        """결과 표시"""
        if self.console:
            if result.get("success"):
                content = result.get("message", "작업 완료")
                self.console.print(Panel(content, title=f"✅ {title}", border_style="green"))
            else:
                error = result.get("error", "알 수 없는 오류")
                self.console.print(Panel(error, title=f"❌ {title}", border_style="red"))
        else:
            if result.get("success"):
                print(f"\n✅ {title}:")
                print(f"  {result.get('message', '작업 완료')}")
            else:
                print(f"\n❌ {title}:")
                print(f"  {result.get('error', '알 수 없는 오류')}")
    
    async def _handle_quit(self):
        """종료 처리"""
        self.running = False
        if self.console:
            self.console.print("[bold blue]시스템 종료 중...[/bold blue]")
        else:
            print("시스템 종료 중...")
        
        await self.interface.shutdown()
        
        if self.console:
            self.console.print("[bold green]👋 Physical AI Code를 사용해 주셔서 감사합니다![/bold green]")
        else:
            print("👋 Physical AI Code를 사용해 주셔서 감사합니다!")
    
    async def _handle_agent_command(self, args: str):
        """에이전트 명령어 처리"""
        if not args:
            # 기본적으로 목록 표시
            args = "list"
        
        if self.console:
            with self.console.status(f"[bold blue]에이전트 작업 처리 중: {args}"):
                result = await self.interface.execute_command("agent", agent_args=args)
        else:
            print(f"에이전트 작업 처리 중: {args}")
            result = await self.interface.execute_command("agent", agent_args=args)
        
        self._display_agent_result(result, args)
    
    def _display_agent_result(self, result: Dict[str, Any], command_args: str):
        """에이전트 작업 결과 표시"""
        if not result.get("success"):
            if self.console:
                self.console.print(f"[bold red]❌ 에이전트 작업 실패: {result.get('error', '알 수 없는 오류')}[/bold red]")
            else:
                print(f"❌ 에이전트 작업 실패: {result.get('error', '알 수 없는 오류')}")
            return
        
        details = result.get("details", {})
        action = details.get("action", "unknown")
        
        if action == "create":
            self._display_agent_creation_result(details)
        elif action == "update":
            self._display_agent_update_result(details)
        elif action == "execute":
            self._display_agent_execution_result(details)
        elif action == "list":
            self._display_agent_list_result(details)
        elif action == "info":
            self._display_agent_info_result(details)
        elif action == "delete":
            self._display_agent_deletion_result(details)
        else:
            if self.console:
                self.console.print(Panel(str(details), title="🤖 에이전트 작업 결과", border_style="green"))
            else:
                print(f"\n🤖 에이전트 작업 결과:\n{details}")
    
    def _display_agent_creation_result(self, details: Dict[str, Any]):
        """에이전트 생성 결과 표시"""
        agent = details.get("agent", {})
        message = details.get("message", "에이전트가 생성되었습니다.")
        
        if self.console:
            # 에이전트 정보 테이블
            agent_table = Table(title="🤖 새로운 에이전트 생성됨", show_header=True, header_style="bold green")
            agent_table.add_column("속성", style="cyan")
            agent_table.add_column("값", style="green")
            
            agent_table.add_row("ID", agent.get("id", "N/A"))
            agent_table.add_row("이름", agent.get("name", "N/A"))
            agent_table.add_row("설명", agent.get("description", "N/A"))
            agent_table.add_row("능력", ", ".join(agent.get("capabilities", [])))
            agent_table.add_row("행동 패턴", str(agent.get("behaviors_count", 0)) + "개")
            agent_table.add_row("태그", ", ".join(agent.get("tags", [])))
            agent_table.add_row("생성일", agent.get("created_at", "N/A"))
            
            self.console.print(agent_table)
            self.console.print(f"[bold green]✅ {message}[/bold green]")
        else:
            print(f"\n🤖 새로운 에이전트 생성됨:")
            print(f"  ID: {agent.get('id', 'N/A')}")
            print(f"  이름: {agent.get('name', 'N/A')}")
            print(f"  설명: {agent.get('description', 'N/A')}")
            print(f"  능력: {', '.join(agent.get('capabilities', []))}")
            print(f"  ✅ {message}")
    
    def _display_agent_update_result(self, details: Dict[str, Any]):
        """에이전트 업데이트 결과 표시"""
        agent = details.get("agent", {})
        message = details.get("message", "에이전트가 업데이트되었습니다.")
        
        if self.console:
            update_table = Table(title="🔄 에이전트 업데이트됨", show_header=True, header_style="bold blue")
            update_table.add_column("속성", style="cyan")
            update_table.add_column("값", style="blue")
            
            update_table.add_row("이름", agent.get("name", "N/A"))
            update_table.add_row("업데이트일", agent.get("updated_at", "N/A"))
            
            # 업데이트 히스토리 표시
            history = agent.get("update_history", [])
            if history:
                update_table.add_row("최근 업데이트", history[-1].get("instruction", "N/A"))
            
            self.console.print(update_table)
            self.console.print(f"[bold blue]✅ {message}[/bold blue]")
        else:
            print(f"\n🔄 에이전트 업데이트됨:")
            print(f"  이름: {agent.get('name', 'N/A')}")
            print(f"  ✅ {message}")
    
    def _display_agent_execution_result(self, details: Dict[str, Any]):
        """에이전트 실행 결과 표시"""
        agent_name = details.get("agent_name", "Unknown")
        execution_count = details.get("execution_count", 0)
        success_rate = details.get("success_rate", 0.0)
        results = details.get("results", [])
        
        if self.console:
            exec_table = Table(title=f"🚀 에이전트 '{agent_name}' 실행 완료", show_header=True, header_style="bold yellow")
            exec_table.add_column("메트릭", style="cyan")
            exec_table.add_column("값", style="yellow")
            
            exec_table.add_row("실행 횟수", str(execution_count))
            exec_table.add_row("성공률", f"{success_rate}%")
            exec_table.add_row("실행된 행동", str(len(results)))
            
            self.console.print(exec_table)
            
            # 실행 결과 상세
            if results:
                for i, result in enumerate(results, 1):
                    behavior = result.get("behavior", "Unknown")
                    status = "✅ 성공" if result.get("result", {}).get("success") else "❌ 실패"
                    self.console.print(f"  {i}. {behavior}: {status}")
                    
        else:
            print(f"\n🚀 에이전트 '{agent_name}' 실행 완료:")
            print(f"  실행 횟수: {execution_count}")
            print(f"  성공률: {success_rate}%")
            print(f"  실행된 행동: {len(results)}개")
    
    def _display_agent_list_result(self, details: Dict[str, Any]):
        """에이전트 목록 결과 표시"""
        agents = details.get("agents", [])
        count = details.get("count", 0)
        
        if self.console:
            if count == 0:
                self.console.print("[bold yellow]등록된 에이전트가 없습니다.[/bold yellow]")
                self.console.print("[dim]'/agent create <지시사항>'으로 새 에이전트를 생성하세요.[/dim]")
                return
            
            agents_table = Table(title=f"🤖 등록된 에이전트 목록 ({count}개)", show_header=True, header_style="bold magenta")
            agents_table.add_column("이름", style="cyan", width=20)
            agents_table.add_column("설명", style="green", width=30)
            agents_table.add_column("상태", style="yellow", width=10)
            agents_table.add_column("실행횟수", style="blue", width=10)
            agents_table.add_column("성공률", style="red", width=10)
            agents_table.add_column("태그", style="magenta", width=15)
            
            for agent in agents:
                agents_table.add_row(
                    agent.get("name", "N/A")[:18] + "..." if len(agent.get("name", "")) > 18 else agent.get("name", "N/A"),
                    agent.get("description", "N/A")[:28] + "..." if len(agent.get("description", "")) > 28 else agent.get("description", "N/A"),
                    agent.get("status", "N/A"),
                    str(agent.get("execution_count", 0)),
                    f"{agent.get('success_rate', 0)}%",
                    ", ".join(agent.get("tags", []))[:13] + "..." if len(", ".join(agent.get("tags", []))) > 13 else ", ".join(agent.get("tags", []))
                )
            
            self.console.print(agents_table)
            self.console.print(f"[dim]'/agent info <이름>'으로 상세 정보를 확인할 수 있습니다.[/dim]")
        else:
            print(f"\n🤖 등록된 에이전트 목록 ({count}개):")
            if count == 0:
                print("  등록된 에이전트가 없습니다.")
                print("  '/agent create <지시사항>'으로 새 에이전트를 생성하세요.")
            else:
                for agent in agents:
                    print(f"  • {agent.get('name', 'N/A')}: {agent.get('description', 'N/A')}")
                    print(f"    상태: {agent.get('status', 'N/A')} | 실행: {agent.get('execution_count', 0)}회 | 성공률: {agent.get('success_rate', 0)}%")
    
    def _display_agent_info_result(self, details: Dict[str, Any]):
        """에이전트 상세 정보 결과 표시"""
        agent = details.get("agent", {})
        
        if self.console:
            # 기본 정보 패널
            info_text = f"""
**ID**: {agent.get('id', 'N/A')}
**이름**: {agent.get('name', 'N/A')}
**설명**: {agent.get('description', 'N/A')}
**상태**: {agent.get('status', 'N/A')}
**생성일**: {agent.get('created_at', 'N/A')}
**최종 업데이트**: {agent.get('updated_at', 'N/A')}
**실행 횟수**: {agent.get('execution_count', 0)}
**성공률**: {agent.get('success_rate', 0)}%
**최종 실행**: {agent.get('last_executed', 'N/A')}

**원본 지시사항**: 
{agent.get('creator_instruction', 'N/A')}
            """
            
            self.console.print(Panel(Markdown(info_text.strip()), title=f"🤖 에이전트 상세 정보", border_style="blue"))
            
            # 능력 테이블
            capabilities = agent.get("capabilities", [])
            if capabilities:
                cap_table = Table(title="💪 에이전트 능력", show_header=True, header_style="bold green")
                cap_table.add_column("능력명", style="cyan")
                cap_table.add_column("설명", style="green")
                cap_table.add_column("필요 도구", style="yellow")
                
                for cap in capabilities:
                    cap_table.add_row(
                        cap.get("name", "N/A"),
                        cap.get("description", "N/A")[:40] + "..." if len(cap.get("description", "")) > 40 else cap.get("description", "N/A"),
                        ", ".join(cap.get("tools_required", []))
                    )
                
                self.console.print(cap_table)
            
            # 행동 패턴
            behaviors = agent.get("behaviors", [])
            if behaviors:
                beh_table = Table(title="🎯 행동 패턴", show_header=True, header_style="bold purple")
                beh_table.add_column("트리거", style="cyan")
                beh_table.add_column("우선순위", style="yellow")
                beh_table.add_column("동작 수", style="green")
                
                for beh in behaviors:
                    beh_table.add_row(
                        beh.get("trigger", "N/A"),
                        str(beh.get("priority", 0)),
                        str(beh.get("action_count", 0))
                    )
                
                self.console.print(beh_table)
            
        else:
            print(f"\n🤖 에이전트 상세 정보:")
            print(f"  ID: {agent.get('id', 'N/A')}")
            print(f"  이름: {agent.get('name', 'N/A')}")
            print(f"  설명: {agent.get('description', 'N/A')}")
            print(f"  상태: {agent.get('status', 'N/A')}")
            print(f"  실행 횟수: {agent.get('execution_count', 0)}")
            print(f"  성공률: {agent.get('success_rate', 0)}%")
    
    def _display_agent_deletion_result(self, details: Dict[str, Any]):
        """에이전트 삭제 결과 표시"""
        agent_name = details.get("agent_name", "Unknown")
        message = details.get("message", "에이전트가 삭제되었습니다.")
        
        if self.console:
            self.console.print(f"[bold red]🗑️ 에이전트 '{agent_name}'가 삭제되었습니다.[/bold red]")
        else:
            print(f"\n🗑️ 에이전트 '{agent_name}'가 삭제되었습니다.")

def create_cli_parser():
    """CLI 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description="Physical AI Code - Unified Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  physical-ai-code                    # 대화형 모드 시작
  physical-ai-code --mission "컵 옮기기"  # 단일 미션 실행
  physical-ai-code --config custom.yaml  # 사용자 설정 파일 사용
        """
    )
    
    parser.add_argument("--config", "-c", default="configs/default.yaml",
                       help="설정 파일 경로 (기본값: configs/default.yaml)")
    parser.add_argument("--mission", "-m", help="실행할 미션")
    parser.add_argument("--batch", "-b", help="배치 명령 파일")
    parser.add_argument("--output", "-o", choices=["rich", "plain", "json"], 
                       default="rich", help="출력 형식")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    
    return parser

async def main():
    """메인 함수"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # CLI 인터페이스 생성
    cli = CLIInterface()
    cli.interface.config_path = args.config
    
    if args.debug:
        cli.interface.interface_config.debug_mode = True
    
    try:
        if args.mission:
            # 단일 미션 모드
            await cli.interface.initialize()
            result = await cli.interface.execute_command("mission", mission=args.mission)
            cli._display_result(result, f"미션 '{args.mission}'")
        elif args.batch:
            # 배치 모드 (향후 구현)
            print("배치 모드는 아직 구현되지 않았습니다.")
        else:
            # 대화형 모드
            await cli.start_interactive_session()
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())