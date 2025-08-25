#!/usr/bin/env python3
"""
Physical AI Code - Unified Interface Entry Point

Claude Code 스타일의 Physical AI 통합 개발 환경
자연어로 로봇을 제어하고, 학습하고, 시뮬레이션할 수 있습니다.

Usage:
    python physical_ai_code.py                    # 대화형 모드
    python physical_ai_code.py --mission "작업"   # 단일 미션 실행
    python physical_ai_code.py --help             # 도움말
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from physical_ai_code.ui.cli_interface import main as cli_main

def main():
    """메인 엔트리포인트"""
    try:
        # Python 버전 확인
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 이상이 필요합니다.")
            print(f"현재 버전: {sys.version}")
            sys.exit(1)
        
        # 비동기 메인 실행
        asyncio.run(cli_main())
        
    except KeyboardInterrupt:
        print("\n👋 Physical AI Code가 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()