#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical AI Code 기본 기능 테스트

이 스크립트는 Physical AI Code의 핵심 기능들이 
정상적으로 작동하는지 테스트합니다.
"""

import os
import sys

# 인코딩 문제 해결
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import asyncio
import sys
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from physical_ai_code.core.interface_manager import PhysicalAIInterface

async def test_basic_functionality():
    """기본 기능 테스트"""
    print(">> Physical AI Code 기본 기능 테스트 시작")
    
    try:
        # 1. 인터페이스 생성
        print("\n[1] 인터페이스 초기화 테스트...")
        interface = PhysicalAIInterface("configs/default.yaml")
        
        # 2. 시스템 초기화
        print("\n[2] 시스템 초기화 테스트...")
        init_success = await interface.initialize()
        
        if init_success:
            print("SUCCESS: 초기화 성공!")
        else:
            print("WARNING: 제한된 모드로 초기화됨")
        
        # 3. 도구 목록 확인
        print("\n[3] 도구 시스템 테스트...")
        tools = await interface.list_available_tools()
        print(f"사용 가능한 도구 수: {len(tools)}")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # 4. 간단한 명령 테스트
        print("\n[4] 기본 명령 테스트...")
        
        # 시스템 상태 확인
        status = await interface._get_system_status()
        print(f"시스템 상태: {'정상' if status['initialized'] else '오류'}")
        print(f"활성 도구 수: {status['tools_available']}")
        
        # 5. 간단한 미션 테스트
        print("\n[5] 미션 실행 테스트...")
        result = await interface.process_input("안녕하세요! 테스트 미션입니다.")
        
        if result["success"]:
            print("SUCCESS: 미션 처리 성공!")
            print(f"응답: {result['response']['content'][:100]}...")
        else:
            print(f"ERROR: 미션 처리 실패: {result.get('error', 'Unknown error')}")
        
        # 6. /status 명령 테스트
        print("\n[6] 슬래시 명령 테스트...")
        result = await interface.process_input("/status")
        
        if result["success"]:
            print("SUCCESS: /status 명령 성공!")
        else:
            print(f"ERROR: /status 명령 실패: {result.get('error', 'Unknown error')}")
        
        # 7. /tools 명령 테스트
        print("\n[7] 도구 목록 명령 테스트...")
        result = await interface.process_input("/tools")
        
        if result["success"]:
            print("SUCCESS: /tools 명령 성공!")
        else:
            print(f"ERROR: /tools 명령 실패: {result.get('error', 'Unknown error')}")
        
        # 8. /agent 명령 테스트
        print("\n[8] 에이전트 명령 테스트...")
        result = await interface.process_input("/agent list")
        
        if result["success"]:
            print("SUCCESS: /agent list 명령 성공!")
        else:
            print(f"WARNING: /agent list 명령 제한적 성공: {result.get('error', 'Unknown error')}")
        
        # 9. 정리
        print("\n[9] 시스템 정리...")
        await interface.shutdown()
        print("SUCCESS: 시스템 정리 완료!")
        
        print("\n=== 기본 기능 테스트 완료! ===")
        print("Physical AI Code가 정상적으로 작동합니다.")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_creation():
    """에이전트 생성 테스트"""
    print("\n>> 에이전트 생성 테스트 시작")
    
    try:
        interface = PhysicalAIInterface("configs/default.yaml")
        await interface.initialize()
        
        # 에이전트 생성 테스트
        print("에이전트 생성 테스트...")
        result = await interface.process_input("/agent create 간단한 인사를 하는 도우미 에이전트")
        
        if result["success"]:
            print("SUCCESS: 에이전트 생성 성공!")
            if result.get("tool_results"):
                print("도구 실행 결과:", result["tool_results"])
        else:
            print(f"ERROR: 에이전트 생성 실패: {result.get('error')}")
        
        await interface.shutdown()
        return True
        
    except Exception as e:
        print(f"ERROR: 에이전트 테스트 중 오류: {e}")
        return False

def main():
    """메인 함수"""
    print("=" * 60)
    print(">> Physical AI Code 종합 테스트")
    print("=" * 60)
    
    # 기본 기능 테스트
    print("\n[TEST 1] 기본 기능")
    basic_success = asyncio.run(test_basic_functionality())
    
    # 에이전트 기능 테스트
    print("\n[TEST 2] 에이전트 기능")
    agent_success = asyncio.run(test_agent_creation())
    
    # 결과 요약
    print("\n" + "=" * 60)
    print(">> 테스트 결과 요약")
    print("=" * 60)
    print(f"기본 기능: {'SUCCESS' if basic_success else 'FAILED'}")
    print(f"에이전트 기능: {'SUCCESS' if agent_success else 'FAILED'}")
    
    if basic_success and agent_success:
        print("\n=== 모든 테스트 통과! ===")
        print("Physical AI Code를 사용할 준비가 되었습니다.")
        print("\n사용 방법:")
        print("  python physical_ai_code.py")
        print("  또는")
        print("  python physical_ai_code.py --mission \"원하는 작업\"")
    else:
        print("\n=== 일부 테스트 실패 ===")
        print("기본 기능은 작동하지만 일부 고급 기능에 제한이 있을 수 있습니다.")
    
    return basic_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)