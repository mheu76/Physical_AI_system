#!/usr/bin/env python3
"""
DynamicCache 오류 수정 테스트
간단한 PHI-3.5 텍스트 생성 테스트
"""

import asyncio
import logging
from foundation_model.phi35_integration import create_phi35_physical_ai

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_cache_fix():
    """캐시 수정 테스트"""
    logger.info("🔧 DynamicCache 수정 테스트 시작")
    
    try:
        # PHI-3.5 생성
        phi35_ai = create_phi35_physical_ai(
            model_name="microsoft/Phi-3.5-mini-instruct",
            device="auto"
        )
        
        # 초기화
        logger.info("🚀 PHI-3.5 초기화 중...")
        success = await phi35_ai.initialize()
        
        if not success:
            logger.error("❌ PHI-3.5 초기화 실패")
            return False
        
        logger.info("✅ PHI-3.5 초기화 성공")
        
        # 간단한 텍스트 생성 테스트
        test_prompts = [
            "Hello, how are you?",
            "What is 2+2?",
            "Tell me about robotics"
        ]
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\n🧪 테스트 {i+1}: '{prompt}'")
            
            try:
                response = await phi35_ai.model_manager.generate_response(
                    prompt, 
                    max_new_tokens=50,
                    temperature=0.7
                )
                
                logger.info(f"✅ 응답 생성 성공:")
                logger.info(f"   응답: {response[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ 응답 생성 실패: {e}")
                
        logger.info("\n🎉 DynamicCache 수정 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_cache_fix())