"""
Hugging Face Spaces Entry Point

이 파일은 Hugging Face Spaces에서 앱을 실행하기 위한 
메인 엔트리포인트입니다.
"""

from demo import create_gradio_interface

if __name__ == "__main__":
    # Hugging Face Spaces용 데모 실행
    demo = create_gradio_interface()
    demo.launch()