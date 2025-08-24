"""
향상된 웹 기반 대화형 학습 인터페이스

Flask + SocketIO를 사용한 실시간 대화형 동작학습 웹 인터페이스입니다.
음성 인식, TTS, 실시간 피드백, 시각적 학습 진행도 등을 제공합니다.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import numpy as np
import base64

# 프로젝트 모듈
from main import PhysicalAI
from realtime_feedback_system import (
    RealTimeFeedbackProcessor, 
    FeedbackType, 
    FeedbackSource,
    LearningAdjustment
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced_learning_interface_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 전역 상태
system_state = {
    'physical_ai': None,
    'feedback_processor': None,
    'learning_sessions': {},
    'active_users': {},
    'system_initialized': False,
    'current_demo_position': {'x': 400, 'y': 300},
    'robot_trajectory': [],
    'skill_progress': {
        'basic_movement': 0.92,
        'object_recognition': 0.72,
        'simple_grasp': 0.50,
        'precise_manipulation': 0.20,
        'collaborative_task': 0.10
    }
}

class LearningSession:
    """학습 세션 클래스"""
    def __init__(self, session_id: str, user_id: str, user_name: str):
        self.session_id = session_id
        self.user_id = user_id
        self.user_name = user_name
        self.started_at = datetime.now()
        self.interactions = []
        self.current_skill = "basic_movement"
        self.learning_progress = {}
        self.feedback_count = 0
        self.success_count = 0

@app.route('/')
def index():
    """메인 대화형 학습 페이지"""
    return render_template('enhanced_learning.html')

@app.route('/api/system/status')
def api_system_status():
    """시스템 상태 API"""
    return jsonify({
        'initialized': system_state['system_initialized'],
        'active_sessions': len(system_state['learning_sessions']),
        'active_users': len(system_state['active_users']),
        'skill_progress': system_state['skill_progress'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/system/initialize', methods=['POST'])
def api_initialize_system():
    """시스템 초기화 API"""
    def initialize():
        try:
            logger.info("시스템 초기화 시작")
            
            # Physical AI 시스템 초기화
            system_state['physical_ai'] = PhysicalAI("configs/default.yaml")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(system_state['physical_ai'].initialize())
            loop.close()
            
            # 피드백 시스템 초기화
            system_state['feedback_processor'] = RealTimeFeedbackProcessor(
                system_state['physical_ai']
            )
            system_state['feedback_processor'].start_processing()
            
            # 피드백 콜백 설정
            setup_feedback_callbacks()
            
            system_state['system_initialized'] = True
            
            socketio.emit('system_initialized', {
                'status': 'success',
                'message': '시스템 초기화 완료',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("시스템 초기화 완료")
            
        except Exception as e:
            error_msg = f"시스템 초기화 실패: {e}"
            logger.error(error_msg)
            socketio.emit('system_error', {
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
    
    threading.Thread(target=initialize, daemon=True).start()
    return jsonify({'status': 'started', 'message': '시스템 초기화 시작됨'})

@app.route('/api/chat/send', methods=['POST'])
def api_send_chat():
    """채팅 메시지 전송 API"""
    data = request.get_json()
    user_id = data.get('user_id', 'anonymous')
    message = data.get('message', '')
    message_type = data.get('type', 'text')  # text, voice, command
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    # 메시지 처리
    threading.Thread(
        target=process_user_message, 
        args=(user_id, message, message_type),
        daemon=True
    ).start()
    
    return jsonify({'status': 'processing', 'message': '메시지 처리 중'})

@app.route('/api/feedback/send', methods=['POST'])
def api_send_feedback():
    """피드백 전송 API"""
    data = request.get_json()
    user_id = data.get('user_id', 'anonymous')
    feedback_type = data.get('feedback_type', 'positive')
    content = data.get('content', '')
    confidence = data.get('confidence', 1.0)
    context = data.get('context', {})
    
    if not system_state['feedback_processor']:
        return jsonify({'error': 'Feedback system not initialized'}), 400
    
    # 피드백 타입 변환
    fb_type_map = {
        'positive': FeedbackType.POSITIVE,
        'negative': FeedbackType.NEGATIVE,
        'corrective': FeedbackType.CORRECTIVE,
        'guidance': FeedbackType.GUIDANCE,
        'emergency': FeedbackType.EMERGENCY_STOP
    }
    
    fb_type = fb_type_map.get(feedback_type, FeedbackType.POSITIVE)
    
    # 피드백 추가
    system_state['feedback_processor'].add_feedback(
        feedback_type=fb_type,
        source=FeedbackSource.USER_TEXT,
        content=content,
        confidence=confidence,
        context=context
    )
    
    return jsonify({'status': 'success', 'message': '피드백 전송 완료'})

@app.route('/api/robot/move', methods=['POST'])
def api_robot_move():
    """로봇 이동 API"""
    data = request.get_json()
    x = data.get('x', 400)
    y = data.get('y', 300)
    
    # 이동 시뮬레이션
    animate_robot_movement(x, y)
    
    return jsonify({'status': 'success', 'position': {'x': x, 'y': y}})

@app.route('/api/learning/session/start', methods=['POST'])
def api_start_learning_session():
    """학습 세션 시작 API"""
    data = request.get_json()
    user_id = data.get('user_id', 'anonymous')
    user_name = data.get('user_name', 'User')
    
    session_id = str(uuid.uuid4())[:8]
    session = LearningSession(session_id, user_id, user_name)
    
    system_state['learning_sessions'][session_id] = session
    system_state['active_users'][user_id] = session_id
    
    socketio.emit('learning_session_started', {
        'session_id': session_id,
        'user_name': user_name,
        'timestamp': datetime.now().isoformat()
    }, room=user_id)
    
    return jsonify({
        'status': 'success', 
        'session_id': session_id,
        'message': '학습 세션 시작됨'
    })

@app.route('/api/analytics/feedback')
def api_feedback_analytics():
    """피드백 분석 데이터 API"""
    if not system_state['feedback_processor']:
        return jsonify({'error': 'Feedback system not initialized'}), 400
    
    analytics = system_state['feedback_processor'].get_feedback_analytics()
    return jsonify(analytics)

# Socket.IO 이벤트 핸들러
@socketio.on('connect')
def handle_connect():
    """클라이언트 연결"""
    user_id = request.sid
    join_room(user_id)
    
    logger.info(f"사용자 연결: {user_id}")
    emit('connected', {
        'user_id': user_id,
        'system_status': system_state['system_initialized'],
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제"""
    user_id = request.sid
    leave_room(user_id)
    
    # 활성 세션 정리
    if user_id in system_state['active_users']:
        session_id = system_state['active_users'][user_id]
        if session_id in system_state['learning_sessions']:
            del system_state['learning_sessions'][session_id]
        del system_state['active_users'][user_id]
    
    logger.info(f"사용자 연결 해제: {user_id}")

@socketio.on('join_learning')
def handle_join_learning(data):
    """학습 세션 참가"""
    user_id = request.sid
    user_name = data.get('user_name', 'User')
    
    emit('learning_ready', {
        'user_id': user_id,
        'user_name': user_name,
        'skill_progress': system_state['skill_progress']
    })

@socketio.on('voice_data')
def handle_voice_data(data):
    """음성 데이터 처리"""
    user_id = request.sid
    audio_data = data.get('audio', '')
    
    # 음성 인식 처리 (실제로는 더 복잡한 처리 필요)
    # 여기서는 간단히 텍스트로 변환되었다고 가정
    recognized_text = "음성으로 입력된 텍스트"  # 실제 음성 인식 결과
    
    emit('voice_recognized', {
        'text': recognized_text,
        'confidence': 0.85
    })
    
    # 메시지 처리
    process_user_message(user_id, recognized_text, 'voice')

# 헬퍼 함수들
def setup_feedback_callbacks():
    """피드백 콜백 설정"""
    def feedback_callback(event_type, data):
        if event_type == "response":
            # AI 응답을 모든 클라이언트에 전송
            socketio.emit('ai_response', {
                'response': data['response'],
                'timestamp': data['timestamp']
            })
        else:
            # 피드백 처리 완료 알림
            socketio.emit('feedback_processed', {
                'feedback_type': data.feedback_type.value,
                'processed_at': datetime.now().isoformat()
            })
    
    def adjustment_callback(adjustment: LearningAdjustment):
        # 학습 조정 알림
        socketio.emit('learning_adjusted', {
            'skill': adjustment.skill_name,
            'parameter': adjustment.parameter,
            'old_value': adjustment.old_value,
            'new_value': adjustment.new_value,
            'reason': adjustment.reason,
            'timestamp': adjustment.applied_at.isoformat()
        })
    
    if system_state['feedback_processor']:
        system_state['feedback_processor'].register_feedback_callback(feedback_callback)
        system_state['feedback_processor'].register_adjustment_callback(adjustment_callback)

def process_user_message(user_id: str, message: str, message_type: str):
    """사용자 메시지 처리"""
    try:
        # AI 응답 생성 시뮬레이션
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        ai_response = loop.run_until_complete(generate_ai_response(message, message_type))
        loop.close()
        
        # 응답 전송
        socketio.emit('ai_response', {
            'response': ai_response,
            'original_message': message,
            'message_type': message_type,
            'timestamp': datetime.now().isoformat()
        }, room=user_id)
        
        # 동작 시뮬레이션
        simulate_robot_action(message)
        
    except Exception as e:
        logger.error(f"메시지 처리 오류: {e}")
        socketio.emit('error', {
            'message': f'메시지 처리 중 오류 발생: {e}',
            'timestamp': datetime.now().isoformat()
        }, room=user_id)

async def generate_ai_response(message: str, message_type: str) -> str:
    """AI 응답 생성"""
    try:
        if not system_state['physical_ai']:
            return "시스템이 아직 초기화되지 않았습니다."
        
        # 컨텍스트 생성
        context = f"""당신은 친근한 로봇 학습 도우미입니다.

현재 상황:
- 입력 방식: {message_type}
- 활성 학습 세션: {len(system_state['learning_sessions'])}개
- 시스템 상태: {'준비됨' if system_state['system_initialized'] else '초기화 중'}

사용자의 메시지에 대해:
1. 자연스럽고 친근한 톤으로 응답하세요
2. 학습과 관련된 내용이면 구체적인 안내를 제공하세요
3. 로봇의 동작이 필요하면 어떤 동작을 할지 설명하세요
4. 격려와 긍정적인 피드백을 포함하세요

사용자 메시지: {message}"""
        
        # PHI-3.5 응답 생성
        if (system_state['physical_ai'] and 
            hasattr(system_state['physical_ai'], 'slm_foundation') and 
            system_state['physical_ai'].slm_foundation.phi35_ai):
            
            response = await system_state['physical_ai'].slm_foundation.phi35_ai.model_manager.generate_response(
                context, max_new_tokens=200, temperature=0.8
            )
            return response
        else:
            # 폴백 응답
            fallback_responses = {
                "안녕": "안녕하세요! 오늘 어떤 동작을 학습해보실까요?",
                "움직": "좋습니다! 어디로 이동할까요? 화면을 클릭해서 목표 지점을 알려주세요.",
                "잘했": "감사합니다! 계속해서 더 나은 동작을 학습하겠습니다.",
                "천천히": "알겠습니다. 속도를 줄여서 더 신중하게 동작하겠습니다."
            }
            
            for key, response in fallback_responses.items():
                if key in message:
                    return response
            
            return "말씀해주셔서 감사합니다. 더 자세히 설명해주시면 도움이 될 것 같아요!"
            
    except Exception as e:
        logger.error(f"AI 응답 생성 실패: {e}")
        return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {e}"

def simulate_robot_action(message: str):
    """로봇 동작 시뮬레이션"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["이동", "움직", "가다", "move"]):
        # 랜덤 이동
        import random
        new_x = random.randint(100, 700)
        new_y = random.randint(100, 500)
        animate_robot_movement(new_x, new_y)
        
    elif any(word in message_lower for word in ["잡다", "집다", "grasp"]):
        # 잡기 동작
        simulate_grasp_action()
        
    elif any(word in message_lower for word in ["돌다", "회전", "rotate"]):
        # 회전 동작
        simulate_rotation_action()

def animate_robot_movement(target_x: int, target_y: int):
    """로봇 이동 애니메이션"""
    current_pos = system_state['current_demo_position']
    
    def move_step(step, total_steps=20):
        if step <= total_steps:
            progress = step / total_steps
            new_x = current_pos['x'] + (target_x - current_pos['x']) * progress
            new_y = current_pos['y'] + (target_y - current_pos['y']) * progress
            
            # 위치 업데이트
            system_state['current_demo_position'] = {'x': int(new_x), 'y': int(new_y)}
            system_state['robot_trajectory'].append({'x': int(new_x), 'y': int(new_y), 'timestamp': time.time()})
            
            # 클라이언트에 위치 전송
            socketio.emit('robot_position_update', {
                'position': {'x': int(new_x), 'y': int(new_y)},
                'progress': progress,
                'timestamp': datetime.now().isoformat()
            })
            
            # 다음 스텝 예약
            threading.Timer(0.1, lambda: move_step(step + 1, total_steps)).start()
        else:
            # 이동 완료
            socketio.emit('robot_action_complete', {
                'action': 'move',
                'final_position': {'x': target_x, 'y': target_y},
                'timestamp': datetime.now().isoformat()
            })
    
    move_step(1)

def simulate_grasp_action():
    """잡기 동작 시뮬레이션"""
    pos = system_state['current_demo_position']
    
    # 잡기 애니메이션 시뮬레이션
    for i in range(5):
        socketio.emit('robot_grasp_animation', {
            'position': pos,
            'phase': i,
            'total_phases': 5,
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(0.2)
    
    socketio.emit('robot_action_complete', {
        'action': 'grasp',
        'position': pos,
        'timestamp': datetime.now().isoformat()
    })

def simulate_rotation_action():
    """회전 동작 시뮬레이션"""
    pos = system_state['current_demo_position']
    
    # 회전 애니메이션
    for angle in range(0, 360, 30):
        socketio.emit('robot_rotation_update', {
            'position': pos,
            'angle': angle,
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(0.1)
    
    socketio.emit('robot_action_complete', {
        'action': 'rotate',
        'position': pos,
        'timestamp': datetime.now().isoformat()
    })

def update_skill_progress():
    """스킬 진행도 주기적 업데이트"""
    while True:
        try:
            if system_state['physical_ai'] and system_state['system_initialized']:
                # 실제 스킬 데이터 업데이트
                dev_engine = system_state['physical_ai'].dev_engine
                if hasattr(dev_engine, 'skill_engine') and hasattr(dev_engine.skill_engine, 'skills_db'):
                    for skill_name, skill in dev_engine.skill_engine.skills_db.items():
                        if skill_name in system_state['skill_progress']:
                            system_state['skill_progress'][skill_name] = skill.success_rate
                
                # 클라이언트에 업데이트 전송
                socketio.emit('skill_progress_update', {
                    'skills': system_state['skill_progress'],
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"스킬 진행도 업데이트 오류: {e}")
        
        time.sleep(5)  # 5초마다 업데이트

# 웹 템플릿 생성
def create_enhanced_learning_template():
    """향상된 학습 인터페이스 HTML 템플릿 생성"""
    template_dir = Path("web_interface/templates")
    template_dir.mkdir(exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 대화형 동작학습 시스템</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            grid-template-rows: 60px 1fr;
            height: 100vh;
            gap: 10px;
            padding: 10px;
        }
        
        .header {
            grid-column: 1 / -1;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .title {
            font-size: 24px;
            font-weight: bold;
            color: #4a5568;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #e53e3e;
            transition: background 0.3s;
        }
        
        .status-dot.connected {
            background: #38a169;
        }
        
        .chat-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .chat-header {
            padding: 15px 20px;
            background: #4299e1;
            color: white;
            font-weight: bold;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f7fafc;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .message.user {
            background: #4299e1;
            color: white;
            margin-left: auto;
        }
        
        .message.ai {
            background: #48bb78;
            color: white;
        }
        
        .message.system {
            background: #ed8936;
            color: white;
            text-align: center;
            max-width: 100%;
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e2e8f0;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        .input-field {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
        }
        
        .input-field:focus {
            outline: none;
            border-color: #4299e1;
        }
        
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #4299e1;
            color: white;
        }
        
        .btn-primary:hover {
            background: #3182ce;
        }
        
        .btn-voice {
            background: #e53e3e;
            color: white;
        }
        
        .btn-voice.listening {
            background: #38a169;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .simulation-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .simulation-header {
            padding: 15px 20px;
            background: #48bb78;
            color: white;
            font-weight: bold;
        }
        
        .simulation-canvas {
            flex: 1;
            position: relative;
            background: #f0f8ff;
            overflow: hidden;
        }
        
        .robot {
            position: absolute;
            width: 40px;
            height: 40px;
            background: #4299e1;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 20px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(66, 153, 225, 0.4);
        }
        
        .target {
            position: absolute;
            width: 30px;
            height: 30px;
            background: #48bb78;
            border-radius: 50%;
            opacity: 0.7;
        }
        
        .trajectory-point {
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(66, 153, 225, 0.6);
            border-radius: 50%;
        }
        
        .control-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            overflow-y: auto;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .panel-section {
            margin-bottom: 25px;
        }
        
        .panel-title {
            font-weight: bold;
            color: #4a5568;
            margin-bottom: 15px;
            font-size: 16px;
        }
        
        .skill-item {
            margin-bottom: 15px;
        }
        
        .skill-name {
            font-size: 14px;
            color: #4a5568;
            margin-bottom: 5px;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4299e1, #48bb78);
            transition: width 0.5s ease;
        }
        
        .feedback-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .btn-feedback {
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-positive {
            background: #48bb78;
            color: white;
        }
        
        .btn-negative {
            background: #e53e3e;
            color: white;
        }
        
        .btn-corrective {
            background: #ed8936;
            color: white;
        }
        
        .btn-guidance {
            background: #805ad5;
            color: white;
        }
        
        .system-info {
            font-size: 12px;
            color: #666;
            line-height: 1.4;
        }
        
        .learning-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .stat-item {
            text-align: center;
            padding: 10px;
            background: #f7fafc;
            border-radius: 8px;
        }
        
        .stat-value {
            font-size: 20px;
            font-weight: bold;
            color: #4299e1;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="title">🎯 대화형 동작학습 시스템</div>
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">연결 중...</span>
            </div>
        </div>
        
        <!-- Chat Panel -->
        <div class="chat-panel">
            <div class="chat-header">💬 AI와 대화</div>
            <div class="chat-messages" id="chatMessages"></div>
            <div class="chat-input">
                <div class="input-group">
                    <input type="text" class="input-field" id="messageInput" placeholder="메시지를 입력하세요...">
                    <button class="btn btn-voice" id="voiceBtn">🎤</button>
                    <button class="btn btn-primary" id="sendBtn">전송</button>
                </div>
            </div>
        </div>
        
        <!-- Simulation Panel -->
        <div class="simulation-panel">
            <div class="simulation-header">👁️ 실시간 시뮬레이션</div>
            <div class="simulation-canvas" id="simulationCanvas">
                <div class="robot" id="robot" style="left: 380px; top: 280px;">🤖</div>
            </div>
        </div>
        
        <!-- Control Panel -->
        <div class="control-panel">
            <!-- Learning Stats -->
            <div class="panel-section">
                <div class="panel-title">📊 학습 통계</div>
                <div class="learning-stats">
                    <div class="stat-item">
                        <div class="stat-value" id="totalInteractions">0</div>
                        <div class="stat-label">총 상호작용</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="successRate">0%</div>
                        <div class="stat-label">성공률</div>
                    </div>
                </div>
            </div>
            
            <!-- Skill Progress -->
            <div class="panel-section">
                <div class="panel-title">🎯 스킬 진행도</div>
                <div id="skillProgress">
                    <!-- Skills will be populated by JavaScript -->
                </div>
            </div>
            
            <!-- Quick Feedback -->
            <div class="panel-section">
                <div class="panel-title">⚡ 빠른 피드백</div>
                <div class="feedback-buttons">
                    <button class="btn-feedback btn-positive" onclick="sendFeedback('positive', '좋습니다!')">👍 좋음</button>
                    <button class="btn-feedback btn-negative" onclick="sendFeedback('negative', '다시 시도')">👎 개선 필요</button>
                    <button class="btn-feedback btn-corrective" onclick="sendFeedback('corrective', '수정 필요')">🔧 수정</button>
                    <button class="btn-feedback btn-guidance" onclick="sendFeedback('guidance', '안내 필요')">📖 안내</button>
                </div>
            </div>
            
            <!-- System Control -->
            <div class="panel-section">
                <div class="panel-title">⚙️ 시스템 제어</div>
                <button class="btn btn-primary" onclick="initializeSystem()" style="width: 100%; margin-bottom: 10px;">시스템 초기화</button>
                <button class="btn btn-primary" onclick="startLearningSession()" style="width: 100%;">학습 세션 시작</button>
            </div>
            
            <!-- System Info -->
            <div class="panel-section">
                <div class="panel-title">ℹ️ 시스템 정보</div>
                <div class="system-info" id="systemInfo">
                    시스템 상태: 대기 중<br>
                    활성 세션: 0개<br>
                    마지막 업데이트: -
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Socket.IO 연결
        const socket = io();
        
        // 전역 변수
        let isListening = false;
        let userId = null;
        let currentSession = null;
        let totalInteractions = 0;
        let successfulInteractions = 0;
        
        // DOM 요소
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const voiceBtn = document.getElementById('voiceBtn');
        const sendBtn = document.getElementById('sendBtn');
        const robot = document.getElementById('robot');
        const simulationCanvas = document.getElementById('simulationCanvas');
        const skillProgressContainer = document.getElementById('skillProgress');
        const systemInfo = document.getElementById('systemInfo');
        
        // Socket 이벤트 핸들러
        socket.on('connected', (data) => {
            userId = data.user_id;
            statusDot.classList.add('connected');
            statusText.textContent = '연결됨';
            addMessage('system', '시스템에 연결되었습니다.');
            updateSystemInfo(data);
        });
        
        socket.on('ai_response', (data) => {
            addMessage('ai', data.response);
            totalInteractions++;
            updateStats();
        });
        
        socket.on('robot_position_update', (data) => {
            const pos = data.position;
            robot.style.left = pos.x - 20 + 'px';
            robot.style.top = pos.y - 20 + 'px';
            
            // 궤적 점 추가
            addTrajectoryPoint(pos.x, pos.y);
        });
        
        socket.on('robot_action_complete', (data) => {
            addMessage('system', `동작 완료: ${data.action}`);
            successfulInteractions++;
            updateStats();
        });
        
        socket.on('skill_progress_update', (data) => {
            updateSkillProgress(data.skills);
        });
        
        socket.on('learning_adjusted', (data) => {
            addMessage('system', `학습 조정: ${data.skill} - ${data.parameter} (${data.reason})`);
        });
        
        socket.on('system_initialized', (data) => {
            addMessage('system', '✅ 시스템 초기화 완료');
            statusText.textContent = '시스템 준비됨';
        });
        
        socket.on('system_error', (data) => {
            addMessage('system', `❌ 오류: ${data.error}`);
        });
        
        // 이벤트 리스너
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        sendBtn.addEventListener('click', sendMessage);
        voiceBtn.addEventListener('click', toggleVoiceInput);
        
        // 시뮬레이션 캔버스 클릭 이벤트
        simulationCanvas.addEventListener('click', (e) => {
            const rect = simulationCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // 로봇 이동 요청
            fetch('/api/robot/move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({x: x, y: y})
            });
            
            addMessage('user', `위치 (${Math.round(x)}, ${Math.round(y)})로 이동 요청`);
        });
        
        // 함수들
        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            messageInput.value = '';
            
            // 서버에 메시지 전송
            fetch('/api/chat/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: userId,
                    message: message,
                    type: 'text'
                })
            });
        }
        
        function toggleVoiceInput() {
            if (!isListening) {
                startVoiceRecognition();
            } else {
                stopVoiceRecognition();
            }
        }
        
        function startVoiceRecognition() {
            if (!('webkitSpeechRecognition' in window)) {
                addMessage('system', '음성 인식을 지원하지 않는 브라우저입니다.');
                return;
            }
            
            const recognition = new webkitSpeechRecognition();
            recognition.lang = 'ko-KR';
            recognition.continuous = false;
            recognition.interimResults = false;
            
            recognition.onstart = () => {
                isListening = true;
                voiceBtn.classList.add('listening');
                voiceBtn.textContent = '🛑';
                addMessage('system', '음성 인식 시작...');
            };
            
            recognition.onresult = (event) => {
                const result = event.results[0][0].transcript;
                messageInput.value = result;
                addMessage('user', `[음성] ${result}`);
                
                // 자동으로 메시지 전송
                setTimeout(() => {
                    sendMessage();
                }, 500);
            };
            
            recognition.onerror = (event) => {
                addMessage('system', `음성 인식 오류: ${event.error}`);
                stopVoiceRecognition();
            };
            
            recognition.onend = () => {
                stopVoiceRecognition();
            };
            
            recognition.start();
        }
        
        function stopVoiceRecognition() {
            isListening = false;
            voiceBtn.classList.remove('listening');
            voiceBtn.textContent = '🎤';
        }
        
        function sendFeedback(type, content) {
            fetch('/api/feedback/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: userId,
                    feedback_type: type,
                    content: content,
                    confidence: 1.0,
                    context: {
                        current_skill: 'basic_movement',
                        action: 'user_feedback'
                    }
                })
            });
            
            addMessage('user', `[피드백] ${content}`);
        }
        
        function initializeSystem() {
            fetch('/api/system/initialize', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    addMessage('system', '시스템 초기화 시작...');
                });
        }
        
        function startLearningSession() {
            const userName = prompt('사용자 이름을 입력하세요:', 'User');
            if (!userName) return;
            
            fetch('/api/learning/session/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: userId,
                    user_name: userName
                })
            })
            .then(response => response.json())
            .then(data => {
                currentSession = data.session_id;
                addMessage('system', `학습 세션 시작: ${currentSession}`);
            });
        }
        
        function addMessage(type, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            const timestamp = new Date().toLocaleTimeString();
            messageDiv.innerHTML = `${content}<div style="font-size: 10px; opacity: 0.7; margin-top: 5px;">${timestamp}</div>`;
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function addTrajectoryPoint(x, y) {
            const point = document.createElement('div');
            point.className = 'trajectory-point';
            point.style.left = x - 2 + 'px';
            point.style.top = y - 2 + 'px';
            
            simulationCanvas.appendChild(point);
            
            // 오래된 점들 제거 (최대 50개)
            const points = simulationCanvas.querySelectorAll('.trajectory-point');
            if (points.length > 50) {
                points[0].remove();
            }
        }
        
        function updateSkillProgress(skills) {
            skillProgressContainer.innerHTML = '';
            
            for (const [skillName, progress] of Object.entries(skills)) {
                const skillDiv = document.createElement('div');
                skillDiv.className = 'skill-item';
                
                const percentage = Math.round(progress * 100);
                const displayName = skillName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                
                skillDiv.innerHTML = `
                    <div class="skill-name">${displayName}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${percentage}%"></div>
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 2px;">${percentage}%</div>
                `;
                
                skillProgressContainer.appendChild(skillDiv);
            }
        }
        
        function updateStats() {
            document.getElementById('totalInteractions').textContent = totalInteractions;
            const successRate = totalInteractions > 0 ? Math.round((successfulInteractions / totalInteractions) * 100) : 0;
            document.getElementById('successRate').textContent = successRate + '%';
        }
        
        function updateSystemInfo(data) {
            const info = `
                시스템 상태: ${data.system_status ? '준비됨' : '초기화 중'}<br>
                사용자 ID: ${userId ? userId.substring(0, 8) + '...' : '-'}<br>
                세션: ${currentSession || '없음'}<br>
                마지막 업데이트: ${new Date().toLocaleTimeString()}
            `;
            systemInfo.innerHTML = info;
        }
        
        // 초기화
        window.addEventListener('load', () => {
            // 기본 스킬 진행도 표시
            updateSkillProgress({
                'basic_movement': 0.92,
                'object_recognition': 0.72,
                'simple_grasp': 0.50,
                'precise_manipulation': 0.20
            });
            
            // 시스템 상태 주기적 업데이트
            setInterval(() => {
                updateSystemInfo({system_status: statusDot.classList.contains('connected')});
            }, 5000);
        });
    </script>
</body>
</html>
    """
    
    with open(template_dir / "enhanced_learning.html", "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == '__main__':
    # 템플릿 생성
    create_enhanced_learning_template()
    
    # 스킬 진행도 업데이트 스레드 시작
    update_thread = threading.Thread(target=update_skill_progress, daemon=True)
    update_thread.start()
    
    print("🎯 향상된 대화형 학습 인터페이스 시작 중...")
    print("📱 http://localhost:5001 에서 접속하세요")
    print("🎤 음성 인식, 실시간 피드백, 시각적 학습 진행도 지원")
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)