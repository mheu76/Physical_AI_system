"""
Physical AI Web Interface - 메인 애플리케이션

Physical AI 시스템을 모니터링하고 제어할 수 있는 웹 인터페이스입니다.
"""

import asyncio
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import PhysicalAI
from utils.common import setup_logging, load_config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'physical_ai_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Physical AI 시스템 인스턴스
physical_ai = None
system_status = {
    'initialized': False,
    'current_mission': None,
    'learning_active': False,
    'safety_status': 'safe',
    'last_update': datetime.now().isoformat()
}

# 스킬 데이터 (실시간 업데이트)
skills_data = {
    'basic_movement': {'success_rate': 0.92, 'practice_count': 0},
    'object_recognition': {'success_rate': 0.72, 'practice_count': 0},
    'simple_grasp': {'success_rate': 0.50, 'practice_count': 0},
    'precise_manipulation': {'success_rate': 0.20, 'practice_count': 0},
    'collaborative_task': {'success_rate': 0.10, 'practice_count': 0}
}

# 미션 히스토리
mission_history = []

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('dashboard.html', 
                         system_status=system_status,
                         skills_data=skills_data,
                         mission_history=mission_history[-10:])  # 최근 10개만

@app.route('/mission')
def mission_page():
    """미션 제어 페이지"""
    return render_template('mission.html', system_status=system_status)

@app.route('/learning')
def learning_page():
    """학습 모니터링 페이지"""
    return render_template('learning.html', skills_data=skills_data)

@app.route('/system')
def system_page():
    """시스템 상태 페이지"""
    return render_template('system.html', system_status=system_status)

@app.route('/api/status')
def api_status():
    """시스템 상태 API"""
    return jsonify({
        'status': 'success',
        'data': system_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/skills')
def api_skills():
    """스킬 데이터 API"""
    return jsonify({
        'status': 'success',
        'data': skills_data,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/missions', methods=['GET', 'POST'])
def api_missions():
    """미션 API"""
    global mission_history
    
    if request.method == 'POST':
        data = request.get_json()
        mission_text = data.get('mission', '')
        
        if not mission_text:
            return jsonify({'status': 'error', 'message': '미션 텍스트가 필요합니다'}), 400
        
        # 미션 실행 (비동기)
        def execute_mission():
            try:
                if physical_ai and system_status['initialized']:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(physical_ai.execute_mission(mission_text))
                    
                    # 결과를 미션 히스토리에 추가
                    mission_record = {
                        'id': len(mission_history) + 1,
                        'mission': mission_text,
                        'status': 'success' if result.success else 'failed',
                        'execution_time': result.execution_time,
                        'timestamp': datetime.now().isoformat(),
                        'actions_count': len(result.actions_performed),
                        'learning_value': result.learning_value
                    }
                    mission_history.append(mission_record)
                    
                    # 웹소켓으로 실시간 업데이트 전송
                    socketio.emit('mission_completed', mission_record)
                    
                    loop.close()
                    
            except Exception as e:
                print(f"미션 실행 오류: {e}")
                socketio.emit('mission_error', {'error': str(e)})
        
        # 별도 스레드에서 미션 실행
        thread = threading.Thread(target=execute_mission)
        thread.start()
        
        return jsonify({'status': 'success', 'message': '미션이 시작되었습니다'})
    
    else:
        return jsonify({
            'status': 'success',
            'data': mission_history,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/initialize', methods=['POST'])
def api_initialize():
    """시스템 초기화 API"""
    global physical_ai, system_status
    
    def initialize_system():
        try:
            global physical_ai
            physical_ai = PhysicalAI("configs/default.yaml")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(physical_ai.initialize())
            loop.close()
            
            system_status['initialized'] = True
            system_status['last_update'] = datetime.now().isoformat()
            
            socketio.emit('system_initialized', {'status': 'success'})
            
        except Exception as e:
            print(f"시스템 초기화 오류: {e}")
            socketio.emit('system_error', {'error': str(e)})
    
    thread = threading.Thread(target=initialize_system)
    thread.start()
    
    return jsonify({'status': 'success', 'message': '시스템 초기화가 시작되었습니다'})

@app.route('/api/learning/start', methods=['POST'])
def api_start_learning():
    """자율 학습 시작 API"""
    global system_status
    
    def start_learning():
        try:
            if physical_ai and system_status['initialized']:
                system_status['learning_active'] = True
                socketio.emit('learning_started', {'status': 'active'})
                
                # 자율 학습 루프
                while system_status['learning_active']:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(physical_ai.developmental_learning_cycle())
                    loop.close()
                    time.sleep(30)  # 30초 대기
                    
        except Exception as e:
            print(f"학습 오류: {e}")
            system_status['learning_active'] = False
            socketio.emit('learning_error', {'error': str(e)})
    
    thread = threading.Thread(target=start_learning)
    thread.start()
    
    return jsonify({'status': 'success', 'message': '자율 학습이 시작되었습니다'})

@app.route('/api/learning/stop', methods=['POST'])
def api_stop_learning():
    """자율 학습 중지 API"""
    global system_status
    system_status['learning_active'] = False
    socketio.emit('learning_stopped', {'status': 'inactive'})
    
    return jsonify({'status': 'success', 'message': '자율 학습이 중지되었습니다'})

@socketio.on('connect')
def handle_connect():
    """클라이언트 연결 처리"""
    print('클라이언트가 연결되었습니다')
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제 처리"""
    print('클라이언트가 연결 해제되었습니다')

def update_skills_data():
    """스킬 데이터 업데이트 (주기적)"""
    global skills_data
    while True:
        try:
            if physical_ai and system_status['initialized']:
                # 실제 스킬 데이터 가져오기
                dev_engine = physical_ai.dev_engine
                if hasattr(dev_engine, 'skill_engine') and hasattr(dev_engine.skill_engine, 'skills_db'):
                    for skill_name, skill in dev_engine.skill_engine.skills_db.items():
                        if skill_name in skills_data:
                            skills_data[skill_name]['success_rate'] = skill.success_rate
                            skills_data[skill_name]['practice_count'] = skill.practice_count
                
                # 웹소켓으로 실시간 업데이트 전송
                socketio.emit('skills_updated', skills_data)
                
        except Exception as e:
            print(f"스킬 데이터 업데이트 오류: {e}")
        
        time.sleep(5)  # 5초마다 업데이트

if __name__ == '__main__':
    # 스킬 데이터 업데이트 스레드 시작
    update_thread = threading.Thread(target=update_skills_data, daemon=True)
    update_thread.start()
    
    print("Physical AI Web Interface 시작 중...")
    print("http://localhost:5000 에서 접속하세요")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
