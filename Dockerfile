# Physical AI System Docker Image
FROM python:3.10-slim

LABEL maintainer="Physical AI System Team"
LABEL description="Developmental Robotics & Embodied AI System"
LABEL version="1.0.0"

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 로그 디렉토리 생성
RUN mkdir -p logs

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 기본 포트 노출 (모니터링용)
EXPOSE 8000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import main; print('OK')" || exit 1

# 기본 실행 명령
CMD ["python", "main.py", "--config", "configs/default.yaml"]

# 사용법 예시:
# docker build -t physical-ai-system .
# docker run -it --rm physical-ai-system python examples/basic_example.py
# docker run -it --rm -v $(pwd)/logs:/app/logs physical-ai-system