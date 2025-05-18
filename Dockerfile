# TensorFlow 2.4.1은 CUDA 11.0과 cuDNN 8.0을 권장합니다.
# nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 이미지를 기반으로 합니다.
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

# 환경 변수 설정 (경고 수정: key=value 형식으로)
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    git \
    cmake \
    build-essential \
    # dlib 및 OpenCV 의존성
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    # Gradio 실행에 필요할 수 있는 패키지 (ffmpeg 등)
    ffmpeg \
    # 기타 유틸리티
    curl \
    wget \
    unzip && \
    # apt-get clean and rm -rf /var/lib/apt/lists/* should be together
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# python3.8을 기본 python3으로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10

# pip 업그레이드 및 setuptools 설치
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# requirements.txt 복사
COPY requirements.txt .

# requirements.txt 설치 (오류 발생 시 더 자세한 로그를 위해 --verbose 추가)
RUN pip3 install --no-cache-dir --verbose -r requirements.txt

# 애플리케이션 코드, 모델, 설정 파일 복사
COPY . .

# 모델 디렉토리 생성
RUN mkdir -p /app/models
RUN mkdir -p /tmp

# Gradio 포트 노출
EXPOSE 7860

# 애플리케이션 실행
CMD ["python3", "app.py"]
