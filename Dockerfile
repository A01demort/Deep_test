
# TensorFlow 2.3.0은 CUDA 10.1과 cuDNN 7.6을 요구합니다.
# nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 이미지를 기반으로 합니다.
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONUNBUFFERED=1

# 시스템 패키지 설치
# Python 3.7 설치 (Ubuntu 18.04는 기본 Python3이 3.6일 수 있음)
RUN apt-get update && apt-get install -y --no-install-recommends     software-properties-common     && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update     && apt-get install -y     python3.7     python3.7-dev     python3.7-venv     python3-pip     git     cmake     build-essential     # dlib 및 OpenCV 의존성
    libopenblas-dev     liblapack-dev     libjpeg-dev     libpng-dev     libtiff-dev     libavcodec-dev     libavformat-dev     libswscale-dev     libv4l-dev     libxvidcore-dev     libx264-dev     libgtk-3-dev     # Gradio 실행에 필요할 수 있는 패키지 (ffmpeg 등)
    ffmpeg     # 기타 유틸리티
    curl     wget     unzip     && rm -rf /var/lib/apt/lists/*

# python3.7을 기본 python3으로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 10

# pip 업그레이드 및 setuptools 설치 (dlib 설치 시 필요할 수 있음)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 코드, 모델, 설정 파일 복사
COPY . .

# 모델 디렉토리 생성 (만약 COPY . . 에서 생성되지 않았다면)
RUN mkdir -p /app/models
RUN mkdir -p /tmp # Gradio 출력용 임시 디렉토리

# Gradio 포트 노출
EXPOSE 7860

# 애플리케이션 실행
CMD ["python3", "app.py"]
