FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    git \
    cmake \
    build-essential \
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
    ffmpeg \
    curl \
    wget \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10

RUN python3 -m pip install --no-cache-dir "pip<22.0" setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --verbose -r requirements.txt

COPY . .

RUN mkdir -p /app/models
RUN mkdir -p /tmp

EXPOSE 7860

CMD ["python3", "app.py"]
