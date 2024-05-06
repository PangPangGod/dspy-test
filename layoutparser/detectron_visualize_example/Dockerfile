# 사용할 베이스 이미지
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND noninteractive

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ca-certificates git wget sudo ninja-build software-properties-common \
    libgl1-mesa-glx

# Python 버전 충돌 해결 (Python 3.9 사용)
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3.9-dev && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

# Python pip 설치
ENV PATH="/usr/local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# torch와 torchvision 설치
RUN pip install torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# detectron2 커밋 버전으로 설치 (안정적인 커밋 버전 사용)
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02"
RUN pip install layoutparser[all]
RUN pip install --upgrade pymupdf

# 작업 디렉터리 설정
WORKDIR /app