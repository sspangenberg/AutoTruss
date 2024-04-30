FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PIP_DEFAULT_TIMEOUT=100

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Base software utilities
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils gcc \
    wget curl vim git ca-certificates kmod \
    && DEBIAN_FRONTEND=noninteractive apt-get remove --yes --quiet --allow-change-held-packages libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

# Python installation and initial dependencies
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev python3-distutils
RUN apt-get install -y libprotobuf-dev protobuf-compiler cmake patchelf libosmesa6-dev libgl1-mesa-glx libglfw3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install mujoco
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

ENTRYPOINT ["python", "main.py"]
