FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

RUN apt update && apt install -y \
    sudo vim build-essential python3 python3-pip python3-dev python3-tk \
    libgl1 libglib2.0-dev \
    cuda-compiler-12-9 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-xfixes0-dev \
    libxcb-image0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-render-util0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-shape0 \
    libxcb-xkb1 \
    libx11-xcb-dev \
    libglu1-mesa \
    libqt5widgets5 \
    libqt5gui5 \
    libqt5core5a \
    qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \ 
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

RUN pip3 install --break-system-packages --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
RUN pip3 install --break-system-packages ipykernel matplotlib ultralytics "onnx>=1.12.0,<1.18.0" onnxruntime-gpu onnxslim onnxscript "faster-coco-eval>=1.6.7" norfair

# user
RUN if id ubuntu >/dev/null 2>&1; then \
      usermod -l hostuser ubuntu && \
      usermod -d /home/hostuser -m hostuser && \
      groupmod -n hostuser ubuntu; \
    fi

ENV USERNAME=hostuser
USER $USERNAME
ENV USER=$USERNAME

ENV YOLO_CONFIG_DIR=/workspace/.config/Ultralytics
WORKDIR /workspace

CMD ["/bin/bash"]

