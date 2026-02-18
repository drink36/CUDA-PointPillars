FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
    libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    libgl1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel
RUN pip install "numpy<2"

RUN pip install \
    easydict pyyaml tqdm \
    tensorboardX numba llvmlite \
    scikit-image \
    opencv-python \
    onnx==1.16.1 onnxsim onnxruntime \
    onnx_graphsurgeon

# spconv（PointPillars 需要）
RUN pip install spconv-cu118

WORKDIR /workspace
CMD ["/bin/bash"]
