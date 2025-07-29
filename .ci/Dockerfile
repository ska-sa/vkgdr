FROM nvidia/cuda:12.9.1-base-ubuntu24.04

# cuda-profiler-api, libcurand-dev are needed to install pycuda
# libegl1, libxext6 are needed to run Vulkan
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    build-essential \
    cuda-nvcc-12-9 \
    cuda-profiler-api-12-9 \
    libcurand-dev-12-9 \
    libegl1 \
    libvulkan-dev \
    libxext6 \
    python3 \
    python3-dev \
    python3-venv

# Enable Vulkan when the container is run by nvidia-container-toolkit
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
