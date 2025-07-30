################################################################################
# Copyright (c) 2025, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

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
