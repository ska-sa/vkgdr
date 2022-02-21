#!/bin/sh
set -e

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum -y install cuda-cudart-devel-11-6 vulkan-devel
