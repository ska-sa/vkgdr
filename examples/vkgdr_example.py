#!/usr/bin/env python3

################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray
from pycuda.compiler import SourceModule

import vkgdr.pycuda

source = """
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}
"""

module = SourceModule(source)
vec_add = module.get_function("VecAdd")

N = 1048576
dtype = np.dtype(np.float32)

g = vkgdr.Vkgdr.open_current_context()
a = vkgdr.pycuda.Memory(g, N * dtype.itemsize)
b = vkgdr.pycuda.Memory(g, N * dtype.itemsize)
c = vkgdr.pycuda.Memory(g, N * dtype.itemsize)
h_a = np.array(a, copy=False).view(dtype)
h_b = np.array(b, copy=False).view(dtype)
h_c = np.array(c, copy=False).view(dtype)
d_a = pycuda.gpuarray.GPUArray(N, dtype, gpudata=a)
d_b = pycuda.gpuarray.GPUArray(N, dtype, gpudata=b)
d_c = pycuda.gpuarray.GPUArray(N, dtype, gpudata=c)
h_a[:] = np.arange(0, N, dtype=dtype)
h_b[:] = 17
a.flush(0, len(a))
b.flush(0, len(b))

vec_add(d_a, d_b, d_c, block=(64, 1, 1), grid=(N // 64, 1, 1))
c.invalidate(0, len(c))
print(h_c[100000:][:50])
