/*******************************************************************************
 * Copyright (c) 2022, National Research Foundation (SARAO)
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy
 * of the License at
 *
 *   https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <cassert>
#include <iostream>
#include "vkgdr.h"

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    const int N = 64;
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;

    // Call cuda to initialise the primary context
    cudaMalloc((void **) &d_C, N * sizeof(float));
    cudaFree(d_C);

    vkgdr_t g = vkgdr_open(0, VKGDR_OPEN_CURRENT_CONTEXT_BIT);
    assert(g);
    vkgdr_memory_t Amem = vkgdr_memory_alloc(g, N * sizeof(float), 0);
    vkgdr_memory_t Bmem = vkgdr_memory_alloc(g, N * sizeof(float), 0);
    vkgdr_memory_t Cmem = vkgdr_memory_alloc(g, N * sizeof(float), 0);
    assert(Amem);
    assert(Bmem);
    assert(Cmem);
    h_A = (float *) vkgdr_memory_get_host_ptr(Amem);
    h_B = (float *) vkgdr_memory_get_host_ptr(Bmem);
    h_C = (float *) vkgdr_memory_get_host_ptr(Cmem);
    d_A = (float *) vkgdr_memory_get_device_ptr(Amem);
    d_B = (float *) vkgdr_memory_get_device_ptr(Bmem);
    d_C = (float *) vkgdr_memory_get_device_ptr(Cmem);
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 7 * i;
        h_B[i] = -3 * i;
        h_C[i] = 0;
    }
    vkgdr_memory_flush(Amem, 0, N * sizeof(float));
    vkgdr_memory_flush(Bmem, 0, N * sizeof(float));
    vkgdr_memory_flush(Cmem, 0, N * sizeof(float));

    VecAdd<<<1, N>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    vkgdr_memory_invalidate(Cmem, 0, N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        std::cout << h_C[i] << ' ';
    }
    std::cout << '\n';

    vkgdr_memory_free(Amem);
    vkgdr_memory_free(Bmem);
    vkgdr_memory_free(Cmem);
    vkgdr_close(g);
    return 0;
}
