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
    h_A = (float *) vkgdr_memory_get_host(Amem);
    h_B = (float *) vkgdr_memory_get_host(Bmem);
    h_C = (float *) vkgdr_memory_get_host(Cmem);
    d_A = (float *) vkgdr_memory_get_device(Amem);
    d_B = (float *) vkgdr_memory_get_device(Bmem);
    d_C = (float *) vkgdr_memory_get_device(Cmem);
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
