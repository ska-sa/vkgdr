from typing import Generator

import numpy as np
import pycuda.compiler
import pycuda.driver
import pycuda.gpuarray
import pycuda.tools
import pytest

import vkgdr.pycuda

DOUBLE_SOURCE = """
    __global__ void double_it(unsigned char *data, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            data[idx] *= 2;
    }
"""


@pytest.fixture(scope="session", autouse=True)
def init_cuda() -> None:
    pycuda.driver.init()


@pytest.fixture()
def context() -> Generator[pycuda.driver.Context, None, None]:
    ctx = pycuda.tools.make_default_context()
    yield ctx
    ctx.pop()


def test_open_invalid_device() -> None:
    """Test Vkgdr constructor with an invalid device handle."""
    with pytest.raises(vkgdr.VkgdrError, match="CUDA_ERROR_INVALID_DEVICE"):
        vkgdr.Vkgdr(-1)


def test_no_current_context() -> None:
    """Test :meth:`Vkgdr.open_current_context()` with no current context."""
    with pytest.raises(vkgdr.VkgdrError, match="CUDA_ERROR_INVALID_CONTEXT"):
        vkgdr.Vkgdr.open_current_context()


@pytest.mark.parametrize("flags", [0, vkgdr.OpenFlags.REQUIRE_COHERENT_BIT, vkgdr.OpenFlags.FORCE_NON_COHERENT_BIT])
def test_basic(context: pycuda.driver.Context, flags: int) -> None:
    n = 4097
    g = vkgdr.Vkgdr.open_current_context(flags)
    mem = vkgdr.pycuda.Memory(g, n)
    assert len(mem) == n
    if flags == vkgdr.OpenFlags.FORCE_NON_COHERENT_BIT:
        assert not mem.is_coherent
    elif vkgdr.OpenFlags.REQUIRE_COHERENT_BIT:
        assert mem.is_coherent
    assert mem.non_coherent_atom_size > 0  # Just checks that the property can be retrieved

    array = np.asarray(mem).view(np.uint8)
    # Write some data to the memory, from the host
    in_data = np.random.default_rng(seed=1).integers(1, 128, (n,), np.uint8)
    array[:] = in_data
    mem.flush(0, n)

    # Process the data on the device
    module = pycuda.compiler.SourceModule(DOUBLE_SOURCE)
    double_it = module.get_function("double_it")
    d_mem = pycuda.gpuarray.GPUArray(n, np.uint8, gpudata=mem)
    double_it(d_mem, np.int32(n), block=(256, 1, 1), grid=(n // 256 + 1,))

    # Read the back the results and verify
    mem.invalidate(0, n)
    np.testing.assert_array_equal(in_data * 2, array)


def test_explicit_free(context: pycuda.driver.Context) -> None:
    n = 4097
    g = vkgdr.Vkgdr.open_current_context()
    mem = vkgdr.pycuda.Memory(g, n)
    mem.free()


def test_out_of_memory(context: pycuda.driver.Context) -> None:
    g = vkgdr.Vkgdr.open_current_context()
    with pytest.raises(vkgdr.VkgdrError, match="VK_ERROR_OUT_OF_DEVICE_MEMORY"):
        vkgdr.pycuda.Memory(g, 10**15)


def test_alloc_without_context(context: pycuda.driver.Context) -> None:
    g = vkgdr.Vkgdr.open_current_context()
    context.pop()
    try:
        with pytest.raises(vkgdr.VkgdrError):
            vkgdr.pycuda.Memory(g, 123)
    finally:
        context.push()  # To balance the pop in the fixture cleanup
