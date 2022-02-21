Introduction
============
Vkgdr is a library to provide CUDA applications with direct access to write (or
read) GPU memory directly from the CPU, by mapping the GPU memory into the
CPU's address space. It is inspired by `gdrcopy`_, which has the same goal,
but has a different implementation. Compared to gdrcopy:

- it works with GeForce GPUs;
- it does not require any special kernel modules;
- it has a different, somewhat simpler (but less powerful) interface;
- it has bindings for both C and Python.

It leverages the capabilities of NVIDIA's Vulkan implementation, so you
will need Vulkan headers to compile it and Vulkan loader to run it.

It is currently only supported on Linux on x86_64, although Windows support
should not be too challenging to add, and it will probably work out of the box
on other architectures.

.. _gdrcopy: https://github.com/NVIDIA/gdrcopy

Installation
------------
You will need header files for Vulkan and CUDA. The libraries are not required
for installation, as they are dynamically loaded at runtime.

C API
^^^^^
There is a Makefile, and it is compiled by running ``make``. If your CUDA
headers are in a non-standard location you will need to edit the Makefile.
There is a ``make install`` target, but you can also just copy the library
and/or header file to the destination of your choice.

Python API
^^^^^^^^^^
To install with pip, simply run ``pip install vkgdr``. For Linux on x86_64,
this will install a binary wheel and you do not even need a compiler.

Concepts
--------
The library is initialised by creating a Vkgdr object. This is a handle which
takes care of loading and preparing the Vulkan and CUDA libraries. Generally
you will only need one such handle. It is safe to use the returned handle from
multiple threads.

The next step is to allocate memory. This must be done using this API, rather
than with functions like :c:func:`cudaMalloc`. From the memory allocation one
may obtain a host pointer (usable on the CPU) and a device pointer (usable
with CUDA).

Memory types
^^^^^^^^^^^^
Vulkan categorises host-visible memory as either coherent or non-coherent,
from the point of view of the host. If it is non-coherent, the application
must take specific steps to synchronise access between the device and the
host. After writing to the memory from the host, it must :dfn:`flush` the
memory before reading it from the device. Similarly, after writing to it with
the device, the host must :dfn:`invalidate` it before reading it. If the
memory is coherent, these steps are not required.

Currently, NVIDIA drivers on desktop systems only provide coherent memory (the
situation on Jetson and similar embedded platforms is unknown). Nevertheless,
vkgdr supports both. You should either

- flush and invalidate memory ranges as necessary for correctness; or

- pass the :c:const:`VKGDR_OPEN_REQUIRE_COHERENT` flag when creating the Vkgdr
  object.

Additionally, device work must be synchronised with the CPU to ensure that the
device will see the latest values. In particular, only kernel launches made
after the memory has been written by the CPU (and flushed, if non-coherent)
are guaranteed to see the updated values.

BAR size
^^^^^^^^
The amount of GPU memory that can be allocated in this way is limited by the
PCIe BAR size. On GeForce cards, this may only be 256 MiB, unless you have the
`resizable BAR`_ feature. You can check the size by running ``nvidia-smi -q -d
MEMORY`` and looking at the statistics for "BAR1 memory usage".

.. _resizable BAR: https://www.nvidia.com/en-us/geforce/news/geforce-rtx-30-series-resizable-bar-support/
