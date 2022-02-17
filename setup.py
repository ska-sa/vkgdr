from setuptools import setup

setup(
    cffi_modules=["src/vkgdr/build.py:ffibuilder"],
    use_katversion=True
)
