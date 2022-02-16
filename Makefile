CC = gcc
NVCC = nvcc
CFLAGS = -Wall -g -I/usr/local/cuda/include
NVCCFLAGS =
LIBS = -L/usr/local/cuda/lib -lcuda
TARGETS = libvkgdr.so libvkgdr.so.1 vkgdr_test

all: $(TARGETS)

libvkgdr.so.1: vkgdr.c vkgdr.h Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname,libvkgdr.so.1 -o $@ $< $(LIBS)

libvkgdr.so:
	ln -s libvkgdr.so.1 libvkgdr.so

vkgdr_test: vkgdr_test.cu vkgdr.h libvkgdr.so Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $< -Xlinker -rpath,$(PWD) -L. -lvkgdr

clean:
	rm -f $(TARGETS)
