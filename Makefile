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

CC = gcc
NVCC = nvcc
CFLAGS = -Wall -g -I/usr/local/cuda/include -fPIC -fvisibility=hidden
NVCCFLAGS =
LIBS = -ldl
TARGETS = libvkgdr.so libvkgdr.so.1 vkgdr_test

all: $(TARGETS)

libvkgdr.so.1: vkgdr.c vkgdr.h Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname,libvkgdr.so.1 -o $@ $< $(LIBS)

libvkgdr.so: libvkgdr.so.1
	rm -f $@
	ln -s $< $@

vkgdr_test: vkgdr_test.cu vkgdr.h libvkgdr.so Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $< -Xlinker -rpath,$(PWD) -L. -lvkgdr

clean:
	rm -f $(TARGETS)
