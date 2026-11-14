# Build the KV260 VART/XIR C++ runner.
#
# Prefer the system Vitis AI installation (/usr/include, /usr/lib). If it is
# absent, fall back to the conda environment in VAI_ENV. This covers both the
# AMD container and the board.
#
#   make
#   ./run_kv260_dpu model.xmodel image.jpg --classes crack,spalling

CXX      ?= g++
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra
TARGET   := run_kv260_dpu
SOURCE   := run_kv260_dpu.cpp

# Override on the command line if your Vitis AI environment lives elsewhere:
#   make VAI_ENV=/path/to/env
VAI_ENV ?= /opt/vitis_ai/conda/envs/vitis-ai-pytorch
VAI_INCLUDE :=
VAI_LIBDIR :=

# Prefer the complete system installation (/usr/include, /usr/lib). Some
# container images ship a partial copy of the XIR headers inside the conda
# environment, so only use that copy when the system one is absent.
ifeq ($(wildcard /usr/include/vart/runner.hpp),)
ifneq ($(wildcard $(VAI_ENV)/include/vart/runner.hpp),)
VAI_INCLUDE := -I$(VAI_ENV)/include
endif
ifneq ($(wildcard $(VAI_ENV)/lib/libvart-runner.so),)
VAI_LIBDIR := -L$(VAI_ENV)/lib -Wl,-rpath=$(VAI_ENV)/lib -Wl,-rpath-link=$(VAI_ENV)/lib
endif
endif

# The container ships OpenCV 3 (pkg-config opencv); the board usually ships
# OpenCV 4 (pkg-config opencv4). Try both, then fall back to explicit libs.
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)
ifeq ($(strip $(OPENCV_LIBS)),)
OPENCV_LIBS := -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
endif

CPPFLAGS += $(VAI_INCLUDE) $(OPENCV_CFLAGS)
LDFLAGS  += $(VAI_LIBDIR)
LDLIBS   += -lvart-runner -lxir $(OPENCV_LIBS) -lpthread

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)
