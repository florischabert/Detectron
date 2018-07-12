# Use Caffe2 image as parent image
FROM nvcr.io/nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends\
 wget git libsm6 libxext6\
 build-essential cmake swig vim gdb\
 libprotobuf-dev protobuf-compiler\
 libgflags-dev libgoogle-glog-dev\
 libgtest-dev libiomp-dev libleveldb-dev\
 liblmdb-dev libopencv-dev libopenmpi-dev\
 libsnappy-dev libprotobuf-dev openmpi-bin\
 openmpi-doc protobuf-compiler\
 python-dev python-pip graphviz

RUN pip install --upgrade pip setuptools
RUN pip install\
 numpy protobuf networkx\
 pydot future enum34\
 pycuda

# Install TensorRT
COPY nv-tensorrt-repo-* /nv-tensorrt.deb
RUN dpkg -i /nv-tensorrt.deb &&\
 apt-get update &&\
 apt-get install -y tensorrt python-libnvinfer-doc &&\
 apt-get remove -y nv-tensorrt-repo-* &&\
 rm -f /nv-tensorrt.deb

# Install Caffe2
RUN git clone --recursive https://github.com/pytorch/pytorch.git /pytorch &&\
 cd pytorch &&\
 git submodule update --init &&\
 mkdir build && cd build &&\
 cmake .. && make -j8 install &&\
 rm -rf /pytorch

# Install ONNX
RUN git clone https://github.com/onnx/onnx.git /onnx &&\
 cd /onnx &&\
 git submodule update --init --recursive &&\
 python setup.py install &&\
 rm -rf /onnx

# Install TensorRT backend
RUN git clone -b fp16 --recursive https://github.com/florischabert/onnx-tensorrt.git /onnx-tensorrt &&\
 mkdir /onnx-tensorrt/build && cd /onnx-tensorrt/build &&\
 cmake .. -DTENSORRT_ROOT=/opt/tensorrt &&\
 make -j8 install &&\
 cd /onnx-tensorrt &&\
 python setup.py install &&\
 rm -rf /onnx-tensorrt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi &&\
 cd /cocoapi/PythonAPI &&\
 pip install cython &&\
 make install &&\
 rm -rf /cocoapi

# Install Detectron
RUN git clone -b retinanet https://github.com/florischabert/detectron /detectron &&\
 cd /detectron &&\
 pip install --ignore-installed -r requirements.txt &&\
 make &&\
 mkdir -p build &&\
 cd /detectron/build &&\
 cmake .. -DCUDA_ARCH_NAME=Volta && make -j8

# Set dataset paths
RUN ln -s /coco /detectron/detectron/datasets/data/coco

# Set working dir
WORKDIR /detectron