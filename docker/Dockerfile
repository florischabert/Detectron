# Use Caffe2 image as parent image
FROM nvcr.io/nvidia/caffe2:18.06-py2

# Install TensorRT repo
ADD nv-tensorrt-repo-* /nv-tensorrt.deb
RUN dpkg -i /nv-tensorrt.deb

# Install packages
RUN apt-get update && apt-get install -y\
 wget git libsm6 libxext6 python-networkx\
 graphviz python-pydot python-pydot-ng\
 build-essential cmake swig vim gdb\
 libprotobuf-dev protobuf-compiler\
 tensorrt python-libnvinfer-doc

# Cleanup
RUN apt-get remove -y nv-tensorrt-repo-*
RUN rm -f /nv-tensorrt.deb

# Install ONNX
# RUN pip uninstall -y onnx
RUN git clone https://github.com/onnx/onnx.git /onnx
WORKDIR /onnx
RUN git submodule update --init --recursive
RUN python setup.py install
RUN rm -rf /onnx

# Install TensorRT backend
RUN git clone --recursive https://github.com/onnx/onnx-tensorrt.git /onnx-tensorrt
WORKDIR /onnx-tensorrt/build
RUN cmake .. -DTENSORRT_ROOT=/opt/tensorrt
RUN make -j8 install
WORKDIR /onnx-tensorrt
RUN python setup.py install
RUN rm -rf /onnx-tensorrt
RUN pip remove onnx

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN pip install cython
RUN make install
RUN rm -rf /cocoapi

# Install Detectron
RUN git clone -b retinanet https://github.com/florischabert/detectron /detectron
WORKDIR /detectron
RUN pip install --ignore-installed -r requirements.txt
RUN make
RUN mkdir -p build
WORKDIR /detectron/build
RUN cmake .. -DCUDA_ARCH_NAME=Volta && make -j8

# Set dataset paths
RUN ln -s /coco /detectron/detectron/datasets/data/coco

# Set working dir
WORKDIR /detectron