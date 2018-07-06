# Use CUDA image as parent image
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Install extra packages
RUN apt-get update && apt-get install -y\
 wget git libsm6 libxext6 python-networkx\
 graphviz python-pydot python-pydot-ng vim\
 gdb cmake protobuf-compiler libprotoc-dev\
 python-pip libblas-dev liblapacke-dev

# Install Caffe2
RUN git clone https://github.com/pytorch/pytorch.git --recursive /pytorch
RUN [ "/bin/bash", "-c", "cp /pytorch/modules/detectron/*.{h,cc,cu} /pytorch/caffe2/operators/" ]
WORKDIR /pytorch/
RUN python setup_caffe2.py install

# Copy Detectron
RUN git clone -b retinanet https://github.com/florischabert/detectron /detectron

# Install Python dependencies
RUN pip install -r /detectron/requirements.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Detectron root
WORKDIR /detectron

# Set up Python modules
RUN make

# Set dataset paths
RUN ln -s /coco /detectron/detectron/datasets/data/coco