# Use CUDA image as parent image
FROM nvcr.io/nvidia/tensorrt:18.06-py2

# Install extra packages
RUN apt-get update && apt-get install -y\
 wget git libsm6 libxext6 python-networkx\
 graphviz python-pydot python-pydot-ng\
 build-essential cmake gdb vim\
 libprotobuf-dev protobuf-compiler

# # Install conda
RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
RUN bash Miniconda2-latest-Linux-x86_64.sh -b -p /conda
RUN rm -f Miniconda2-latest-Linux-x86_64.sh
ENV PATH="/conda/bin:${PATH}"

# Install Caffe2
# RUN conda install -c caffe2 caffe2-cuda9.0-cudnn7

# # Install ONNX 
RUN conda install -c conda-forge onnx

# Install TensorRT backend
RUN git clone --recursive https://github.com/onnx/onnx-tensorrt.git /onnx-tensorrt
RUN mkdir build
WORKDIR /onnx-tensorrt/build
RUN cmake .. -DTENSORRT_ROOT=/opt/tensorrt
# RUN make -j8 && make install

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
# RUN make
# RUN make ops

# Set dataset paths
RUN ln -s /coco /detectron/detectron/datasets/data/coco
