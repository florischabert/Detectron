# Use CUDA image as parent image
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Install extra packages
RUN apt-get update
RUN apt-get install -y wget git libsm6 libxext6 python-networkx\
 graphviz vim

# Install conda and caffe2
RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
RUN bash Miniconda2-latest-Linux-x86_64.sh -b -p /conda
ENV PATH="/conda/bin:${PATH}"
RUN conda install -c caffe2 caffe2-cuda9.0-cudnn7

# Copy Detectron
RUN git clone https://github.com/facebookresearch/detectron /detectron

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