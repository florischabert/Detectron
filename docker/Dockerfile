# Use Caffe2 image as parent image
FROM nvcr.io/nvidia/caffe2:18.06-py2

# Clone the Detectron repository
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
