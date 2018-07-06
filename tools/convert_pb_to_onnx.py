import onnx
import caffe2.python.onnx.frontend
from caffe2.proto import caffe2_pb2

data_type = onnx.TensorProto.FLOAT
data_shape = (1, 3, 800, 800)
value_info = {
    'data': (data_type, data_shape)
}

predict_net = caffe2_pb2.NetDef()
with open('model.pb', 'rb') as f:
    predict_net.ParseFromString(f.read())

init_net = caffe2_pb2.NetDef()
with open('model_init.pb', 'rb') as f:
    init_net.ParseFromString(f.read())

onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
    predict_net,
    init_net,
    value_info,
)

onnx.checker.check_model(onnx_model)