import argparse
import onnx
import caffe2.python.onnx.frontend
from caffe2.proto import caffe2_pb2

data_type = onnx.TensorProto.FLOAT
data_shape = (1, 3, 800, 800)
value_info = {
    'data': (data_type, data_shape)
}

def load_net(filename):
    net = caffe2_pb2.NetDef()
    with open(filename, 'rb') as f:
        net.ParseFromString(f.read())
    net.device_option.device_type = caffe2_pb2.CUDA
    net.device_option.cuda_gpu_id = 0
    return net

def remove_non_onnx_ops(net):
    for op in net.op:
        bad_args = [arg for arg in op.arg if arg.name == 'exhaustive_search']
        for arg in bad_args: op.arg.remove(arg)

def convert_ops_to_onnx(net):
    for op in net.op:
        if op.type == 'ResizeNearest':
            op.type = 'Upsample'

def convert(predict, init, output):
    predict_net = load_net(predict)
    init_net = load_net(init)

    remove_non_onnx_ops(predict_net)
    convert_ops_to_onnx(predict_net)

    onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
        predict_net, init_net, value_info)

    onnx.save(onnx_model, output)

def main():
    parser = argparse.ArgumentParser(description='Convert a caffe2 network to ONNX')
    parser.add_argument('--model', dest='model', help='model file', default='predict_net.pb', type=str)
    parser.add_argument('--init', dest='init', help='init file', default='init_net.pb', type=str)
    parser.add_argument('--output', dest='output', help='output file', default='model.onnx', type=str)
    args = parser.parse_args()
    
    convert(args.model, args.init, args.output)

if __name__ == '__main__':
    main()
