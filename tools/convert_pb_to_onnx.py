import argparse
import onnx
from caffe2.python import core, workspace
import caffe2.python.onnx.frontend
from caffe2.proto import caffe2_pb2
import detectron.utils.c2 as c2_utils

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
c2_utils.import_custom_ops()

@classmethod
def _annotate_consumed_ignore(c, g): pass
caffe2.python.onnx.frontend.Caffe2Frontend._annotate_consumed = _annotate_consumed_ignore

data_type = onnx.TensorProto.FLOAT
data_shape = (1, 3, 800, 800)
value_info = {
    'data': (data_type, data_shape)
}

def load_net(filename):
    net = caffe2_pb2.NetDef()
    with open(filename, 'rb') as f:
        net.ParseFromString(f.read())
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

    onnx.checker.check_model(onnx_model)
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