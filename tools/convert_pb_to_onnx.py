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

@classmethod
def _create_upsample(cls, op_def, shapes):
    scales = [1.0, 1.0, 2.0, 2.0]
    for arg in op_def.arg:
        if arg.name == 'width_scale': scales[-2] = arg.f
        if arg.name == 'height_scale': scales[-1] = arg.f

    node = onnx.helper.make_node(
        'Upsample',
        inputs=op_def.input,
        outputs=op_def.output,
        scales=scales,
        mode='nearest'
    )
    return node
caffe2.python.onnx.frontend.Caffe2Frontend._create_upsample = _create_upsample
caffe2.python.onnx.frontend.Caffe2Frontend._special_operators['ResizeNearest'] = '_create_upsample'

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

def convert(predict, init, output):
    predict_net = load_net(predict)
    init_net = load_net(init)

    remove_non_onnx_ops(predict_net)

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