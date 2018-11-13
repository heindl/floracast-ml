from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

def max_pool_v2(value, ksize, strides, padding, data_format="NHWC", name=None):
  with ops.name_scope(name, "MaxPoolV2", [value]) as name:
    value = ops.convert_to_tensor(value, name="input")
    return gen_nn_ops._max_pool_v2(value,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)


