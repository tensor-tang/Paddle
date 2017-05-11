#!/usr/bin/env python
from paddle.trainer_config_helpers import *

use_dummy = get_config_arg("use_dummy", bool, True)
batch_size = get_config_arg('batch_size', int, 1)
is_predict = get_config_arg("is_predict", bool, False)
is_test = get_config_arg("is_test", bool, False)
layer_num = get_config_arg('layer_num', int, 7)

####################Data Configuration ##################
# 10ms as one step
dataSpec = dict(
    uttLengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
    counts = [3, 10, 11, 13, 14, 13, 9, 8, 5, 4, 3, 2, 2, 2, 1],
    lblLengths = [7, 17, 35, 48, 62, 78, 93, 107, 120, 134, 148, 163, 178, 193, 209],
    freqBins = 161,
    charNum = 29, # 29 chars
    scaleNum = 1280
    )
num_classes = dataSpec['charNum']
if not is_predict:
    train_list = 'data/train.list' if not is_test else None
    test_list = None #'data/test.list'
    args = {
        'uttLengths': dataSpec['uttLengths'],
        'counts': dataSpec['counts'],
        'lblLengths': dataSpec['lblLengths'],
        'freqBins': dataSpec['freqBins'],
        'charNum': dataSpec['charNum'],
        'scaleNum': dataSpec['scaleNum'],
        'batch_size': batch_size
    }
    define_py_data_sources2(
        train_list,
        test_list,
        module='dummy_provider' if use_dummy else 'image_provider',
        obj='process',
        args=args)

###################### Algorithm Configuration #############
settings(
    batch_size=batch_size,
    learning_rate=1e-3,
#    learning_method=AdamOptimizer(),
#    regularization=L2Regularization(8e-4),
)

####################### Deep Speech 2 Configuration #############
def mkldnn_CBR(input, kh, kw, sh, sw, ic, oc, clipped = 20):
    tmp = mkldnn_conv(
        input = input,
        num_channels = ic,
        num_filters = oc,
        filter_size_y = kh,
        filter_size = kw,
        stride_y = sh,
        stride = sw,
        act = LinearActivation()
    )
    return mkldnn_bn(
        input = tmp,
        num_channels = oc,
        act = MkldnnReluActivation()) # TODO: change to clippedRelu

def BiDRNN(input, dim_out, dim_in=None):
    tmp = mkldnn_fc(input=input, size=dim_out, bias_attr=False, act=LinearActivation()) #act=None
    tmp = mkldnn_bn(input = tmp, isSeq=True, num_channels = dim_out, act = None)
    return mkldnn_rnn(
            input=tmp,
            input_mode="skip_input",
            bi_direction = True,
            activation = "rnn_relu",
            output_mode = "sum",
            layer_num=1)

######## DS2 model ########
tmp = data_layer(name = 'data', size = dataSpec['freqBins'])

tmp = mkldnn_reorder(input = tmp,
                format_from='nchw',
                format_to='nhwc',
                dims_from=[-1, -1, 1, dataSpec['freqBins']],
                bs_index=0)

tmp = mkldnn_reshape(input=tmp,
                name="view_to_noseq",
                reshape_type=ReshapeType.TO_NON_SEQUENCE,
                img_dims=[1, dataSpec['freqBins'], -1])


# conv, bn, relu
tmp = mkldnn_CBR(tmp, 5, 20, 2, 2, 1, 32)
tmp = mkldnn_CBR(tmp, 5, 10, 1, 2, 32, 32)

# (bs, 32, 75, seq) to (seq,bs,2400)
tmp = mkldnn_reorder(
                input = tmp,
                format_from='nhwc',
                format_to='chwn',
                dims_from=[1, -1, 2400, -1],
                bs_index=1)

tmp = mkldnn_reshape(input=tmp,
                name="view_to_mklseq",
                reshape_type=ReshapeType.TO_MKLDNN_SEQ,
                img_dims=[2400, 1, 1],
                seq_len=-1)
                
for i in xrange(layer_num):
    tmp = BiDRNN(tmp, 1760)

tmp = mkldnn_fc(input=tmp, size=num_classes + 1, act=LinearActivation()) #act=None

# (seq, bs, dim) to (bs, dim, seq)
tmp = mkldnn_reorder(
                input = tmp,
                format_from='chwn',
                format_to='nhwc',
                dims_from=[-1, -1, num_classes + 1, 1],
                bs_index=1)

# (bs, dim, seq) to (bs, seq, dim)
tmp = mkldnn_reorder(
                input = tmp,
                format_from='nchw',
                format_to='nhwc',
                dims_from=[-1, num_classes + 1, -1, 1],
                bs_index=0)

output = mkldnn_reshape(input=tmp,
                name="view_to_paddle_seq",
                reshape_type=ReshapeType.TO_PADDLE_SEQUENCE,
                img_dims=[-1, 1, 1],
                seq_len=-1)

if not is_predict:
    lbl = data_layer(name='label', size=num_classes)
    cost = warp_ctc_layer(input=output, name = "WarpCTC", blank = 0, label=lbl, size = num_classes + 1) # CTC size should +1
# use ctc so we can use multi threads
#    cost = ctc_layer(input=output, name = "CTC", label=lbl, size = num_classes + 1) # CTC size should +1
    outputs(cost)
else:
    outputs(output)
