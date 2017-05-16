#!/usr/bin/env python
from paddle.trainer_config_helpers import *

use_dummy = get_config_arg("use_dummy", bool, True)
batch_size = get_config_arg('batch_size', int, 1)
is_predict = get_config_arg("is_predict", bool, False)
is_test = get_config_arg("is_test", bool, False)
layer_num = get_config_arg('layer_num', int, 6)

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
### TODO:
###     1. change all relu to clipped relu
###     2. rnn


def mkldnn_CBR(input, kh, kw, sh, sw, ic, oc, clipped = 20):
    tmp = mkldnn_conv(
        input = input,
        num_channels = ic,
        num_filters = oc,
        filter_size = [kw, kh],
        stride = [sw, sh],
        act = LinearActivation()
    )
    return mkldnn_bn(
        input = tmp,
        num_channels = oc,
        act = MkldnnReluActivation())

def BiDRNN(input, dim_out, dim_in=None):
    if dim_in is None:
        dim_in = dim_out
    tmp = mkldnn_fc(input=input, dim_in=dim_in, dim_out=dim_out,
                    bias_attr=False, act=LinearActivation()) # maybe act=None
    tmp = mkldnn_bn(input = tmp, isSeq=True, num_channels = dim_out, act = None)
    return mkldnn_rnn(
            input=tmp,
            input_mode=MkldnnRnnConfig.SKIP_INPUT,
            alg_kind = MkldnnRnnConfig.RNN_RELU,  # try to use clipped
            use_bi_direction = True,
            sum_output = True,
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
                reshape_type=ReshapeType.TO_MKL_SEQUENCE,
                img_dims=[2400, 1, 1],
                seq_len=-1)

tmp = BiDRNN(tmp, 1760, 2400)
for i in xrange(layer_num):
    tmp = BiDRNN(tmp, 1760)

# since ctc should +1 of the dim
ctc_dim = num_classes + 1

tmp = mkldnn_fc(input=tmp,
                dim_in = 1760,
                dim_out = ctc_dim,
                act=LinearActivation()) #act=None

# (seq, bs, dim) to (bs, dim, seq)
tmp = mkldnn_reorder(
                input = tmp,
                format_from='chwn',
                format_to='nhwc',
                dims_from=[-1, -1, ctc_dim, 1],
                bs_index=1)

# (bs, dim, seq) to (bs, seq, dim)
tmp = mkldnn_reorder(
                input = tmp,
                format_from='nchw',
                format_to='nhwc',
                dims_from=[-1, ctc_dim, -1, 1],
                bs_index=0)

output = mkldnn_reshape(input=tmp,
                name="view_to_paddle_seq",
                reshape_type=ReshapeType.TO_PADDLE_SEQUENCE,
                img_dims=[ctc_dim, 1, 1],
                seq_len=-1)

if not is_predict:
    lbl = data_layer(name='label', size=num_classes)
    cost = warp_ctc_layer(input=output, name = "WarpCTC", blank = 0, label=lbl, size = ctc_dim) # CTC size should +1
# use ctc so we can use multi threads
#    cost = ctc_layer(input=output, name = "CTC", label=lbl, size = num_classes + 1) # CTC size should +1
    outputs(cost)
else:
    outputs(output)
