#import io, os
import numpy as np
from paddle.trainer.PyDataProvider2 import *

def initHook(settings, uttLengths, counts, lblLengths, batch_size, **kwargs):
    settings.uttLengths = uttLengths
    settings.counts = counts
    settings.lblLengths = lblLengths
    settings.freqBins = kwargs.get('freqBins', 161)
    settings.charNum = kwargs.get('charNum', 29)
    settings.scaleNum = kwargs.get('scaleNum', 1280)
    assert settings.scaleNum % batch_size == 0 # TODO: in real data should consider more

    for i in xrange(len(counts)):
        settings.counts[i] = counts[i] * settings.scaleNum

    settings.slots = {
        'data': dense_vector_sequence(settings.freqBins), # fixed dim
        'label': integer_value_sequence(settings.charNum) # classes range
    }
    settings.logger.info("freq bin: %d" % (settings.freqBins))


@provider(
    init_hook=initHook, pool_size = 128, should_shuffle=False) # min_pool_size=-1, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file_list):
    max_utt_len = max(settings.uttLengths)
    max_lbl_len = max(settings.lblLengths)
    extra_len = 1000 # just for more space of random table
    data_table = np.random.rand(max_utt_len + extra_len, settings.freqBins) #(1500+1000, 161)
    data_table = data_table.astype('float32')
    label_table = np.random.random_integers(0, settings.charNum-1, max_lbl_len + extra_len) # TODO: range (0~max) or (1~max-1)
    label_table = label_table.astype('int')
    #print(data_table.shape, label_table.shape)
    i = 0
    for cnts in settings.counts:
        for cnt in xrange(cnts):
            utt_len = settings.uttLengths[i]
            lbl_len = settings.lblLengths[i]
            dat_start_idx = np.random.randint(0, max_utt_len + extra_len - utt_len) # [0, len)
            lbl_start_idx = np.random.randint(0, max_lbl_len + extra_len - lbl_len)
            #print ("data range", dat_start_idx, "~", dat_start_idx+utt_len)
            #print ("labl range", lbl_start_idx, "~", lbl_start_idx+lbl_len)
            dat = data_table[dat_start_idx : dat_start_idx+utt_len][:]
            lbl = label_table[lbl_start_idx : lbl_start_idx+lbl_len]
            yield {
                'data': dat.tolist(),
                'label': lbl.tolist()
            }
        i += 1