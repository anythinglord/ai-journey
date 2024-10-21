import numpy as np
import tensorflow as tf

# MDN-RNN model
class MdnRNN(object):

    # parameters and variables initialization
    # Hyper parameters
    def __init__(self, hps, reuse = False, gpu_mode = False):
        self.hps = hps
        self.reuse = reuse
        with tf.variable_scope('mdn_rnn', reuse = self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model trained with cpu')
                    self.g = tf.Graph()
                    if self.g.as_default():
                        self.build_model(hps)
            else:
                tf.logging.info('Model trained with gpu')
                self.g = tf.Graph()
                if self.g.as_default():
                    self.build_model(hps)
        self._init_session()

    # create method for the MDN-RNN model architecture
    def build_model(self, hps):
        # Build RNN
        self.num_mixture = hps.num_mixture # Number of Dentisy mixtures 
        KMIX = self.num_mixture
        INWIDTH =  hps.input_seq_width # input width 
        OUTWIDTH = hps.output_seq_width # output width 
        LENGTH = hps.max_seq_width # max step`s num
        if hps.is_trainning:
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)