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