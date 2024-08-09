import numpy as np
import tensorflow as tf

# VAE model
class ConvVAE(object):

    # parameters and variables initialization
    def __init__(self, z_size =32, batch_size = 1, learning_rate = 0.0001,
               kl_tolerance = 0.5, is_training = False, reuse = False, gpu_model = False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse
        with tf.variable_scope('conv_vae', reuse = self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model trained with cpu')
                    self._build_graph()
            else:
                tf.logging.info('Model trained with gpu')
                self._build_graph()
        self._init_session()