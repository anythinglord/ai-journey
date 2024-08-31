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

    # create method for the VAE model architecture
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.x = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])
            # build encoder (convolution)
            h = tf.layers.conv2d(self.x, 32, 4, strides = 2, activation = tf.nn.relu, name='enc_conv1') # 32 filters of 4x4
            h = tf.layers.conv2d(h, 64, 4, strides = 2, activation = tf.nn.relu, name='enc_conv2')
            h = tf.layers.conv2d(h, 128, 4, strides = 2, activation = tf.nn.relu, name='enc_conv3')
            h = tf.layers.conv2d(h, 256, 4, strides = 2, activation = tf.nn.relu, name='enc_conv4')
            h = tf.reshape(h, shape=[-1, 2 * 2 * 256]) # column vector, flattened vector

            # build the variational stage 
            self.mu = tf.layers.dense(h, self.z_size, name = 'enc_fc_mu') # h  flattened vector, self.z_size number of networks
            self.logvar = tf.layers.dense(h, self.z_size, name = 'enc_fc_logavar') # variance`s logarithm
            self.sigma = tf.exp(self.logvar/2.0)
            self.epsilon = tf.random_normal([self.batch_size, self.z_size]) # N(0,1) distribution
            self.z = self.mu + self.sigma * self.epsilon

            # build decoder (reverse convolution)
            h = tf.layers.dense(self.z, 1024, name = 'dec_fc')
            h = tf.reshape(h, shape=[-1, 1 * 1 * 1024]) # -1 to get a column vector
            h = tf.layers.conv2d_transpose(h, 128, 5, strides = 2, activation = tf.nn.relu, name='dec_deconv1')
            h = tf.layers.conv2d_transpose(h, 64, 5, strides = 2, activation = tf.nn.relu, name='dec_deconv2')
            h = tf.layers.conv2d_transpose(h, 32, 6, strides = 2, activation = tf.nn.relu, name='dec_deconv3')
            self.y = tf.layers.conv2d_transpose(h, 3, 6, strides = 2, activation = tf.nn.sigmoid, name='dec_deconv4') # final prediction 

            # build the training operation
            if self.is_training:
                # descendent gradient technique
                self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
                self.r_loss = tf.reduce_sum(tf.square(self.x - self.y), reduction_indices = [1, 2 ,3]) # mean square error
                self.r_loss = tf.reduce_mean(self.r_loss)      
                # Kullback–Leibler divergence technique                            
                self.kl_loss = -0.5 * tf.reduce_sum((1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)), reduction_indices = 1)
                self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
                self.kl_loss = tf.reduce_mean(self.kl_loss)
                self.loss = self.r_loss + self.kl_loss
                self.lr = tf.Variable(self.learning_rate, trainable = False) # learning rate 
                self.optimizer = tf.train.AdamOptimizer(self.lr)                           
                grads = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(grads, global_step = self.global_step, name = "train_step")
                self.init = tf.global_variables_initializer()