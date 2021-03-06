import tensorflow as tf
import numpy as np
import math
import glob
from matplotlib import pyplot as plt
import os, sys, re

def lrelu(x, rate=0.1):
    # return tf.nn.relu(x)
    return tf.maximum(tf.minimum(x * rate, 0), x)

conv2d = tf.contrib.layers.convolution2d
conv2d_t = tf.contrib.layers.convolution2d_transpose
fc_layer = tf.contrib.layers.fully_connected

# num_outputs = num filters
def conv2d_bn_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, int(num_outputs), kernel_size, stride,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    conv = lrelu(conv)
    return conv

def conv2d_bn_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, int(num_outputs), kernel_size, stride,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    conv = tf.nn.relu(conv)
    return conv


def conv2d_t_bn_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, int(num_outputs), kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    conv = tf.nn.relu(conv)
    return conv

def conv2d_t_bn_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, int(num_outputs), kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    conv = lrelu(conv)
    return conv


def conv2d_t_bn(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, int(num_outputs), kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    return conv


def fc_bn_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, int(num_outputs),
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc)
    fc = lrelu(fc)
    return fc


def fc_bn_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, int(num_outputs),
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc)
    fc = tf.nn.relu(fc)
    return fc


class Network:
    def __init__(self, dataset, logger, version, base_dir, netname):
        self.dataset = dataset
        self.batch_size = dataset.batch_size

        self.learning_rate_placeholder = tf.placeholder(shape=[], dtype=tf.float32, name="lr_placeholder")

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        # A unique name should be given to each instance of subclasses during initialization
        self.name = netname

        # Keeping track of version of this model etc
        self.version = version
        self.base_dir = "models/" + netname + "_v" + str(self.version)

        # These should be updated accordingly
        self.iteration = 0
        self.learning_rate = 0.0
        self.read_only = False

        # Keep a reference to the logger
        self.LOG = logger

    def make_model_path(self):
        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)

    def print_network(self):
        self.make_model_path()
        if os.path.isdir(self.base_dir):
            for f in os.listdir(self.base_dir):
                if re.search(r"events.out*", f):
                    os.remove(os.path.join(self.base_dir, f))
        self.writer = tf.summary.FileWriter(self.base_dir, self.sess.graph)
        self.writer.flush()

    """ Save network, if network file already exists back it up to models/old folder. Only one back up will be created
    for each network """
    def save_network(self):
        if not self.read_only:
            # Saver and Summary ops cannot run in GPU
            with tf.device('/cpu:0'):
                saver = tf.train.Saver()
            self.make_model_path()
            if not os.path.isdir("models/old"):
                os.mkdir("models/old")
            file_name = self.base_dir + "/" + self.name + ".ckpt"
            if os.path.isfile(file_name):
                os.rename(file_name, "models/old/" + self.name + "_v" + str(self.version) + ".ckpt")
            saver.save(self.sess, file_name)

    """ Either initialize or load network from file.
    Always run this at end of initialization for every subclass to initialize Variables properly """
    def init_network(self, restart=False):
        self.sess.run(tf.global_variables_initializer())
        if restart:
            return
        file_name = self.base_dir + "/" + self.name + ".ckpt"
        if len(glob.glob(file_name + '*')) != 0:
            saver = tf.train.Saver()
            try:
                saver.restore(self.sess, file_name)
            except:
                self.LOG.warning("Warning: network load failed, reinitializing all variables", sys.exc_info()[0])
                self.sess.run(tf.global_variables_initializer())
        else:
            self.LOG.info("No checkpoint file found, Initializing model from random")

    """ This function should train on the given batch and return the training loss """
    def train(self, batch_input, batch_target, condition=None):
        return None

    """ This function should take the input and return the reconstructed images """
    def test(self, batch_input, condition=None):
        return None

