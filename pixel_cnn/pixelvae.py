"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr_gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting






# Just encoding these arguments in a way that this doesn't have to be run in a script (hacky times call for hacky measures!)

# # data I/O
# parser.add_argument('-i', '--data_dir', type=str, default='/local_home/tim/pxpp/data', help='Location for the dataset')
# parser.add_argument('-o', '--save_dir', type=str, default='/local_home/tim/pxpp/save', help='Location for parameter checkpoints and samples')
# parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
# parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
# parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
# # model
# parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
# parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
# parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
# parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
# parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
# parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')
# # optimization
# parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
# parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
# parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
# parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
# parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
# parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
# parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# # evaluation
# parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')

class Args:
    def __init__(self):
        self.nr_resnet = 3 # 5
        self.nr_filters = 160
        self.nr_logistic_mix = 10
        self.resnet_nonlinearity = 'relu'
        self.learning_rate = 0.001
        self.lr_decay = 0.999995
        self.dropout_p = 0.3 # 0.5
        self.polyak_decay = 0.9995




def make_pixel_cnn(ground_images, prev_samples, latents, gpus, min_highway_connection, max_highway_connection)
    """
    This is the code written to cannabalize OpenAI's PixelCNN, to use as a decoder in a PixelVAE

    Being as lazy as possible.
    Passing in parameters from the VAE framework
    Translating between our input params, and the Open AI's params
    C&P OpenAI's code to make the pixel cnn++

    TODO: properly write this + clean up all of the commenting!!
    """
    # Constants
    args = Args()

    # Define OpenAI's params in terms of ours (kinda, hacky times meant I had to change things in how this runs)
    xs = list(tf.split(ground_images, num_or_size_of_splits=gpus))    # need to split targets to split across gpus
    hs = list(tf.split(latents, num_or_size_of_splits=gpus))
    x_init = xs[0]
    h_init = hs[0]

    # placeholders for when we want to run in generative mode
    x_shape = images.get_shape().as_list()
    latent_shape = latents.get_shape().as_list()
    x_sample = tf.placeholder(tf.float32, shape=x_shape) 
    h_sample = tf.placeholder(tf.float32, shape=latent_shape) 
    x_samples = list(tf.split(x_sample, num_or_size_of_splits=gpus))    # need to split targets to split across gpus
    h_samples = list(tf.split(h_sample, num_or_size_of_splits=gpus))


    # Make PixelCNN++
    # create the model
    model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity }
    model = tf.make_template('model', model_spec)

    # run once for data dependent initialization of parameters
    init_pass = None
    with tf.variable_scope(scope="pixelcnn", reuse=tf.AUTO_REUSE):
        init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

    # keep track of moving average (n.b. edited code to only get pixelcnn params)
    actually_all_params = tf.trainable_variables()
    for var in tf_vars:
        if substr in var.name:
            subset.append(var)
    return subset
    ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
    maintain_averages_op = tf.group(ema.apply(all_params))


    # build model
    train_outs = []
    test_outs = []
    for i in range(gpus):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope(scope="pixelcnn", reuse=tf.AUTO_REUSE):
                train_out = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
                train_out = nn.sample_from_discretized_mix_logistic(train_out, args.nr_logistic_mix)
                train_outs.append(train_out)
                test_out = model(x_sample[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
                test_out = nn.sample_from_discretized_mix_logistic(test_out, args.nr_logistic_mix)
                test_outs.append(test_out)

    with tf.variable_scope(scope="pixelcnn", reuse=tf.AUTO_REUSE):
        # concatenate the outputs from the different gpus
        train_out = tf.concat(train_outs, axis=0)
        test_out = tf.concat(test_outs, axis=0)

        # add a highway connection (per IMAGE not per PIXEL)
        highway_train_ratio = min_highway_ratio + (max_highway_ratio-min_highway_ratio) * tf.contrib.layers.fully_connected(latents, 1, activation_fn=tf.sigmoid, scope="pixelcnn")
        highway_train_out = tf.multiply(highway_train_ratio, train_out) + tf.multiply(1-highway_train_ratio, prev_samples)

        # add a highway connection for the test image (use scoping to share params :)
        highway_test_ratio = min_highway_ratio + (max_highway_ratio-min_highway_ratio) * tf.contrib.layers.fully_connected(h_sample, 1, activation_fn=tf.sigmoid, scope="pixelcnn")



    # things that we need to return to run autoregressive sampling, ontop of what is usually returned from another VAE generator network
    # we need to have latents evaluated, but repeatedly fed in to a sampled generation process
    cache = {}
    cache['produce_latent_op'] = latents # op to run for, we're assuming this isn't step 0 here! (Otherwise this is a placeholder)
    cache['x_feed'] = x_sample # placeholder to feed input x into the network at generation time (start with x = zeros, and then sample pixels at a time) 
    cache['h_feed'] = h_sample # placholder to feed latents. (N.B. we can't just use latents here, because it's a stochastic operation, and latents needs to be consistent between autoregressive samples) 
    cache['sample_op'] = test_out # pixelvae has a different 
    cache['highway_test_ratio'] = highway_test_ratio # to provide a mixing ratio from the previous and current sample
    cache['prev_samples'] = prev_samples # needed to generate images from the previous step (to be mixed)

    cache['update_exp_move_avg_op'] = maintain_averages_op # op to run to update the exponential moving averages, needs to be grouped with the train op of seqvae
    cache['init_pass'] = init_pass # initial pass over the network, needed for some reason, to initialize the data dependent variables

    # return everything (zero noise)
    return highway_train_out, 0, cache










# sample from the model, takes the cache from above, and unpacks it first
def sample_from_model(sess, cache, feed_dict_for_latents):
    produce_latent_op = cache['produce_latent_op']
    x_feed = cache['x_sample']
    h_feed = cache['h_sample']
    sample_op = cache['sample_op']
    highway_ratio = cache['highway_test_ratio']
    sample_prev_step = cache['prev_samples']

    # sample the previous step, the latents for this step and the highway connection for this step
    prev_samples, latents, ratio = sess.run([sample_prev_step, produce_latent_op, highway_ratio], feed_dict=feed_dict_for_latents)

    # sample the image for this step
    x_shape = x_feed.get_shape().as_list()
    x_gen = [np.zeros(x_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            feed_dict = {x_feed: x_gen, h_feed: latents}
            new_x_gen = sess.run(sample_op, feed_dict=feed_dict)
            for i in range(args.nr_gpu):
                x_gen[:,yi,xi,:] = new_x_gen_np[:,yi,xi,:]

    # combine according to highway connection ratio
    final_samples = ratio * x_gen + (1.0 - ratio) * prev_samples

    return final_samples
