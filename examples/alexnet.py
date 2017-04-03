################################################################################
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/asanakoy/caffe-tensorflow
#
# Copyright (c) 2016 Artsiom Sanakoyeu
################################################################################
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import cv2
import sys
import argparse
import multiprocessing
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack import TowerContext
class Alexnet(object):
    """
    Net description
    (self.feed('data')
            .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            .lrn(2, 2e-05, 0.75, name='norm1')
            .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
            .conv(5, 5, 256, 1, 1, group=2, name='conv2')
            .lrn(2, 2e-05, 0.75, name='norm2')
            .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            .conv(3, 3, 384, 1, 1, name='conv3')
            .conv(3, 3, 384, 1, 1, group=2, name='conv4')
            .conv(3, 3, 256, 1, 1, group=2, name='conv5')
            .fc(4096, name='fc6')
            .fc(4096, name='fc7')
            .fc(num_classes, relu=False, name='fc8')
            .softmax(name='prob'))

    WARNING! You should feed images in HxWxC BGR format!
        """
    


    class RandomInitType:
        GAUSSIAN = 0,
        XAVIER_UNIFORM = 1,
        XAVIER_GAUSSIAN = 2

    def __init__(self, init_model=None, num_classes=1,
                 im_shape=(227, 227, 3), device_id='/gpu:0', num_layers_to_init=8,
                 random_init_type=RandomInitType.GAUSSIAN, use_batch_norm=False,
                 gpu_memory_fraction=None, **params):
        """
         Args:
          init_model: dict containing network weights, or a string with path to .np file with the dict,
            if is None then init using random weights and biases
          num_classes: number of output classes
          gpu_memory_fraction: Fraction on the max GPU memory to allocate for process needs.
            Allow auto growth if None (can take up to the totality of the memory).
        :return:
        """
        self.input_shape = im_shape
        self.num_classes = num_classes
        self.device_id = device_id
        self.num_layers_to_init = num_layers_to_init
        self.random_init_type = random_init_type
        self.trainable_vars = None

        if len(self.input_shape) == 2:
            self.input_shape += (3,)

        assert len(self.input_shape) == 3
        if self.num_layers_to_init > 8 or self.num_layers_to_init < 0:
            raise ValueError('Number of layer to init must be in [0, 8] ({} provided)'.
                             format(self.num_layers_to_init))

        if init_model is None:
            net_data = None
        elif isinstance(init_model, basestring):
            if not os.path.exists(init_model):
                raise IOError('Net Weights file not found: {}'.format(init_model))
            print 'Loading Net Weights from: {}'.format(init_model)
            net_data = np.load(init_model).item()

        self.global_iter_counter = tf.Variable(0, name='global_iter_counter', trainable=False)
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, (None,) + self.input_shape, name='x')
            self.y_gt = tf.placeholder(tf.int32, shape=(None,), name='y_gt')
            self.is_phase_train = tf.placeholder(tf.bool, shape=tuple(), name='is_phase_train')
            
        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l 
            
        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

       
        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        with TowerContext('', is_training=True):
            self.create_architecture(net_data, use_batch_norm)

        self.graph = tf.get_default_graph()
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        # please do not use the totality of the GPU memory.
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        self.sess = tf.Session(config=config)
         
       
        

    
    def create_architecture(self,net_data,use_batch_norm):
        tr_vars = dict()
        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l 
        
        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        
        
        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                        return l
        
        
        
        
        
        
        cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock)
        }
        with tf.device(self.device_id):
            layer_index = 0
            # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            with tf.variable_scope('conv1'):
                kernel_height = 11
                kernel_width = 11
                kernels_num = 64
                group = 1
                num_input_channels = int(self.x.get_shape()[3])
                s_h = 4  # stride by the H dimension (height)
                s_w = 4  # stride by the W dimension (width)
                tr_vars['conv1w'], tr_vars['conv1b'] = \
                    self.get_conv_weights(layer_index, net_data,
                                          kernel_height, kernel_width,
                                          num_input_channels / group, kernels_num)
                layer_index += 1
                conv1_in = Alexnet.conv(self.x, tr_vars['conv1w'], tr_vars['conv1b'],
                                        kernel_height, kernel_width,
                                        kernels_num, s_h, s_w, padding="SAME", group=group)
                conv1 = tf.nn.relu(conv1_in)
                
        DEPTH =18
        defs, block_func = cfg[DEPTH]
        
        
        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'):
            logits = (LinearWrap(conv1)
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                      .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', block_func, 128, defs[1], 2)
                      .apply(layer, 'group2', block_func, 256, defs[2], 2)
                      .apply(layer, 'group3', block_func, 512, defs[3], 2)
                      .BNReLU('bnlast')
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000, nl=tf.nn.relu)())
            print logits.get_shape()
            Graph = tf.get_default_graph()
            self.W = Graph.get_tensor_by_name('linear/W:0')
            self.b = Graph.get_tensor_by_name('linear/b:0')
            self.t= Graph.get_tensor_by_name('group1/block0/conv2/W:0')
            print self.t
            self.logits_relu = tf.nn.relu(logits, name='relu')
            self.logits_keep_prob = tf.placeholder_with_default(1.0, tuple(),
                                                                 name='keep_prob_pl')
            self.logit_dropout = tf.nn.dropout(self.logits_relu, self.logits_keep_prob, name='dropout')
            
            
        with tf.variable_scope('fc7'):
            num_inputs = int(self.logit_dropout.get_shape()[1])
            print num_inputs
            layer_index=99
            num_outputs = 1000
            tr_vars['fc7w'], tr_vars['fc7b'] = \
                    self.get_fc_weights(layer_index, net_data, num_inputs, num_outputs)
            layer_index += 1
            self.fc7 = tf.add(tf.matmul(self.logit_dropout, tr_vars['fc7w']), tr_vars['fc7b'],
                                  name='fc')
            if use_batch_norm:
                print 'Using batch_norm after FC7'
                self.fc7_bn = tflayers.batch_norm(self.fc7,decay=0.999,
                                                      is_training=self.is_phase_train,
                                                      trainable=False)
                out = self.fc7_bn
            else:
                out = self.fc7

                self.fc7_relu = tf.nn.relu(out, name='relu')

                self.fc7_keep_prob = tf.placeholder_with_default(1.0, tuple(),
                                                                 name='keep_prob_pl')
                self.fc7_dropout = tf.nn.dropout(self.fc7_relu, self.fc7_keep_prob, name='dropout')

            self.logits = self.fc7
            with tf.variable_scope('output'):
                self.prob = tf.nn.softmax(self.fc7, name='prob')
        self.trainable_vars = tr_vars
            

       
            
             

        

    def restore_from_snapshot(self, snapshot_path, num_layers, restore_iter_counter=False):
        """
        :param snapshot_path: path to the snapshot file
        :param num_layers: number layers to restore from the snapshot
                            (conv1 is the #1, fc8 is the #8)
        :param restore_iter_counter: if True restore global_iter_counter from the snapshot

        WARNING! A call of sess.run(tf.initialize_all_variables()) after restoring from snapshot
                 will overwrite all variables and set them to initial state.
                 Call restore_from_snapshot() only after sess.run(tf.initialize_all_variables())!
        """
        if num_layers > 8 or num_layers < 0:
            raise ValueError('You can restore only 0 to 8 layers.')
        if num_layers == 0:
            print 'Not restoring anything'
            return
        items = self.trainable_vars.items()
        items.sort()
        vars_names_to_restore = [items[i][0] for i in xrange(num_layers * 2)]
        vars_to_restore = [items[i][1] for i in xrange(num_layers * 2)]
        print 'Restoring {} layers from the snapshot: {}'.format(num_layers, vars_names_to_restore)
        if restore_iter_counter:
            try:
                saver = tf.train.Saver(var_list=[self.global_iter_counter])
                saver.restore(self.sess, snapshot_path)
            except:
                print 'Could not restore global_iter_counter.'

        with self.graph.as_default():
            saver = tf.train.Saver(var_list=vars_to_restore)
            saver.restore(self.sess, snapshot_path)

    def get_conv_weights(self, layer_index, net_data, kernel_height, kernel_width,
                         num_input_channels, kernels_num):
        layer_names = ['conv{}'.format(i) for i in xrange(1, 6)] + \
                      ['fc{}'.format(i) for i in xrange(6, 9)]
        wights_std = [0.01] * 5 + [0.005, 0.005, 0.01]
        bias_init_values = [0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0]

        l_name = layer_names[layer_index]
        if net_data is not None and layer_index < self.num_layers_to_init:
            assert net_data[l_name]['weights'].shape == (kernel_height, kernel_width,
                                                 num_input_channels,
                                                 kernels_num)
            assert net_data[l_name]['biases'].shape == (kernels_num,)

        if layer_index >= self.num_layers_to_init or net_data is None:
            print 'Initializing {} with random'.format(l_name)
            w = self.random_weight_variable((kernel_height, kernel_width,
                                             num_input_channels,
                                             kernels_num),
                                            stddev=wights_std[layer_index])
            b = self.random_bias_variable((kernels_num,), value=bias_init_values[layer_index])
        else:
            w = tf.Variable(net_data[l_name]['weights'], name='weight')
            b = tf.Variable(net_data[l_name]['biases'], name='bias')
        return w, b

    def get_fc_weights(self,layer_index,net_data, num_inputs, num_outputs,
                       wights_std=0.01,
                       bias_init_value=0.0):
        print num_inputs
        
        w = self.random_weight_variable((num_inputs, num_outputs),
                                            stddev=wights_std)
        b = self.random_bias_variable((num_outputs,), value=bias_init_value)
        return w,b

    def random_weight_variable(self, shape, stddev=0.01):
        """
        stddev is used only for RandomInitType.GAUSSIAN
        """
        
        if self.random_init_type == Alexnet.RandomInitType.GAUSSIAN:
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial, name='weight')
        elif self.random_init_type == Alexnet.RandomInitType.XAVIER_GAUSSIAN:
            return tf.get_variable("weight", shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       uniform=False))
        elif self.random_init_type == Alexnet.RandomInitType.XAVIER_UNIFORM:
            return tf.get_variable("weight", shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       uniform=True))
        else:
            raise ValueError('Unknown random_init_type')

    @staticmethod
    def random_bias_variable(shape, value=0.1):
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial, name='bias')

    @staticmethod
    def conv(input, kernel, biases, kernel_height, kernel_width,
             kernels_num, s_h, s_w, padding="VALID", group=1):
        """
        From https://github.com/ethereon/caffe-tensorflow
        """
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert kernels_num % group == 0

        def convolve(inp, w, name=None):
            return tf.nn.conv2d(inp, w, [1, s_h, s_w, 1], padding=padding, name=name)

        if group == 1:
            conv = convolve(input, kernel, name='conv')
        else:
            input_groups = tf.split(input, group, 3)
            kernel_groups = tf.split(kernel, group,3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(output_groups,3)
        return tf.reshape(tf.nn.bias_add(conv, biases),
                          [-1] + conv.get_shape().as_list()[1:], name='conv')
    
   
        
        