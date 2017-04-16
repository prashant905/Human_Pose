import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack import *
from tensorpack.callbacks.dump import DumpParamAsImage
from tensorpack.tfutils.summary import *
import tensorflow as tf
import numpy as np

sigma = 32.0
BATCH_SIZE = None
DEPTH =18

def pdf_debug_img(name, float_image, sigma):
    # max_val = 1.0 / (np.sqrt(2 * (sigma ** 2) * np.pi))
    max_val = tf.reduce_max(float_image)
    float_image = tf.maximum(float_image, np.float(0))
    debug = tf.cast(255 * (float_image / max_val), tf.uint8)
    return tf.summary.image(name, debug, max_outputs=5)


def gaussian_image(label):
    indices = np.indices([368, 368])[:, ::8, ::8].astype(np.float32)
    coords = tf.constant(indices)
    stretch = tf.reshape(tf.to_float(label), [-1, 2, 1, 1])
    stretch = tf.tile(stretch, [1, 1, 46, 46])
    # pdf = 1.0/(np.sqrt(2*(sigma**2)*np.pi)) * tf.exp(-tf.pow(coords-stretch,2)/(2*sigma**2))
    pdf = tf.pow(coords - stretch, 2) / (2 * sigma ** 2)
    pdf = tf.reduce_sum(pdf, [1])
    # pdf = tf.reduce_prod(pdf,[1])
    # print debug
    pdf = tf.expand_dims(pdf, 3)
    debug = tf.exp(-pdf)  # 1.0 / (np.sqrt(2 * (sigma ** 2) * np.pi)) *
    pdf_debug_img('super', debug, sigma)

    return debug


class Model(ModelDesc):
    def __init__(self):
        super(Model, self).__init__()

    def _get_inputs(self):
        return [InputDesc(tf.float32, [BATCH_SIZE, 368, 368, 3], 'input'),
                InputDesc(tf.int32, [BATCH_SIZE, 2], 'label')
                ]
    def _get_optimizer(self):
        lr = tf.Variable(1E-3, trainable=False, name='learning_rate')
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9)
    
    def _build_graph(self, input_vars):
        is_training =True
        image, label = input_vars
        if is_training:
            tf.summary.image("train_image", image, 10)
            
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

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

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
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = cfg[DEPTH]


        pred = (LinearWrap(image)
                .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                .apply(layer, 'group1', block_func, 128, defs[1], 2)())
                #.apply(layer, 'group2', block_func, 256, defs[2], 2)
                #.apply(layer, 'group3', block_func, 512, defs[3], 2)
                #.BNReLU('bnlast')
                #.GlobalAvgPooling('gap')
                #.FullyConnected('linear', 1000, nl=tf.identity)())

        def add_stage(stage, l):
            l = tf.concat(3, [l, shared])
            for i in range(1, 6):
                l = Conv2D('Mconv{}_stage{}'.format(i, stage), l, 128, kernel_shape=7)
            l = Conv2D('Mconv6_stage{}'.format(stage), l, 128, kernel_shape=1)
            l = Conv2D('Mconv7_stage{}'.format(stage), l, BODY_PART_COUNT, kernel_shape=1, nl=tf.identity)
            pred = tf.transpose(l, perm=[0, 3, 1, 2])
            pred = tf.reshape(pred, [-1, 46, 46, 1])
            error = tf.squared_difference(pred, gaussian, name='se_{}'.format(stage))
            return l, error

        """
        pred = LinearWrap(image) \
            .MaxPooling('pool1', 2, stride=2, padding='SAME') \
            .MaxPooling('pool2', 2, stride=2, padding='SAME') \
            .MaxPooling('pool3', 2, stride=2, padding='SAME') \
            .Conv2D('conv4.4', out_channel=1, kernel_shape=1, nl=tf.identity)()
        """

        # debug_pred = 1.0 / (np.sqrt(2 * (sigma ** 2) * np.pi)) * tf.exp(-pred)
        # pred = tf.reshape(tf.nn.softmax(tf.reshape(pred,[5,64*64])),[5,64,64,1])
        belief_maps_output = tf.identity(pred, "belief_maps_output")
        pdf_debug_img('pred', pred, sigma)

        gaussian = gaussian_image(label)
        # diff = (pred - gaussian)
        # dbg = tf.reduce_sum(tf.to_float(tf.is_nan(gaussian)))
        cost = tf.squared_difference(pred, gaussian, name='l2_norm')
        pdf_debug_img('cost', cost, sigma)

        cost = tf.reduce_mean(cost, name='mse')


        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = symbf.rms(gaussian - pred, name='wrong')
        # monitor training error
        #add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.multiply(0.000001,
                         regularize_cost('conv.*/W', tf.nn.l2_loss),
                         name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])  # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')
