#! /usr/bin/env python
# coding=utf-8


import core.common as common
import tensorflow as tf


slim = tf.contrib.slim


class Layer:
    # stem_block
    def _stem_block(self, input_x, num_init_channel=32, is_training=True, reuse=False):
        block_name = 'stem_block'
        with tf.variable_scope(block_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                normalizer_fn=slim.batch_norm,
                                activation_fn=tf.nn.relu) as s:
                conv0 = slim.conv2d(input_x, num_init_channel, 3, 2, scope='stem_block_conv0')

                conv1_l0 = slim.conv2d(conv0, int(num_init_channel / 2), 1, 1, scope='stem_block_conv1_l0')
                conv1_l1 = slim.conv2d(conv1_l0, num_init_channel, 3, 2, scope='stem_block_conv1_l1')

                maxpool1_r0 = slim.max_pool2d(conv0, 2, 2, padding='SAME', scope='stem_block_maxpool1_r0')

                filter_concat = tf.concat([conv1_l1, maxpool1_r0], axis=-1)

                output = slim.conv2d(filter_concat, num_init_channel, 1, 1, scope='stem_block_output')

            return output

    def _dense_block(self, input_x, stage, num_block, k, bottleneck_width, is_training=True, reuse=False):
        with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu) as s:
            output = input_x

            for index in range(num_block):
                dense_block_name = 'stage_{}_dense_block_{}'.format(stage, index)
                with tf.variable_scope(dense_block_name) as scope:
                    if reuse:
                        scope.reuse_variables()

                    inter_channel = k * bottleneck_width
                    # left channel
                    conv_left_0 = slim.conv2d(output, inter_channel, 1, 1, scope='conv_left_0')
                    conv_left_1 = slim.conv2d(conv_left_0, k, 3, 1, scope='conv_left_1')
                    conv_left_1  = common.cbam_block(conv_left_1 , "cbam_left")
                    # right channel
                    conv_right_0 = slim.conv2d(output, inter_channel, 1, 1, scope='conv_right_0')
                    conv_right_1 = slim.conv2d(conv_right_0, k, 3, 1, scope='conv_right_1')
                    conv_right_2 = slim.conv2d(conv_right_1, k, 3, 1, scope='conv_right_2')
                    conv_right_2 = common.cbam_block(conv_right_2, "cbam_right")

                    output = tf.concat([output, conv_left_1, conv_right_2], axis=3)


            return output

    def _transition_layer(self, input_x, stage, output_channel, is_avgpool=True, is_training=True, reuse=False):
        transition_layer_name = 'stage_{}_transition_layer'.format(stage)

        with tf.variable_scope(transition_layer_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                normalizer_fn=slim.batch_norm,
                                activation_fn=tf.nn.relu) as s:
                conv0 = slim.conv2d(input_x, output_channel, 1, 1, scope='transition_layer_conv0')
                if is_avgpool:
                    is_training = tf.cast(True, tf.bool)
                    output = common.separable_conv('transition_layer_avgpool', conv0, output_channel, output_channel,
                                                   is_training, downsample=True)
                    # output = common.convolutional(conv0, filters_shape=(3, 3, output_channel, output_channel),
                    #                                   trainable=is_training, name='transition_layer_avgpool', downsample=True)
                    # output = slim.avg_pool2d(conv0, 2, 2, scope='transition_layer_avgpool')
                else:
                    output = conv0
            output = common.cbam_block(output, "output_0")
            return output


def peleetnet_yolov3(input_x,trainable):

    layer = Layer()

    input_x = common.convolutional(input_x, filters_shape=(1, 1, 3, 3), trainable=trainable, name='conv0')


    stem_block_output = layer._stem_block(input_x, 32)

    # stem_block_output = common.convolutional(stem_block_output, filters_shape=(3, 3, 32, 64), trainable=trainable, name='conv1')
    #
    # stem_block_output = common.convolutional(stem_block_output, filters_shape=(1, 1, 64, 32), trainable=trainable, name='conv2')

    dense_block_output = layer._dense_block(stem_block_output, 0, 3, 16, 2)

    transition_layer_output = layer._transition_layer(dense_block_output, 0, 128)

    dense_block_output1 = layer._dense_block(transition_layer_output, 1, 4, 16, 2)

    transition_layer_output1 = layer._transition_layer(dense_block_output1, 1, 256)

    dense_block_output2 = layer._dense_block(transition_layer_output1, 2, 8, 16, 4)

    transition_layer_output2 = layer._transition_layer(dense_block_output2, 2, 512)

    dense_block_output3 = layer._dense_block(transition_layer_output2, 3, 16, 16, 4)

    transition_layer_output3 = layer._transition_layer(dense_block_output3, 3, 1024, is_avgpool=False)

    return dense_block_output1, dense_block_output2, transition_layer_output3


def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data




