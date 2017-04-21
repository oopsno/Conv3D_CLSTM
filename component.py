from __future__ import print_function
import keras as k
from keras.activations import relu as ReLU
import tensorflow as tf
from convlstm.cell import ConvLSTMCell


def Conv3D(filters=None):
    def __conv_3d__(x):
        return k.layers.Conv3D(filters=filters, kernel_size=[3, 3, 3], padding='same')(x)

    return __conv_3d__


def BatchNorm(x):
    norm = k.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    return k.activations.relu(norm)


def Pooling2D(xs):
    def __pooling_2d__(x):
        return k.layers.MaxPooling2D(pool_size=xs, padding='same')(x)

    return __pooling_2d__


def FlattenedMaxPooling2D(x, s):
    import functools
    with tf.name_scope('FlattenedMaxPooling_{}x{}'.format(s, s)):
        pool = Pooling2D([s, s])(x)
        flattened_dim = functools.reduce(int.__mul__, pool.shape.as_list()[1:])
        return tf.reshape(pool, [-1, flattened_dim])


def Pooling3D(xs):
    def __pooling_3d__(x):
        return k.layers.MaxPooling3D(pool_size=xs, strides=xs, padding='same')(x)

    return __pooling_3d__


def component_3d_cnn(x):
    Conv3D_256 = Conv3D(filters=256)
    Conv3D_128 = Conv3D(filters=128)
    Conv3D_64  = Conv3D(filters=64)
    Pooling3D_222 = Pooling3D([2, 2, 2])
    Pooling3D_122 = Pooling3D([1, 2, 2])

    with tf.name_scope('CNN-BN-1'):  # 3D CNN-BN Layer 1
        l1 = Pooling3D_122(
             ReLU(
             BatchNorm(
             Conv3D_64(x))))
    with tf.name_scope('CNN-BN-2'):  # 3D CNN-BN Layer 2
        l2 = Pooling3D_222(
             ReLU(
             BatchNorm(
             Conv3D_128(l1))))
    with tf.name_scope('CNN-BN-3'):  # 3D CNN-BN Layer 3
        l3 = ReLU(
             BatchNorm(
             Conv3D_256(
             Conv3D_256(l2))))
    return l3


def component_conv_lstm(x):
    with tf.name_scope('ConvLSTM-1') as s1:
        cell256 = ConvLSTMCell(x.shape.as_list()[2:4], 256, [3, 3])
        stack_1, _ = tf.nn.dynamic_rnn(cell256, x, dtype=x.dtype, time_major=True, scope=s1)
    with tf.name_scope('ConvLSTM-2') as s2:
        cell384 = ConvLSTMCell(stack_1.shape.as_list()[2:4], 384, [3, 3])
        stack_2, _ = tf.nn.dynamic_rnn(cell384, stack_1, dtype=x.dtype, time_major=True, scope=s2)
        return stack_2[:, -1, ...]


def component_spp(x):
    with tf.name_scope('SPP'):
        return k.layers.concatenate([
            FlattenedMaxPooling2D(x, 4),
            FlattenedMaxPooling2D(x, 7),
            FlattenedMaxPooling2D(x, 14),
            FlattenedMaxPooling2D(x, 28)
        ])


def component_score(x, classes):
    with tf.name_scope('Score'):
        fc = k.layers.Dense(classes)
        dropout = k.layers.Dropout(rate=0.5)
        return dropout(fc(x))


def network(x, classes=None, reuse=None):
    assert classes is not None
    assert reuse is not None
    with tf.variable_scope('Network', reuse=reuse):
        return component_score(component_spp(component_conv_lstm(component_3d_cnn(x))), classes)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    k.backend.set_session(sess)

    inputs = k.layers.Input([32, 112, 112, 3])
    print('Input    =>', inputs.shape)

    cnn = component_3d_cnn(inputs)
    print('3D-CNN   =>', cnn.shape)

    lstm = component_conv_lstm(cnn)
    print('ConvLSTM =>', lstm.shape)

    spp = component_spp(lstm)
    print('SPP      =>', spp.shape)

    score = component_score(spp, 42)
    print('Score    =>', score.shape)

    tf.summary.FileWriter('model', sess.graph).add_graph(sess.graph)
