"""
Implements PointNet.

See https://arxiv.org/pdf/1612.00593.pdf

Adapted from https://keras.io/examples/vision/pointnet/

Authors: Emilio Villasana, Andrew Rice, Raghu Ramanujan, Dylan Sparks
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package='Custom', name='OrthogonalRegularizer')
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """
    A reimplementation of orthogonal regularization. This incentivizes the rows of 
    the matrix to be orthogonal to each other.
    
    Is not directly replaceable with the tensorflow implementation of this regularizer
    because this version regularizes matrix shaped output from a layer.
    """
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - tf.eye(self.num_features)))
    
    def get_config(self):
        return {'l2reg': self.l2reg,
                'num_features': self.num_features}

    
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def tnet(inputs, num_features):
    """ Layer sizes are from Appendix C of the PointNet paper """
    x = conv_bn(inputs, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)

    # Initialise bias as the identity matrix (based on section C from Appendix)
    bias = keras.initializers.Constant(np.eye(num_features).flatten())

    # Incentivize rows of matrix to be orthogonal (Eq. 2 from page 4)
    reg = OrthogonalRegularizer(num_features)

    x = layers.Dense(num_features * num_features,
                     kernel_initializer="zeros",
                     bias_initializer=bias,
                     activity_regularizer=reg)(x)

    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


def create_shared_layers(inputs, num_dimensions):
    """ Implements shared layers from Fig. 2 """
    x = tnet(inputs, num_dimensions)
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    local_features = tnet(x, 64)
    x = conv_bn(local_features, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    global_features = layers.GlobalMaxPooling1D()(x)
    
    return local_features, global_features


def create_event_wise_head(global_features, num_classes, is_regression):
    """ Implements classification layers from Fig. 2 (but final head could be regression) """
    x = dense_bn(global_features, 512)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    
    if not is_regression:
        return layers.Dense(num_classes, activation="softmax")(x)
    return layers.Dense(1, activation="linear")(x)    


def create_point_wise_head(local_features, global_features, num_points, num_classes, is_regression):
    """ Implements segmentation layers from Fig. 2 (the yellow shaded region) """
    # x = tf.expand_dims(global_features, axis=1)
    # x = tf.repeat(x, repeats=num_points, axis=1)
    # x = layers.Reshape((1, global_features.shape[-1]))(global_features)
    x = layers.RepeatVector(num_points)(global_features)
    concat_features = layers.Concatenate(axis=2)([local_features, x])
    x = conv_bn(concat_features, 512)
    x = conv_bn(x, 256)
    x = conv_bn(x, 128)
    x = layers.Dropout(0.3)(x)
    x = conv_bn(x, 128)
    x = layers.Dropout(0.3)(x)
    
    if not is_regression:
        return layers.Dense(num_classes, activation="softmax")(x)
    return layers.Dense(1, activation="linear")(x)
 
    
def create_pointnet_model(num_points, 
                          num_classes, 
                          num_dimensions=3, #for changing number of features
                          is_regression=False,
                          is_pointwise_prediction=True):
    inputs = keras.Input(shape=(num_points, num_dimensions))
    local_features, global_features = create_shared_layers(inputs, num_dimensions)
    
    if not is_pointwise_prediction:
        outputs = create_event_wise_head(global_features, num_classes, is_regression)
    else:
        outputs = create_point_wise_head(local_features, 
                                         global_features, 
                                         num_points, 
                                         num_classes,
                                         is_regression)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="PointNet")
