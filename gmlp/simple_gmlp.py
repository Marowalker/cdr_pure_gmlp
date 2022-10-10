import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout


class gMLPLayer(Layer):
    def __init__(self, dropout_rate=0.1, **kwargs):
        super(gMLPLayer, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.norm = LayerNormalization(epsilon=1e-6)
        self.proj_in = Sequential([
            Dense(units=input_shape[-1] * 2, activation='gelu'),
            Dropout(rate=self.dropout_rate),
        ])
        self.sgu = SpatialGatingUnit(input_shape[-2])
        self.proj_out = Dense(input_shape[-1])

    def call(self, inputs):
        shortcut = self.norm(inputs)
        x = self.proj_in(shortcut)
        x = self.sgu(x)
        x = self.proj_out(x)

        return x + shortcut


class SpatialGatingUnit(Layer):
    def __init__(self, dim_in, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.dim_in = dim_in

    def build(self, input_shape):
        self.norm = LayerNormalization(epsilon=1e-6)
        self.proj = Dense(units=self.dim_in, bias_initializer="Ones")

    def call(self, inputs):
        u, v = tf.split(inputs, 2, axis=-1)

        v = self.norm(v)

        v = tf.linalg.matrix_transpose(v)
        v = self.proj(v)
        v = tf.linalg.matrix_transpose(v)

        return u * v
