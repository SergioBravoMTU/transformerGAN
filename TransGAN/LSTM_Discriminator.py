import tensorflow as tf
from tensorflow.keras import layers
def lstm_Discriminator(maxlen, vocab_size, name='Discriminator'):

    inputs = tf.keras.Input(shape=(maxlen, vocab_size))
    lstm_output = layers.LSTM(units=64)(inputs)
    dropout_layer = layers.Dropout(0.1)(lstm_output)
    dense_layer = layers.Dense(128, activation="relu")(dropout_layer)
    dense_layer2 = layers.Dense(32, activation="relu")(dense_layer)
    outputs = layers.Dense(2, activation="softmax")(dense_layer2)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)