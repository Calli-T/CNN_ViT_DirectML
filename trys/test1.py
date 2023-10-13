import tensorflow as tf
print("TensorFlow version:", tf.__version__)

'''
import os
os.environ["DML_VISIBLE_DEVICES"] = "0"
'''

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

predictions = model(x_train[:1]).numpy()
print(predictions)
#
print(tf.nn.softmax(predictions).numpy())
#
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#
with tf.device('/GPU:0'):
    model.fit(x_train, y_train, epochs=5)