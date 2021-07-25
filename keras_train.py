import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras


def preprocess(x, y):
    # [0~255] => [-1~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1. # 直接从0~1可能效果不会很好
    y = tf.cast(y, dtype=tf.int32)
    return x,y

batchsz = 128
# [50k, 32, 32, 3], 返回的是[10k, 1]的shape，要把这个1的维度给消去
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
y = tf.squeeze(y) # 见上
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10) # [50k, 10]
y_val = tf.one_hot(y_val, depth=10) # [10k, 10]
print('datasets:', x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())
# datasets: (50000, 32, 32, 3) (50000, 10) (10000, 32, 32, 3) (10000, 10) 0 255

train_db = tf.compat.v1.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.compat.v1.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[1].shape)
# batch: (128, 32, 32, 3) (128, 10)

# 新建网络对象

class MyDense(layers.Layer):
    # to replace standard layers.Dense()
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_weight('w', [inp_dim, outp_dim])  # 不要写成[inp_dim], [outp_dim])
        # 可以把add_Variable换成add_weight
       #  self.bias = self.add_variable('b', [out_dim])  #要实现的dense层是没有bias参数的，这里自己设计


    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return  x

class MyNetwork(keras.Model): # 不是models
    # 五层
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(32 * 32 * 3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)


    def call(self, inputs, training=None):
        """

        :param inputs:[b, 32, 32, 3]
        :param training:
        :return:
        """
        x = tf.reshape(inputs, [-1, 32 * 32 * 3])
        # [b, 32 * 32 * 3] => [b, 256]
        x = self.fc1(x)
        # 因为没有激活函数，所以要自己写
        x = tf.nn.relu(x) # 也可以写在fc1里面
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)

        return x # 作为logits返回

# 进行装配
network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']) # optimizer不要写成optimizers了
#network.fit(train_db, eopchs = 10, validation_data = test_db, validation_freq = 1)
network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)
# 草 epochs写成了eopchs

# 写一个模型的保存，优先写权值的给保存
network.evaluate(test_db) #确认是否保存
network.save_weights('ckpt/weights.ckpt')
del network
print('saved to ckpt/weights.ckpt')

# 重新创建
network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']) # optimizer不要写成optimizers了
network.load_weights('ckpt/weights.ckpt')
print('loaded weights from file.')
network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)
# loss: 1.0239 - accuracy: 0.6461 - val_loss: 1.7382 - val_accuracy: 0.5037 产生了过拟合
