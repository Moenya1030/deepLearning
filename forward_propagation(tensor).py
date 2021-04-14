import tensorflow as tf
from tensorflow import keras # tensorflow集成了keras
from tensorflow.keras import datasets # datasets是keras自带的一个自动管理数据集的工具
import os # 环境变量

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #可以屏蔽tensorflow的一堆奇奇怪怪的东西

### 这次只实现一个最简单的矩阵想加减的功能，不使用任何的优化器

### 加载数据集
(x, y) , _ = datasets.mnist.load_data() # 加载minst数据集，可以自动加载
# 这里 x: [60k, 28, 28], y: [60k]
# 将数据转化乘一个tensor
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
# 这里的输出是 (60000, 28, 28) (60000,) <dtype: 'float32'> <dtype: 'int32'>
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y)) # 看一下x和y的最小值和最大值
# tf.Tensor(0.0, shape=(), dtype=float32) tf.Tensor(255.0, shape=(), dtype=float32)
# tf.Tensor(0, shape=(), dtype=int32) tf.Tensor(9, shape=(), dtype=int32)

### 创建数据集，可以很方便地取出batch
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
# batch是批。深度学习每一次参数的更新所需要损失函数并不是由一个{data：label}获得的，而是由一组数据加权得到的，这一组数据的数量就是[batch size]
train_iter = iter(train_db) #创建迭代器，可以方便地next取得想要的元素
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape) # 数组都有一个shape属性，它是一个元组，返回各个维度的维数
# batch: (128, 28, 28) (128,)


### 创建权值
# [b, 784] => [b, 256] => [b, 128] => [b, 10] 降维的过程
# [dim_in, dim_out] w1 , [dim_out] b1 这里设定的是shape，要符合矩阵运算的规则

#这里为了解决梯度爆炸,设定了方差等于0.01
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev = 0.01)) # 第一层
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev = 0.01))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev = 0.01))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3 # learning rate

for epoch in range(10): #对数据集循环迭代十次

    ### 前向运算
    # 这里的x就是之前的那个batch
    for step, (x, y) in enumerate(train_db):  # 有目的的进行打印，每进行一个batch叫做step
        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28 * 28])  # 维度变换，可以理解为一个数据预处理

        with tf.GradientTape() as tape:  # tf自带的自动求导工具要带这一句,可以使得梯度信息被记录下来
            # 默认只跟踪tf.Variable这个类型，如果不是这个类型的，就不会自动记录梯度信息
            # 所以要把tf.tensor封装成tf.Variable

            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b, 784] @ [784, 256] + [256] => [b, 256] + [256] // 其实这里的加号可以自动做一个broadcast
            h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])  # 把原来一个[256]的shape变成了一个[b ,256]的shape
            # [b, 256] + [256] => [b, 256] + [b, 256]
            h1 = tf.nn.relu(h1)  # 把它非线性化

            h2 = h1 @ w2 + b2  # [b, 256] => [b, 128]
            h2 = tf.nn.relu(h2)

            # 这里是最后一层了，所有直接用out，然后不加relu
            # h3 = h2@w3 + b3 # [b, 256] => [b, 10]
            out = h2 @ w3 + b3

            ### compute loss
            # out: [b, 10]， 但是真实的y的shape就是一个 [b] ,要做一个均方差
            # y: [b] => [b, 10] 也可以在数据集加载的时候进行onehot encoding，只需要在计算loss前完成即可
            y_onehot = tf.one_hot(y, depth=10)
            # onehot 其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都有它独立的寄存器位，并且在任意时候，其中只有一位有效

            # 均方差 mse = mean(sum(y - out)^2)
            loss = tf.square(y_onehot - out)  # [b, 10]
            # mean: scalar
            loss = tf.reduce_mean(loss)

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])  # 会返回跟后面这个wb相对应的list
        # 更新梯度
        # w1 = w1 - lr * w1_grad ,lr是learning rate
        w1.assign_sub(lr * grads[0])  # 原地更新
        # w1 = w1 - lr * grads[0] # w1处于第0个位置，所有grads也在第零个位置
        # TypeError: unsupported operand type(s) for *: 'float' and 'NoneType' 后面这个grads是个none类型
        # 然后因为减去了lr * grads， w1从tf.Variable又变成了tf.tensor,所以后面梯度就不会自动跟踪了，所以需要原地更新
        # b1 = b1 - lr * grads[1]
        # w2 = w2 - lr * grads[2]
        # b2 = b2 - lr * grads[3]
        # w3 = w3 - lr * grads[4]
        # b3 = b3 - lr * grads[5]
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))  # tensor转换为numpy的float
