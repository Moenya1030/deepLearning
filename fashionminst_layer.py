import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    # x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.0
    # y = tf.convert_to_tensor(y, dtype=tf.int32)
    # from_tensor_slices自动转tensor了

    x = tf.cast(x, dtype=tf.float32) / 255. #255.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)
#(60000, 28, 28) (60000, )

batchsize = 128 #反正数据集小

db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(10000).batch(batchsize)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batchsize)

db_iter = iter(db)
sample = next(db_iter)
print('batch:', sample[0].shape, sample[1].shape)

# 都是全连接层
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu), # [b, 784] => [b, 256]
    layers.Dense(128, activation=tf.nn.relu), # [b, 256] => [b, 128]
    layers.Dense(64, activation=tf.nn.relu), # [b, 128] => [b, 64]
    layers.Dense(32, activation=tf.nn.relu), # [b, 64] => [b, 32]
    layers.Dense(10) # [b, 32] => [b, 10], params= 330 = 32*10 + 10
])
model.build(input_shape=[None, 28*28]) #[784, 256]
model.summary() # 打印网络结构

# w = w - lr*grad 直接传就好
optimizer = optimizers.Adam(lr=1e-3)

def main():


    for epoch in range(30):


        for step, (x, y) in enumerate(db):

            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x) # 完成一次前向传播
                y_onehot = tf.one_hot(y, depth=10) # 记得one-hot encoding
                # [b]

                # 几种loss
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits)) # [b]
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True) # 为了数据稳定，传logits而不是prob
                # loss_ce只带这行会报错
                loss_ce = tf.reduce_mean(loss_ce)

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # zip把grads, model.trainable_variables 这两个list的元素拼在一起，及梯度在前参数在后


            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_mse), float(loss_ce))


        # test
        total_correct = 0
        total_num = 0
        for x,y in db_test:

            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])
            # [b, 10]
            logits = model(x)
            # logits => prob, [b, 10]
            prob = tf.nn.softmax(logits, axis=1)
            # [b, 10] => [b], int64
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # pred:[b]
            # y: [b]
            # correct: [b], True: equal, False: not equal
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32)) # 令True=1, False=0

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)







if __name__ == '__main__':
    main()
