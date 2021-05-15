import tensorflow as tf
from 	tensorflow import keras

x = tf.random.normal([2, 3])

model = keras.Sequential([
		keras.layers.Dense(2, activation='relu'),
		keras.layers.Dense(2, activation='relu'),
		keras.layers.Dense(2)
	])
model.build(input_shape=[None, 3])
model.summary()

for p in model.trainable_variables:
	print(p.name, p.shape)

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 2)                 8 （3行2列）         
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 6  （2*2+2）       
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 6   （2*2+2）      
=================================================================
Total params: 20
Trainable params: 20
Non-trainable params: 0
_________________________________________________________________
dense/kernel:0 (3, 2)
dense/bias:0 (2,)
dense_1/kernel:0 (2, 2)
dense_1/bias:0 (2,)
dense_2/kernel:0 (2, 2)
dense_2/bias:0 (2,)
'''
