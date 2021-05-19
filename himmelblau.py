import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import tensorflow as tf

def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
#f(x, y) = (x^2 + y -11)^2 + (x + y^2 - 7)^2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y) #并行生成一系列坐标点
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# [1., 0.], [-4, 0.], [4, 0.]
x = tf.constant([4., 0.])

for step in range(200):

    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)

    grads = tape.gradient(y, [x])[0] #[0]为取[x - grad]的第0号元素
    x -= 0.01 * grads #lr=0.01

    if step % 20 == 0:
        print('step {}: x = {}, f(x) = {}'
              .format(step, x.numpy(), y.numpy()))
