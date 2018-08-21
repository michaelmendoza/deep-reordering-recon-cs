
'''
Tensorflow Code for a color segmentation network
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from data_loader import Data
data = Data(percent_load = 0.005)

# Import Models
from model import unet

# Training Parameters
learning_rate = 0.0001
num_steps = 100
batch_size = 32
display_step = 10

# Network Parameters 
WIDTH = 256; HEIGHT = 256; CHANNELS = 2
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS = 2 

# Network Varibles and placeholders
X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS])  # Input
Y = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_OUTPUTS]) # Truth Data - Output

# Define loss and optimizer
logits, _ = unet(X, NUM_OUTPUTS) 

prediction = logits
loss = tf.reduce_mean(tf.square(prediction - Y)) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Initalize varibles, and run network 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print ('Start Training: BatchSize:', batch_size,' LearningRate:', learning_rate)

# Train network
_step = []
_train_loss = []
_test_loss = []

for step in range(num_steps):
    batch_xs, batch_ys = data.next_batch(batch_size)
    sess.run( trainer, feed_dict={ X: batch_xs, Y: batch_ys } )

    if(step % display_step == 0):
        train_loss = sess.run(loss, feed_dict={ X: batch_xs, Y: batch_ys })   
        test_loss = sess.run(loss, feed_dict={ X: data.x_test, Y: data.y_test })
        _step.append(step)
        _train_loss.append(train_loss)
        _test_loss.append(test_loss)
        print("Step: " + str(step) + " Train Loss: " + str(train_loss) + " Test Loss: " + str(test_loss))

pred_test = sess.run(prediction, feed_dict={ X: data.x_test })

# Plot Accuracy
plt.plot(_step, _train_loss, label="Train Loss")
plt.plot(_step, _test_loss, label="Test Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss for reordering")
plt.savefig('results/loss.png')

# Show results
index = 0
true_reorder = data.denormalize(data.x_test[index, :, :, 1].squeeze(), data.y_min, data.y_max)
new_reorder = data.denormalize(pred_test[index, :, :, 1].squeeze(), data.y_min, data.y_max)
diff = true_reorder - new_reorder

matplotlib.image.imsave('results/true_reorder.png', true_reorder, cmap='gray') 
matplotlib.image.imsave('results/new_reorder.png', new_reorder, cmap='gray') 
matplotlib.image.imsave('results/diff_reorder.png', diff, cmap='gray') 

subplot(3,1,1)
plt.imshow(true_reorder, cmap='gray')
subplot(3,1,2)
plt.imshow(new_reorder, cmap='gray')
subplot(3,1,3)
plt.imshow(diff, cmap='gray')
plt.show() 