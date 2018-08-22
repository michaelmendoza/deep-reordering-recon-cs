
'''
Tensorflow Code for a color segmentation network
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from data_loader import Data
data = Data(percent_load = 1.0, filepath='../mri-data/Cardic_Undersampled_for_CS/training_data_small.h5') 

# Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 

# Load Models
saver = tf.train.import_meta_graph("./tmp/model.meta")
saver.restore(sess,tf.train.latest_checkpoint('./tmp'))

# Get Variables
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
prediction = graph.get_tensor_by_name("Logits/BiasAdd:0")

# Run Prediction
pred_test = sess.run(prediction, feed_dict={ X: data.x_test })

index = 0
new_reorder_test = data.denormalize(pred_test[index, :, :, 1].squeeze(), data.y_min, data.y_max)
plt.imshow(new_reorder_test, cmap='gray', vmin=0, vmax=255)
plt.show()
