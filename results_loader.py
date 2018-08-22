
import numpy as np
import matplotlib.pyplot as plt

new_reorder_test = np.load('results/new_reorder_test.npy')
new_reorder = np.load('results/new_reorder.npy')

plt.subplot(2,1,1)
plt.imshow(new_reorder_test, cmap='gray', vmin=0, vmax=255)
plt.subplot(2,1,2)
plt.imshow(new_reorder, cmap='gray', vmin=0, vmax=255)
plt.show()
