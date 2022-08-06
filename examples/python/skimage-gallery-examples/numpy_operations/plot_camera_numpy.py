"""
Using simple NumPy operations for manipulating images
=====================================================

This script illustrates how to use basic NumPy operations, such as slicing,
masking and fancy indexing, in order to modify the pixel values of an image.
"""

import cupy as cp
import numpy as np
from skimage import data
import matplotlib.pyplot as plt

camera = cp.array(data.camera())
camera[:10] = 0
mask = camera < 87
camera[mask] = 255
inds_x = cp.arange(len(camera))
inds_y = (4 * inds_x) % len(camera)
camera[inds_x, inds_y] = 0

l_x, l_y = camera.shape[0], camera.shape[1]
X, Y = cp.ogrid[:l_x, :l_y]
outer_disk_mask = (X - l_x / 2)**2 + (Y - l_y / 2)**2 > (l_x / 2)**2
camera[outer_disk_mask] = 0

plt.figure(figsize=(4, 4))
plt.imshow(cp.asnumpy(camera), cmap='gray')
plt.axis('off')
plt.show()
