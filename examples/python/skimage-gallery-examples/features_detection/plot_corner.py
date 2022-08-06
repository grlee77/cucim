"""
================
Corner detection
================

Detect corner points using the Harris corner detector and determine the
subpixel position of corners ([1]_, [2]_).

.. [1] https://en.wikipedia.org/wiki/Corner_detection
.. [2] https://en.wikipedia.org/wiki/Interest_point_detection

"""
import cupy as cp
from matplotlib import pyplot as plt
from skimage import data
from skimage.draw import ellipse
from skimage.feature import corner_subpix

from cucim.skimage.feature import corner_harris, corner_peaks
from cucim.skimage.transform import warp, AffineTransform


# Sheared checkerboard
tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7,
                        translation=(110, 30))
board = cp.array(data.checkerboard()[:90, :90])
image = warp(board, tform.inverse, output_shape=(200, 310))
# Ellipse
rr, cc = ellipse(160, 175, 10, 100)
image[rr, cc] = 1
# Two squares
image[30:80, 200:250] = 1
image[80:130, 250:300] = 1

coords = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)

# TODO: implement corner_subpix on the GPU
image = cp.asnumpy(image)
coords = cp.asnumpy(coords)
coords_subpix = corner_subpix(image, coords, window_size=13)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
ax.axis((0, 310, 200, 0))
plt.show()
