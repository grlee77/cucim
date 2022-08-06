"""
====================
Finding local maxima
====================

The ``peak_local_max`` function returns the coordinates of local peaks (maxima)
in an image. Internally, a maximum filter is used for finding local maxima. This
operation dilates the original image and merges neighboring local maxima closer
than the size of the dilation. Locations where the original image is equal to the
dilated image are returned as local maxima.

"""
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy import ndimage as ndi
from skimage import data

from cucim.skimage import img_as_float
from cucim.skimage.feature import peak_local_max


im = img_as_float(cp.array(data.coins()))

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(im, size=20, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(im, min_distance=20)

# display results
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(cp.asnumpy(im), cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(cp.asnumpy(image_max), cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Maximum filter')

ax[2].imshow(cp.asnumpy(im), cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(cp.asnumpy(coordinates[:, 1]), cp.asnumpy(coordinates[:, 0]), 'r.')
ax[2].axis('off')
ax[2].set_title('Peak local max')

fig.tight_layout()

plt.show()
