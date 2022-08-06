"""
==================================================
Comparing edge-based and region-based segmentation
==================================================

In this example, we will see how to segment objects from a background. We use
the ``coins`` image from ``skimage.data``, which shows several coins outlined
against a darker background.
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from cucim.skimage import morphology
from cucim.skimage.color import label2rgb
from cucim.skimage.exposure import histogram
from cucim.skimage.feature import canny
from cucim.skimage.filters import sobel
from cupyx.scipy import ndimage as ndi

coins = cp.array(data.coins())
hist, hist_centers = histogram(coins)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(cp.asnumpy(coins), cmap=plt.cm.gray)
axes[0].axis('off')
axes[1].plot(cp.asnumpy(hist_centers), cp.asnumpy(hist), lw=2)
axes[1].set_title('histogram of gray values')

######################################################################
#
# Thresholding
# ============
#
# A simple way to segment the coins is to choose a threshold based on the
# histogram of gray values. Unfortunately, thresholding this image gives a
# binary image that either misses significant parts of the coins or merges
# parts of the background with the coins:

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

axes[0].imshow(cp.asnumpy(coins > 100), cmap=plt.cm.gray)
axes[0].set_title('coins > 100')

axes[1].imshow(cp.asnumpy(coins > 150), cmap=plt.cm.gray)
axes[1].set_title('coins > 150')

for a in axes:
    a.axis('off')

plt.tight_layout()

######################################################################
# Edge-based segmentation
# =======================
#
# Next, we try to delineate the contours of the coins using edge-based
# segmentation. To do this, we first get the edges of features using the
# Canny edge-detector.

edges = canny(coins)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(cp.asnumpy(edges), cmap=plt.cm.gray)
ax.set_title('Canny detector')
ax.axis('off')

######################################################################
# These contours are then filled using mathematical morphology.


fill_coins = ndi.binary_fill_holes(edges)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(cp.asnumpy(fill_coins), cmap=plt.cm.gray)
ax.set_title('filling the holes')
ax.axis('off')


######################################################################
# Small spurious objects are easily removed by setting a minimum size for
# valid objects.

coins_cleaned = morphology.remove_small_objects(fill_coins, 21)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(cp.asnumpy(coins_cleaned), cmap=plt.cm.gray)
ax.set_title('removing small objects')
ax.axis('off')

######################################################################
# However, this method is not very robust, since contours that are not
# perfectly closed are not filled correctly, as is the case for one unfilled
# coin above.
#
# Region-based segmentation
# =========================
#
# We therefore try a region-based method using the watershed transform.
# First, we find an elevation map using the Sobel gradient of the image.

elevation_map = sobel(coins)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(cp.asnumpy(elevation_map), cmap=plt.cm.gray)
ax.set_title('elevation map')
ax.axis('off')

######################################################################
# Next we find markers of the background and the coins based on the extreme
# parts of the histogram of gray values.

markers = cp.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(cp.asnumpy(markers), cmap=plt.cm.nipy_spectral)
ax.set_title('markers')
ax.axis('off')

# TODO: use cuCIM's watershed once implemented

######################################################################
# Finally, we use the watershed transform to fill regions of the elevation
# map starting from the markers determined above:
from skimage import segmentation

segmentation_coins = segmentation.watershed(cp.asnumpy(elevation_map),
                                            cp.asnumpy(markers))

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(segmentation_coins, cmap=plt.cm.gray)
ax.set_title('segmentation')
ax.axis('off')

######################################################################
# This last method works even better, and the coins can be segmented and
# labeled individually.
segmentation_coins = cp.asarray(segmentation_coins)
segmentation_coins = ndi.binary_fill_holes(segmentation_coins - 1)
labeled_coins, _ = ndi.label(segmentation_coins)
image_label_overlay = label2rgb(labeled_coins, image=coins, bg_label=0)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(cp.asnumpy(coins), cmap=plt.cm.gray)
axes[0].contour(cp.asnumpy(segmentation_coins), [0.5], linewidths=1.2, colors='y')
axes[1].imshow(cp.asnumpy(image_label_overlay))

for a in axes:
    a.axis('off')

plt.tight_layout()

plt.show()
