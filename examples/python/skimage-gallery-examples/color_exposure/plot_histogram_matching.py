"""
==================
Histogram matching
==================

This example demonstrates the feature of histogram matching. It manipulates the
pixels of an input image so that its histogram matches the histogram of the
reference image. If the images have multiple channels, the matching is done
independently for each channel, as long as the number of channels is equal in
the input image and the reference.

Histogram matching can be used as a lightweight normalisation for image
processing, such as feature matching, especially in circumstances where the
images have been taken from different sources or in different conditions (i.e.
lighting).
"""

import cupy as cp
import matplotlib.pyplot as plt
from skimage import data

from cucim.skimage import exposure
from cucim.skimage.exposure import match_histograms

reference = cp.array(data.coffee())
image = cp.array(data.chelsea())

matched = match_histograms(image, reference, channel_axis=-1)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(cp.asnumpy(image))
ax1.set_title('Source')
ax2.imshow(cp.asnumpy(reference))
ax2.set_title('Reference')
ax3.imshow(cp.asnumpy(matched))
ax3.set_title('Matched')

plt.tight_layout()
plt.show()


######################################################################
# To illustrate the effect of the histogram matching, we plot for each
# RGB channel, the histogram and the cumulative histogram. Clearly,
# the matched image has the same cumulative histogram as the reference
# image for each channel.

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))


for i, img in enumerate((image, reference, matched)):
    for c, c_color in enumerate(('red', 'green', 'blue')):
        img_hist, bins = map(
            cp.asnumpy, exposure.histogram(img[..., c], source_range='dtype')
        )
        axes[c, i].plot(bins, img_hist / img_hist.max())
        img_cdf, bins = map(
            cp.asnumpy, exposure.cumulative_distribution(img[..., c])
        )
        axes[c, i].plot(bins, img_cdf)
        axes[c, 0].set_ylabel(c_color)

axes[0, 0].set_title('Source')
axes[0, 1].set_title('Reference')
axes[0, 2].set_title('Matched')

plt.tight_layout()
plt.show()
