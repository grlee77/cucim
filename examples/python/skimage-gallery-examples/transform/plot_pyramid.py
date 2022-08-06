"""
====================
Build image pyramids
====================

The ``pyramid_gaussian`` function takes an image and yields successive images
shrunk by a constant scale factor. Image pyramids are often used, e.g., to
implement algorithms for denoising, texture discrimination, and scale-invariant
detection.

"""
import cupy as cp
import matplotlib.pyplot as plt
from skimage import data

from cucim.skimage.transform import pyramid_gaussian


image = cp.array(data.astronaut())
rows, cols, dim = image.shape
pyramid = tuple(pyramid_gaussian(image, downscale=2, channel_axis=-1))

composite_image = cp.zeros((rows, cols + cols // 2, 3), dtype=cp.double)

composite_image[:rows, :cols, :] = pyramid[0]

i_row = 0
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig, ax = plt.subplots()
ax.imshow(cp.asnumpy(composite_image))
plt.show()
