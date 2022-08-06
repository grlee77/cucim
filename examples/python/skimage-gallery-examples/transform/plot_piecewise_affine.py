"""
===============================
Piecewise Affine Transformation
===============================

This example shows how to use the Piecewise Affine Transformation.

"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from cucim.skimage.transform import PiecewiseAffineTransform, warp



image = cp.array(data.astronaut())
rows, cols = image.shape[0], image.shape[1]

src_cols = cp.linspace(0, cols, 20)
src_rows = cp.linspace(0, rows, 10)
src_rows, src_cols = cp.meshgrid(src_rows, src_cols)
src = cp.stack([src_cols.ravel(), src_rows.ravel()], axis=-1)

# add sinusoidal oscillation to row coordinates
dst_rows = src[:, 1] - cp.sin(cp.linspace(0, 3 * np.pi, src.shape[0])) * 50
dst_cols = src[:, 0]
dst_rows *= 1.5
dst_rows -= 1.5 * 50
dst = cp.stack([dst_cols, dst_rows], axis=-1)


tform = PiecewiseAffineTransform()
tform.estimate(src, dst)

out_rows = image.shape[0] - 1.5 * 50
out_cols = cols
out = warp(image, tform, output_shape=(out_rows, out_cols))

fig, ax = plt.subplots()
ax.imshow(cp.asnumpy(out))
ax.plot(
    cp.asnumpy(tform.inverse(src)[:, 0]),
    cp.asnumpy(tform.inverse(src)[:, 1]),
    '.b'
)
ax.axis((0, out_cols, out_rows, 0))
plt.show()
