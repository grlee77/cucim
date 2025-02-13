import math
import warnings
from collections.abc import Sequence

import cupy as cp
import numpy as np

from cucim.skimage._shared.utils import _ndarray_argwhere
from cucim.skimage._vendored import ndimage as ndi

try:
    from cupyx.scipy.spatial.distance import cdist

    have_cdist = True
except ImportError:
    have_cdist = False

convex_deps = dict()
convex_deps["image_convex"] = ["image"]  # computed by regionprops_image
convex_deps["area_convex"] = ["image_convex"]
convex_deps["feret_diameter_max"] = ["image_convex"]
convex_deps["solidity"] = ["area", "area_convex"]


def pdist_max_blockwise(
    coords,
    metric="sqeuclidean",
    *,
    return_argmax=False,
    coords_per_block=4000,
    cdist_kwargs={},
):
    """Find maximum pointwise distance.

    Computes by processing blocks of coordinates to reduce overall memory
    requirement. The memory used at runtime will be proportional to
    ``coords_per_block**2``.

    A block size of >= 2000 is recommended to overhead poor GPU resource usage
    and to reduce kernel launch overhead.

    Parameters
    ----------
    coords : np.ndarray (num_points, ndim)
        The coordinates to process.
    metric : str, optional
        Can be any metric supported by `scipy.spatial.distance.cdist`. The
        default is the squared Euclidean distance (sqeuclidean).
    return_argmax : bool, optional
        If True, a tuple containing two indices into the coords array that
        correspond to the maximum pairwise distance are returned.
    coords_per_block : bool, optional
        Internally, calls to cdist will be made with subsets of coords where
        the subset size is (coords_per_block, ndim).
    cdist_kwargs = dict, optional
        Can provide any additional kwargs to cdist (e.g. `p` for Minkowski
        norms).

    cdist requires num_coords**2 elements. This is very wasteful if we only
    want to find the maximum pointwise distance. In that case, this function
    processes smaller blocks of the overall cdist result, keeping a log of the
    maximum value.

    A schematic of the block sizes processed for an array of 10,000 coordinates
    with a block size of 4000 points is shown below:

      ┌────────┬────────┬────────┐
      │  4000  │  4000  │  2000  │
      │  4000  │  4000  │  2000  │
      ├────────┼────────┼────────┤
      │   X    │  4000  │  2000  │
      │        │  4000  │  2000  │
      ├────────┼────────┼────────┤
      │   X    │   X    │  2000  │
      │        │        │  2000  │
      └────────┴────────┴────────┘

    where due to the symmetry of the pdist matrix, we don't compute the lower
    triangular blocks.
    """
    num_coords, ndim = coords.shape
    if num_coords == 0:
        raise RuntimeError("No coordinates to process")

    blocks_per_dim = math.ceil(num_coords / coords_per_block)
    if blocks_per_dim > 1:
        # reuse the same temporary storage array for most blocks
        # (last block in row and column may be smaller)
        temp = cp.zeros((coords_per_block, coords_per_block), dtype=cp.float32)
    if coords.dtype not in [cp.float32, cp.float64]:
        coords = coords.astype(cp.float32, copy=False)
    if not coords.flags.c_contiguous:
        coords = cp.ascontiguousarray(coords)
    max_dist = 0
    for i in range(blocks_per_dim):
        for j in range(blocks_per_dim):
            if j < i:
                # skip symmetric regions
                continue
            sl_m = slice(
                i * coords_per_block,
                min((i + 1) * coords_per_block, num_coords),
            )
            sl_n = slice(
                j * coords_per_block,
                min((j + 1) * coords_per_block, num_coords),
            )
            # print(f"\t{i=}: {sl_m}, {j=}: {sl_n}")
            coords_block1 = coords[sl_m, :]
            coords_block2 = coords[sl_n, :]
            if i < blocks_per_dim - 1 and j < blocks_per_dim - 1:
                cdist(coords_block1, coords_block2, metric=metric, out=temp)
                current_max = float(temp.max())
                if return_argmax:
                    if current_max > max_dist:
                        loc_index = i, j
                        distances_max = temp.copy()
                        max_dist = current_max
                else:
                    max_dist = max(current_max, max_dist)
            else:
                out = cdist(coords_block1, coords_block2, metric=metric)
                current_max = float(out.max())
                if return_argmax:
                    if current_max > max_dist:
                        loc_index = i, j
                        distances_max = out.copy()
                        max_dist = current_max
                else:
                    max_dist = max(current_max, max_dist)
    if return_argmax:
        i, j = loc_index
        loc = np.unravel_index(int(distances_max.argmax()), distances_max.shape)
        loc = (
            int(loc[0]) + i * coords_per_block,
            int(loc[1]) + j * coords_per_block,
        )
        return max_dist, loc
    return max_dist


def regionprops_area_convex(
    images_convex,
    max_label=None,
    spacing=None,
    area_dtype=cp.float64,
    props_dict=None,
):
    if max_label is None:
        max_label = len(images_convex)
    if not isinstance(images_convex, Sequence):
        raise ValueError("Expected `images_convex` to be a sequence of images")
    area_convex = cp.zeros((max_label,), dtype=area_dtype)
    for i in range(max_label):
        area_convex[i] = images_convex[i].sum()
    if spacing is not None:
        if isinstance(spacing, cp.ndarray):
            pixel_area = cp.product(spacing)
        else:
            pixel_area = math.prod(spacing)
        area_convex *= pixel_area
    if props_dict is not None:
        props_dict["area_convex"] = area_convex
    return area_convex


def _regionprops_coords_perimeter(
    image,
    connectivity=1,
):
    """
    Takes an image of a single labeled region (e.g. one element of the tuple
    resulting from regionprops_image) and returns the coordinates of the voxels
    at the edge of that region.
    """

    # remove non-boundary pixels
    binary_image = image > 0
    footprint = ndi.generate_binary_structure(
        binary_image.ndim, connectivity=connectivity
    )
    binary_image_eroded = ndi.binary_erosion(binary_image, footprint)
    binary_edges = binary_image * ~binary_image_eroded
    edge_coords = _ndarray_argwhere(binary_edges)
    return edge_coords


def _feret_diameter_max(image_convex, spacing=None, return_argmax=False):
    """Compute the maximum Feret diameter of a single convex image region."""
    if image_convex.size == 1:
        warnings.warn(
            "single element image, returning 0 for feret diameter", UserWarning
        )
        return 0
    coords = _regionprops_coords_perimeter(image_convex, connectivity=1)
    coords = coords.astype(cp.float32)

    if spacing is not None:
        if all(s == 1.0 for s in spacing):
            spacing = None
        else:
            spacing = cp.asarray(spacing, dtype=cp.float32).reshape(1, -1)
            coords *= spacing

    out = pdist_max_blockwise(
        coords,
        metric="sqeuclidean",
        return_argmax=return_argmax,
        coords_per_block=4000,
    )
    if return_argmax:
        return math.sqrt(out[0]), out[1]
    return math.sqrt(out)


def regionprops_feret_diameter_max(
    images_convex, spacing=None, props_dict=None
):
    """Compute the maximum Feret diameter of the convex hull of each image in
    images_convex.

    Parameters
    ----------
    image_convex : cupy.ndarray
        The convex hull of the region.
    spacing : tuple of float, optional
        The pixel spacing of the image.
    props_dict : dict, optional
        A dictionary to store the computed properties.

    Notes
    -----
    The maximum Feret diameter is the maximum distance between any two
    points on the convex hull of the region. The implementation here is based
    on pairwise distances of all boundary coordinates rather than using
    marching squares or marching cubes as in scikit-image. The implementation
    here is n-dimensional.

    The distance is between pixel centers and so may be approximately one pixel
    width less than the one computed by scikit-image.
    """
    if not isinstance(images_convex, Sequence):
        raise ValueError("Expected `images_convex` to be a sequence of images")
    diameters = cp.asarray(
        tuple(
            _feret_diameter_max(
                image_convex, spacing=spacing, return_argmax=False
            )
            for image_convex in images_convex
        )
    )
    if props_dict is not None:
        props_dict["feret_diameter_max"] = diameters
    return diameters
