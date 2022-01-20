import os
import pickle

import cucim.skimage
import cucim.skimage.filters
import numpy as np
import pandas as pd
import skimage
import skimage.filters

from _image_bench import ImageBench

pfile = "cucim_filters_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()
dtypes = [np.float32]

for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # _gabor.py
    (
        "gabor",
        dict(n_stds=3),
        dict(frequency=[0.075, 0.1, 0.2, 0.3]),
        False,
        False,
    ),
    # _gaussian.py
    (
        "gaussian",
        dict(truncate=4.0, preserve_range=True),
        dict(sigma=[0.25, 1, 4]),
        True,
        True,
    ),
    # _median.py
    ("median", dict(mode="nearest"), dict(), False, True),
    # _rank_order.py
    ("rank_order", dict(), dict(), False, True),
    # _unsharp_mask.py
    (
        "unsharp_mask",
        dict(),
        dict(radius=[0.5, 1.0, 2.0, 3.0]),
        True,
        True,
    ),
    # edges.py
    ("sobel", dict(), dict(axis=[None, 0, -1]), False, True),
    ("prewitt", dict(), dict(axis=[None, 0, -1]), False, True),
    ("scharr", dict(), dict(axis=[None, 0, -1]), False, True),
    ("roberts", dict(), dict(), False, False),
    ("roberts_pos_diag", dict(), dict(), False, False),
    ("roberts_neg_diag", dict(), dict(), False, False),
    ("farid", dict(), dict(), False, False),
    ("laplace", dict(ksize=3), dict(), False, True),
    # lpi_filter.py
    # TODO: benchmark wiener
    # ridges.py
    # TODO: had to set meijering, etc allow_nd to False just due to insufficient GPU memory
    (
        "meijering",
        dict(sigmas=range(1, 10, 2), alpha=None),
        dict(black_ridges=[True, False], mode=["reflect"]),
        False,
        False,
    ),
    (
        "sato",
        dict(sigmas=range(1, 10, 2)),
        dict(black_ridges=[True, False], mode=["reflect"]),
        False,
        False,
    ),
    (
        "frangi",
        dict(sigmas=range(1, 10, 2)),
        dict(black_ridges=[True, False], mode=["reflect"]),
        False,
        False,
    ),
    (
        "hessian",
        dict(sigmas=range(1, 10, 2)),
        dict(black_ridges=[True, False], mode=["reflect"]),
        False,
        False,
    ),
    # thresholding.py
    ("threshold_isodata", dict(), dict(nbins=[64, 256]), False, True),
    ("threshold_otsu", dict(), dict(nbins=[64, 256]), False, True),
    ("threshold_yen", dict(), dict(nbins=[64, 256]), False, True),
    # TODO: threshold_local should support n-dimensional data
    (
        "threshold_local",
        dict(),
        dict(block_size=[5, 15], method=["gaussian", "mean", "median"]),
        False,
        False,
    ),
    ("threshold_li", dict(), dict(), False, True),
    ("threshold_minimum", dict(), dict(nbins=[64, 256]), False, True),
    ("threshold_mean", dict(), dict(), False, True),
    ("threshold_triangle", dict(), dict(nbins=[64, 256]), False, True),
    ("threshold_niblack", dict(), dict(window_size=[7, 15, 65]), False, True),
    ("threshold_sauvola", dict(), dict(window_size=[7, 15, 65]), False, True),
    ("apply_hysteresis_threshold", dict(low=0.15, high=0.6), dict(), False, True),
    ("threshold_multiotsu", dict(), dict(nbins=[64, 256], classes=[3]), False, True),
]:

    for shape in [(512, 512), (3840, 2160), (3840, 2160, 3), (192, 192, 192)]:

        ndim = len(shape)
        if not allow_nd:
            if not allow_color:
                if ndim > 2:
                    continue
            else:
                if ndim > 3 or (ndim == 3 and shape[-1] not in [3, 4]):
                    continue
        if shape[-1] == 3 and not allow_color:
            continue

        if function_name == "gabor" and np.prod(shape) > 1000000:
            # avoid cases that are too slow on the CPU
            var_kwargs["frequency"] = [f for f in var_kwargs["frequency"] if f >= 0.1]

        if function_name == "median":
            footprints = []
            ndim = len(shape)
            footprint_sizes = [3, 5, 7, 9] if ndim == 2 else [3, 5, 7]
            for footprint_size in [3, 5, 7, 9]:
                footprints.append(
                    np.ones((footprint_size,) * ndim, dtype=bool)
                )
            var_kwargs["footprint"] = footprints

        if function_name in ["gaussian", "unsharp_mask"]:
            fixed_kwargs["channel_axis"] = -1 if shape[-1] == 3 else None

        B = ImageBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.filters,
            module_gpu=cucim.skimage.filters,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
