import math
import os
import pickle

import cucim.skimage
import cucim.skimage.measure
import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.measure

from _image_bench import ImageBench


class LabelBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        contiguous_labels=True,
        dtypes=np.float32,
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=skimage.measure,
        module_gpu=cucim.skimage.measure,
    ):

        self.contiguous_labels = contiguous_labels

        super().__init__(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            index_str=index_str,
            module_cpu=module_cpu,
            module_gpu=module_gpu,
        )

    def set_args(self, dtype):
        a = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 4, 0],
                [2, 2, 0, 0, 3, 0, 4, 4],
                [0, 0, 0, 0, 0, 5, 0, 0],
            ]
        )
        tiling = tuple(s // a_s for s, a_s in zip(shape, a.shape))
        if self.contiguous_labels:
            image = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            image = np.tile(a, tiling)
        imaged = cp.asarray(image)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class RegionpropsBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        contiguous_labels=True,
        dtypes=np.float32,
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=skimage.measure,
        module_gpu=cucim.skimage.measure,
    ):

        self.contiguous_labels = contiguous_labels

        super().__init__(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            index_str=index_str,
            module_cpu=module_cpu,
            module_gpu=module_gpu,
        )

    def set_args(self, dtype):
        a = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 4, 0],
                [2, 2, 0, 0, 3, 0, 4, 4],
                [0, 0, 0, 0, 0, 5, 0, 0],
            ]
        )
        tiling = tuple(s // a_s for s, a_s in zip(shape, a.shape))
        if self.contiguous_labels:
            image = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            image = np.tile(a, tiling)
        imaged = cp.asarray(image)
        label_dev = cucim.skimage.measure.label(imaged).astype(int)
        label = cp.asnumpy(label_dev)

        self.args_cpu = (label, image)
        self.args_gpu = (label_dev, imaged)


class FiltersBench(ImageBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            im1 = skimage.data.camera()
        else:
            im1 = skimage.data.camera() / 255.0
            im1 = im1.astype(dtype)
        if len(self.shape) == 3:
            im1 = im1[..., np.newaxis]
        n_tile = [math.ceil(s / im_s) for s, im_s in zip(self.shape, im1.shape)]
        slices = tuple([slice(s) for s in self.shape])
        image = np.tile(im1, n_tile)[slices]
        imaged = cp.asarray(image)
        assert imaged.dtype == dtype
        assert imaged.shape == self.shape
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


pfile = "cucim_measure_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()
dtypes = [np.float32]

for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # _gaussian.py
    (
        "label",
        dict(return_num=False, background=0),
        dict(connectivity=[1, 2]),
        False,
        True,
    ),
    # regionprops.py
    ("regionprops", dict(), dict(), False, True),
]:

    for shape in [(512, 512), (3840, 2160), (3840, 2160, 3), (192, 192, 192)]:

        ndim = len(shape)
        if not allow_nd:
            if allow_color:
                if ndim > 2:
                    continue
            else:
                if ndim > 3 or (ndim == 3 and shape[-1] not in [3, 4]):
                    continue
        if shape[-1] == 3 and not allow_color:
            continue

        Tester = LabelBench if function_name == "label" else RegionpropsBench

        for contiguous_labels in [True, False]:
            if contiguous_labels:
                index_str = f"contiguous"
            else:
                index_str = None
            B = Tester(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                contiguous_labels=contiguous_labels,
                index_str=index_str,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.measure,
                module_gpu=cucim.skimage.measure,
            )
            results = B.run_benchmark(duration=1)
            all_results = all_results.append(results["full"])


for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # _moments.py
    ("moments", dict(), dict(order=[1, 2, 3, 4]), False, False),
    ("moments_central", dict(), dict(order=[1, 2, 3]), False, True),
    # omited from benchmarks (only tiny arrays): moments_normalized, moments_hu
    ("centroid", dict(), dict(), False, True),
    ("inertia_tensor", dict(), dict(), False, True),
    ("inertia_tensor_eigvals", dict(), dict(), False, True),
    # _polygon.py
    # TODO: approximate_polygon, subdivide_polygon
    # block.py
    (
        "block_reduce",
        dict(),
        dict(
            func=[
                cp.sum,
            ]
        ),
        True,
        True,
    ),  # variable block_size configured below
    # entropy.py
    ("shannon_entropy", dict(base=2), dict(), True, True),
    # profile.py
    (
        "profile_line",
        dict(src=(5, 7)),
        dict(reduce_func=[cp.mean], linewidth=[1, 2, 4], order=[1, 3]),
        True,
        False,
    ),  # variable block_size configured below
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

        if function_name == "block_reduce":
            ndim = len(shape)
            if shape[-1] == 3:
                block_sizes = [(b,) * (ndim - 1) + (3,) for b in (16, 32, 64)]
            else:
                block_sizes = [(b,) * ndim for b in (16, 32, 64)]
            var_kwargs["block_size"] = block_sizes

        if function_name == "profile_line":
            fixed_kwargs["dst"] = (shape[0] - 32, shape[1] + 9)

        if function_name == "median":
            footprints = []
            ndim = len(shape)
            footprint_sizes = [3, 5, 7, 9] if ndim == 2 else [3, 5, 7]
            for footprint_size in [3, 5, 7, 9]:
                footprints.append(
                    np.ones((footprint_sizes,) * ndim, dtype=bool)
                )
            var_kwargs["footprint"] = footprints

        if function_name in ["gaussian", "unsharp_mask"]:
            fixed_kwargs["channel_axis"] = -1 if shape[-1] == 3 else None

        B = FiltersBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.measure,
            module_gpu=cucim.skimage.measure,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
