# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Testing mask support for filtering functions."""

import math

import cupy as cp
import pytest

from cucim.skimage._vendored import ndimage as ndi


def create_checkerboard_mask(shape):
    """Create a checkerboard mask pattern.

    e.g.
    [
        1, 1, 1, 0, 0, 0,
        1, 1, 1, 0, 0, 0,
        1, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 1,
        0, 0, 0, 1, 1, 1,
        0, 0, 0, 1, 1, 1,
    ]

    Parameters
    ----------
    shape : tuple
        Desired shape of the output mask.

    Returns
    -------
        cupy.ndarray: Boolean mask with checkerboard pattern.
    """

    # Create base 2x2x... checkerboard pattern
    base_shape = (2,) * len(shape)
    base_pattern = cp.zeros(base_shape, dtype=bool)

    # Set alternating blocks to True
    # For a checkerboard, we want indices where sum of coordinates is even
    indices = cp.meshgrid(
        *[cp.arange(2) for _ in range(len(shape))], indexing="ij"
    )
    coord_sum = sum(indices)
    base_pattern = coord_sum % 2 == 0

    # Expand using Kronecker product
    ones_shape = tuple(math.ceil(s / 2) for s in shape)
    expanded = cp.kron(base_pattern, cp.ones(ones_shape, dtype=bool))
    print(f"{expanded.shape = }")

    # Crop to desired shape
    slices = tuple(slice(0, s) for s in shape)
    return expanded[slices]


@pytest.mark.parametrize("filter_func", ["correlate", "convolve"])
@pytest.mark.parametrize("dtype", [cp.float32])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize(
    "mode", ["reflect", "constant", "nearest", "mirror", "wrap"]
)
@pytest.mark.parametrize("mask_type", ["random", "checkerboard"])
def test_correlate_convolve_with_mask(
    filter_func, dtype, ndim, mode, mask_type
):
    """Test correlate/convolve with mask parameter."""
    filter_fn = getattr(ndi, filter_func)
    # Create test data
    shape = (24,) * ndim
    rng = cp.random.default_rng(42)
    input_data = rng.random(shape, dtype=cp.float32).astype(dtype)

    # Create a binary mask
    if mask_type == "random":
        mask = rng.random(shape) > 0.5
    elif mask_type == "checkerboard":
        mask = create_checkerboard_mask(shape)
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")

    # Create weights
    weights_shape = (3,) * ndim
    weights = cp.ones(weights_shape, dtype=cp.float32)

    # Apply filter without mask
    filtered_no_mask = filter_fn(input_data, weights, mode=mode)

    # Apply filter with mask
    filtered_with_mask = filter_fn(input_data, weights, mode=mode, mask=mask)

    # regions outside the mask should preserve original values
    cp.testing.assert_array_equal(filtered_with_mask[~mask], input_data[~mask])

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("filter_func", ["correlate1d", "convolve1d"])
@pytest.mark.parametrize("axis", [0, 1])
def test_correlate1d_convolve1d_with_mask(filter_func, axis):
    """Test correlate1d/convolve1d with mask parameter."""
    filter_fn = getattr(ndi, filter_func)
    # Create test data
    input_data = cp.array(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dtype=cp.float32,
    )

    # Create a mask
    mask = cp.array(
        [
            [True, False, True, False, True],
            [False, True, False, True, False],
            [True, True, False, False, True],
        ],
        dtype=bool,
    )

    # Create weights
    weights = cp.array([1, 2, 1], dtype=cp.float32) / 4.0

    # Apply filter without mask
    filtered_no_mask = filter_fn(input_data, weights, axis=axis, mode="reflect")

    # Apply filter with mask
    filtered_with_mask = filter_fn(
        input_data, weights, axis=axis, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_equal(filtered_with_mask[~mask], input_data[~mask])

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("size", [3, 5])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_uniform_filter_with_mask(size, ndim):
    """Test uniform_filter with mask parameter."""
    # Create test data
    shape = (32,) * ndim
    rng = cp.random.default_rng(123)
    input_data = rng.random(shape, dtype=cp.float32) * 100

    # Create mask region
    mask = create_checkerboard_mask(shape)

    # Apply filter without mask
    filtered_no_mask = ndi.uniform_filter(input_data, size=size, mode="reflect")

    # Apply filter with mask
    filtered_with_mask = ndi.uniform_filter(
        input_data, size=size, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[~mask], input_data[~mask]
    )

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_uniform_filter1d_with_mask(axis):
    """Test uniform_filter1d with mask parameter."""
    # Create test data
    input_data = cp.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ],
        dtype=cp.float32,
    )

    # Create a mask - every other element
    mask = cp.zeros_like(input_data, dtype=bool)
    mask[::2, ::2] = True
    mask[1::2, 1::2] = True

    # Apply filter without mask
    filtered_no_mask = ndi.uniform_filter1d(
        input_data, size=3, axis=axis, mode="reflect"
    )

    # Apply filter with mask
    filtered_with_mask = ndi.uniform_filter1d(
        input_data, size=3, axis=axis, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[~mask], input_data[~mask]
    )

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("sigma", [1.0, 2.0])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_gaussian_filter_with_mask(sigma, ndim):
    """Test gaussian_filter with mask parameter."""
    # Create test data
    shape = (25,) * ndim
    rng = cp.random.default_rng(456)
    input_data = rng.random(shape, dtype=cp.float32) * 255

    # Create a checkerboard mask
    mask = create_checkerboard_mask(shape)

    # Apply filter without mask
    filtered_no_mask = ndi.gaussian_filter(
        input_data, sigma=sigma, mode="reflect"
    )

    # Apply filter with mask
    filtered_with_mask = ndi.gaussian_filter(
        input_data, sigma=sigma, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[~mask], input_data[~mask]
    )

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_gaussian_filter1d_with_mask(axis):
    """Test gaussian_filter1d with mask parameter."""
    # Create test data
    input_data = cp.arange(24, dtype=cp.float32).reshape(4, 6)

    # Create a mask - only filter certain rows/columns
    if axis == 0:
        mask = cp.array(
            [
                [True, False, True, False, True, False],
                [False, True, False, True, False, True],
                [True, False, True, False, True, False],
                [False, True, False, True, False, True],
            ],
            dtype=bool,
        )
    else:
        mask = cp.array(
            [
                [True, True, False, False, True, True],
                [False, False, True, True, False, False],
                [True, True, False, False, True, True],
                [False, False, True, True, False, False],
            ],
            dtype=bool,
        )

    # Apply filter without mask
    filtered_no_mask = ndi.gaussian_filter1d(
        input_data, sigma=1.0, axis=axis, mode="reflect"
    )

    # Apply filter with mask
    filtered_with_mask = ndi.gaussian_filter1d(
        input_data, sigma=1.0, axis=axis, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[~mask], input_data[~mask]
    )

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize(
    "filter_func,filter_kwargs",
    [
        ("minimum_filter", {}),
        ("maximum_filter", {}),
        ("median_filter", {}),
        ("rank_filter", {"rank": 0}),
        ("rank_filter", {"rank": 4}),
        ("percentile_filter", {"percentile": 0}),
        ("percentile_filter", {"percentile": 50}),
    ],
)
@pytest.mark.parametrize("size", [3, 5])
def test_rank_filters_with_mask(filter_func, filter_kwargs, size):
    """Test rank-based filters (minimum, maximum, rank, percentile) with mask parameter."""
    filter_fn = getattr(ndi, filter_func)
    # Create test data with a larger array
    rng = cp.random.default_rng(999)
    input_data = rng.integers(1, 10, size=(30, 40), dtype=cp.int32).astype(
        cp.float32
    )

    # Create a checkerboard mask
    mask = create_checkerboard_mask(input_data.shape)

    # Apply filter without mask
    filtered_no_mask = filter_fn(
        input_data, size=size, mode="reflect", **filter_kwargs
    )

    # Apply filter with mask
    filtered_with_mask = filter_fn(
        input_data, size=size, mode="reflect", mask=mask, **filter_kwargs
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_equal(filtered_with_mask[~mask], input_data[~mask])

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_minimum_filter1d_with_mask(axis):
    """Test minimum_filter1d with mask parameter."""
    # Create test data
    input_data = cp.array(
        [[10, 20, 30, 40, 50], [15, 25, 35, 45, 55], [12, 22, 32, 42, 52]],
        dtype=cp.float32,
    )

    # Create a mask
    mask = cp.ones_like(input_data, dtype=bool)
    mask[0, :] = False  # Don't filter first row

    # Apply filter without mask
    filtered_no_mask = ndi.minimum_filter1d(
        input_data, size=3, axis=axis, mode="reflect"
    )

    # Apply filter with mask
    filtered_with_mask = ndi.minimum_filter1d(
        input_data, size=3, axis=axis, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_equal(filtered_with_mask[~mask], input_data[~mask])

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_maximum_filter1d_with_mask(axis):
    """Test maximum_filter1d with mask parameter."""
    # Create test data
    input_data = cp.array(
        [[10, 20, 30, 40, 50], [15, 25, 35, 45, 55], [12, 22, 32, 42, 52]],
        dtype=cp.float32,
    )

    # Create a mask
    mask = cp.ones_like(input_data, dtype=bool)
    mask[:, 0] = False  # Don't filter first column

    # Apply filter without mask
    filtered_no_mask = ndi.maximum_filter1d(
        input_data, size=3, axis=axis, mode="reflect"
    )

    # Apply filter with mask
    filtered_with_mask = ndi.maximum_filter1d(
        input_data, size=3, axis=axis, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_equal(filtered_with_mask[~mask], input_data[~mask])

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("filter_func", ["prewitt", "sobel"])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_edge_filters_with_mask(filter_func, axis):
    """Test prewitt/sobel edge filters with mask parameter."""
    filter_fn = getattr(ndi, filter_func)
    # Create test data with edges
    input_data = cp.zeros((24, 32), dtype=cp.float32)
    input_data[6:, :] = 100  # Horizontal edge

    mask = create_checkerboard_mask(input_data.shape)

    # Apply filter without mask
    filtered_no_mask = filter_fn(input_data, axis=axis, mode="reflect")

    # Apply filter with mask
    filtered_with_mask = filter_fn(
        input_data, axis=axis, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[~mask], input_data[~mask]
    )
    # within the mask, should match filtering without a mask
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


def test_laplace_with_mask():
    """Test laplace filter with mask parameter."""
    # Create test data
    rng = cp.random.default_rng(321)
    input_data = rng.random((50, 50), dtype=cp.float32) * 10

    # Create an interior mask
    mask = create_checkerboard_mask(input_data.shape)

    # Apply filter without mask
    filtered_no_mask = ndi.laplace(input_data, mode="reflect")

    # Apply filter with mask
    filtered_with_mask = ndi.laplace(input_data, mode="reflect", mask=mask)

    # regions outside the mask should preserve original values
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[~mask], input_data[~mask]
    )

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_gaussian_laplace_with_mask(sigma):
    """Test gaussian_laplace filter with mask parameter."""
    # Create test data
    rng = cp.random.default_rng(101)
    input_data = rng.random((48, 32), dtype=cp.float32) * 10

    # Create a checkerboard mask
    mask = create_checkerboard_mask(input_data.shape)

    # Apply filter without mask
    filtered_no_mask = ndi.gaussian_laplace(
        input_data, sigma=sigma, mode="reflect"
    )

    # Apply filter with mask
    filtered_with_mask = ndi.gaussian_laplace(
        input_data, sigma=sigma, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[~mask], input_data[~mask]
    )

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_gaussian_gradient_magnitude_with_mask(sigma):
    """Test gaussian_gradient_magnitude filter with mask parameter."""
    # Create test data with gradient
    input_data = cp.zeros((24, 64), dtype=cp.float32)
    for i in range(12):
        input_data[i, :] = i * 10  # Linear gradient

    # Create a checkerboard mask
    mask = create_checkerboard_mask(input_data.shape)

    # Apply filter without mask
    filtered_no_mask = ndi.gaussian_gradient_magnitude(
        input_data, sigma=sigma, mode="reflect"
    )

    # Apply filter with mask
    filtered_with_mask = ndi.gaussian_gradient_magnitude(
        input_data, sigma=sigma, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[~mask], input_data[~mask]
    )

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_almost_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


@pytest.mark.parametrize("filter_func", ["grey_erosion", "grey_dilation"])
@pytest.mark.parametrize("size", [3, 5])
def test_grey_morphology_with_mask(filter_func, size):
    """Test grey_erosion/grey_dilation with mask parameter."""
    filter_fn = getattr(ndi, filter_func)
    # Create test data
    input_data = cp.array(
        [[5, 8, 3, 7, 2], [9, 1, 6, 4, 8], [3, 7, 2, 9, 5], [6, 4, 8, 1, 7]],
        dtype=cp.float32,
    )

    # Create a checkerboard mask
    mask = cp.zeros_like(input_data, dtype=bool)
    mask[::2, ::2] = True
    mask[1::2, 1::2] = True

    # Apply filter without mask
    filtered_no_mask = filter_fn(input_data, size=size, mode="reflect")

    # Apply filter with mask
    filtered_with_mask = filter_fn(
        input_data, size=size, mode="reflect", mask=mask
    )

    # regions outside the mask should preserve original values
    cp.testing.assert_array_equal(filtered_with_mask[~mask], input_data[~mask])

    # within the mask, should match filtering without a mask
    cp.testing.assert_array_equal(
        filtered_with_mask[mask], filtered_no_mask[mask]
    )


def test_mask_all_false():
    """Test that when mask is all False, original data is preserved."""
    input_data = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    mask = cp.zeros_like(input_data, dtype=bool)

    # Test with various filters
    result = ndi.gaussian_filter(input_data, sigma=1.0, mask=mask)
    cp.testing.assert_array_equal(result, input_data)

    result = ndi.uniform_filter(input_data, size=3, mask=mask)
    cp.testing.assert_array_equal(result, input_data)

    result = ndi.minimum_filter(input_data, size=3, mask=mask)
    cp.testing.assert_array_equal(result, input_data)

    result = ndi.maximum_filter(input_data, size=3, mask=mask)
    cp.testing.assert_array_equal(result, input_data)


def test_mask_all_true():
    """Test that when mask is all True, result matches unmasked filtering."""
    input_data = cp.arange(2048, dtype=cp.float32).reshape(64, 32)
    mask = cp.ones_like(input_data, dtype=bool)

    # Test with gaussian filter
    result_with_mask = ndi.gaussian_filter(input_data, sigma=1.0, mask=mask)
    result_no_mask = ndi.gaussian_filter(input_data, sigma=1.0)
    cp.testing.assert_array_almost_equal(result_with_mask, result_no_mask)

    # Test with uniform filter
    result_with_mask = ndi.uniform_filter(input_data, size=3, mask=mask)
    result_no_mask = ndi.uniform_filter(input_data, size=3)
    cp.testing.assert_array_almost_equal(result_with_mask, result_no_mask)


def test_mask_with_output_parameter():
    """Test that mask works correctly with pre-allocated output."""
    input_data = cp.arange(1600, dtype=cp.float32).reshape(40, 40)
    # Create a checkerboard mask
    mask = create_checkerboard_mask(input_data.shape)

    # Pre-allocate output
    output = cp.empty_like(input_data)

    # Apply filter with mask and output parameter
    result = ndi.gaussian_filter(
        input_data, sigma=1.0, output=output, mask=mask
    )

    # Check that result and output are the same
    assert result is output

    # regions outside the mask should preserve original values
    cp.testing.assert_array_almost_equal(result[~mask], input_data[~mask])

    # within the mask, should match filtering without a mask
    unmasked_result = ndi.gaussian_filter(input_data, sigma=1.0)
    cp.testing.assert_array_almost_equal(unmasked_result[mask], output[mask])


def test_shared_memory_algorithm_rejects_mask():
    """Test that shared_memory algorithm properly rejects mask parameter."""
    input_data = cp.arange(2000, dtype=cp.float32).reshape(40, 50)
    mask = cp.ones_like(input_data, dtype=bool)

    # This should raise NotImplementedError
    with pytest.raises(
        NotImplementedError,
        match="algorithm 'shared_memory' does not support mask",
    ):
        ndi.convolve1d(
            input_data,
            cp.array([1, 2, 1]),
            axis=0,
            algorithm="shared_memory",
            mask=mask,
        )
