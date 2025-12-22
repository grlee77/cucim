# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""
Test script to validate CUB library availability and compilation with CuPy RawKernel.

This validates that the CUB primitives required for the wavelet matrix median filter
can be compiled and executed correctly.
"""

import cupy as cp
import pytest

# Compilation options required for CUB
# - --std=c++17: Required by recent CCCL versions
# - -DCUB_DISABLE_BF16_SUPPORT: Workaround for CUDA 12.2+ fp16/bf16 header issues
CUB_COMPILE_OPTIONS = ("--std=c++17", "-DCUB_DISABLE_BF16_SUPPORT")


# Test kernel using cub::WarpScan (exclusive sum)
_warp_scan_test_kernel = r"""
#include <cub/warp/warp_scan.cuh>

extern "C" __global__ void test_warp_scan(
    const int* input,
    int* output,
    int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Each thread gets one value
    int thread_data = input[tid];

    // Use CUB WarpScan for exclusive prefix sum
    typedef cub::WarpScan<int> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage;

    int exclusive_sum;
    WarpScan(temp_storage).ExclusiveSum(thread_data, exclusive_sum);

    output[tid] = exclusive_sum;
}
"""


# Test kernel using cub::WarpReduce (sum reduction)
_warp_reduce_test_kernel = r"""
#include <cub/warp/warp_reduce.cuh>

extern "C" __global__ void test_warp_reduce(
    const int* input,
    int* output,
    int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;

    // Each thread gets one value
    int thread_data = (tid < n) ? input[tid] : 0;

    // Use CUB WarpReduce for sum
    typedef cub::WarpReduce<int> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    int warp_sum = WarpReduce(temp_storage).Sum(thread_data);

    // First thread in warp writes result
    if (threadIdx.x % 32 == 0 && warp_id * 32 < n) {
        output[warp_id] = warp_sum;
    }
}
"""


# Test kernel using cub::BlockScan
_block_scan_test_kernel = r"""
#include <cub/block/block_scan.cuh>

extern "C" __global__ void test_block_scan(
    const int* input,
    int* output,
    int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread gets one value
    int thread_data = (tid < n) ? input[tid] : 0;

    // Use CUB BlockScan for exclusive prefix sum
    typedef cub::BlockScan<int, 256> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int exclusive_sum;
    BlockScan(temp_storage).ExclusiveSum(thread_data, exclusive_sum);

    if (tid < n) {
        output[tid] = exclusive_sum;
    }
}
"""


# Combined test with multiple CUB primitives (similar to wavelet matrix usage)
_combined_cub_test_kernel = r"""
#include <cub/warp/warp_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_scan.cuh>

extern "C" __global__ void test_combined_cub(
    const unsigned int* input,
    unsigned int* scan_output,
    unsigned int* reduce_output,
    int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Load data
    unsigned int thread_data = (tid < n) ? input[tid] : 0;

    // Test WarpScan with custom type
    typedef cub::WarpScan<unsigned int> WarpScanU32;
    __shared__ typename WarpScanU32::TempStorage warp_scan_storage[8];  // Up to 8 warps per block

    unsigned int warp_exclusive;
    unsigned int warp_inclusive;
    WarpScanU32(warp_scan_storage[warp_id]).Scan(thread_data, warp_inclusive, warp_exclusive, 0u, cub::Sum());

    // Test WarpReduce
    typedef cub::WarpReduce<unsigned int> WarpReduceU32;
    __shared__ typename WarpReduceU32::TempStorage warp_reduce_storage[8];

    unsigned int warp_sum = WarpReduceU32(warp_reduce_storage[warp_id]).Sum(thread_data);

    // Write results
    if (tid < n) {
        scan_output[tid] = warp_exclusive;
    }
    if (lane_id == 0 && warp_id == 0) {
        reduce_output[blockIdx.x] = warp_sum;
    }
}
"""


@pytest.fixture(scope="module")
def warp_scan_kernel():
    """Compile and return the warp scan test kernel."""
    return cp.RawKernel(
        _warp_scan_test_kernel,
        "test_warp_scan",
        options=CUB_COMPILE_OPTIONS,
    )


@pytest.fixture(scope="module")
def warp_reduce_kernel():
    """Compile and return the warp reduce test kernel."""
    return cp.RawKernel(
        _warp_reduce_test_kernel,
        "test_warp_reduce",
        options=CUB_COMPILE_OPTIONS,
    )


@pytest.fixture(scope="module")
def block_scan_kernel():
    """Compile and return the block scan test kernel."""
    return cp.RawKernel(
        _block_scan_test_kernel,
        "test_block_scan",
        options=CUB_COMPILE_OPTIONS,
    )


@pytest.fixture(scope="module")
def combined_kernel():
    """Compile and return the combined CUB test kernel."""
    return cp.RawKernel(
        _combined_cub_test_kernel,
        "test_combined_cub",
        options=CUB_COMPILE_OPTIONS,
    )


class TestCUBWarpScan:
    """Test CUB WarpScan compilation and correctness."""

    def test_compilation(self, warp_scan_kernel):
        """Test that CUB WarpScan kernel compiles successfully."""
        assert warp_scan_kernel is not None

    def test_exclusive_sum_single_warp(self, warp_scan_kernel):
        """Test WarpScan exclusive sum with a single warp (32 threads)."""
        n = 32
        input_data = cp.ones(n, dtype=cp.int32)
        output_data = cp.zeros(n, dtype=cp.int32)

        warp_scan_kernel((1,), (32,), (input_data, output_data, n))
        cp.cuda.Device().synchronize()

        # Exclusive sum of [1,1,1,...] should be [0,1,2,3,...]
        expected = cp.arange(n, dtype=cp.int32)
        cp.testing.assert_array_equal(output_data, expected)

    def test_exclusive_sum_values(self, warp_scan_kernel):
        """Test WarpScan exclusive sum with varying values."""
        n = 32
        input_data = cp.arange(1, n + 1, dtype=cp.int32)  # [1, 2, 3, ..., 32]
        output_data = cp.zeros(n, dtype=cp.int32)

        warp_scan_kernel((1,), (32,), (input_data, output_data, n))
        cp.cuda.Device().synchronize()

        # Verify exclusive prefix sum
        expected = cp.zeros(n, dtype=cp.int32)
        expected[1:] = cp.cumsum(input_data[:-1])
        cp.testing.assert_array_equal(output_data, expected)


class TestCUBWarpReduce:
    """Test CUB WarpReduce compilation and correctness."""

    def test_compilation(self, warp_reduce_kernel):
        """Test that CUB WarpReduce kernel compiles successfully."""
        assert warp_reduce_kernel is not None

    def test_sum_single_warp(self, warp_reduce_kernel):
        """Test WarpReduce sum with a single warp."""
        n = 32
        input_data = cp.ones(n, dtype=cp.int32)
        output_data = cp.zeros(1, dtype=cp.int32)

        warp_reduce_kernel((1,), (32,), (input_data, output_data, n))
        cp.cuda.Device().synchronize()

        assert int(output_data[0]) == 32

    def test_sum_values(self, warp_reduce_kernel):
        """Test WarpReduce sum with varying values."""
        n = 32
        input_data = cp.arange(1, n + 1, dtype=cp.int32)  # [1, 2, ..., 32]
        output_data = cp.zeros(1, dtype=cp.int32)

        warp_reduce_kernel((1,), (32,), (input_data, output_data, n))
        cp.cuda.Device().synchronize()

        expected_sum = n * (n + 1) // 2  # Sum of 1 to 32 = 528
        assert int(output_data[0]) == expected_sum


class TestCUBBlockScan:
    """Test CUB BlockScan compilation and correctness."""

    def test_compilation(self, block_scan_kernel):
        """Test that CUB BlockScan kernel compiles successfully."""
        assert block_scan_kernel is not None

    def test_exclusive_sum_full_block(self, block_scan_kernel):
        """Test BlockScan exclusive sum with full block (256 threads)."""
        n = 256
        input_data = cp.ones(n, dtype=cp.int32)
        output_data = cp.zeros(n, dtype=cp.int32)

        block_scan_kernel((1,), (256,), (input_data, output_data, n))
        cp.cuda.Device().synchronize()

        # Exclusive sum of [1,1,1,...] should be [0,1,2,3,...,255]
        expected = cp.arange(n, dtype=cp.int32)
        cp.testing.assert_array_equal(output_data, expected)


class TestCUBCombined:
    """Test combined CUB primitives (similar to wavelet matrix usage)."""

    def test_compilation(self, combined_kernel):
        """Test that combined CUB kernel compiles successfully."""
        assert combined_kernel is not None

    def test_combined_operations(self, combined_kernel):
        """Test combined WarpScan and WarpReduce operations."""
        n = 32
        input_data = cp.arange(n, dtype=cp.uint32)
        scan_output = cp.zeros(n, dtype=cp.uint32)
        reduce_output = cp.zeros(1, dtype=cp.uint32)

        combined_kernel(
            (1,), (32,), (input_data, scan_output, reduce_output, n)
        )
        cp.cuda.Device().synchronize()

        # Verify scan output (exclusive prefix sum)
        expected_scan = cp.zeros(n, dtype=cp.uint32)
        expected_scan[1:] = cp.cumsum(input_data[:-1])
        cp.testing.assert_array_equal(scan_output, expected_scan)

        # Verify reduce output (sum of first warp)
        expected_sum = int(cp.sum(input_data[:32]))
        assert int(reduce_output[0]) == expected_sum


class TestCUBCompilationOptions:
    """Test that compilation options work correctly."""

    def test_cpp17_required(self):
        """Test that --std=c++17 flag is required for CUB."""
        # This kernel uses if constexpr which requires C++17
        kernel_code = r"""
        #include <cub/warp/warp_scan.cuh>

        extern "C" __global__ void test_cpp17(int* out) {
            if constexpr (sizeof(int) == 4) {
                out[0] = 1;
            } else {
                out[0] = 0;
            }
        }
        """
        # Should compile with C++17
        kernel = cp.RawKernel(
            kernel_code, "test_cpp17", options=CUB_COMPILE_OPTIONS
        )
        out = cp.zeros(1, dtype=cp.int32)
        kernel((1,), (1,), (out,))
        cp.cuda.Device().synchronize()
        assert int(out[0]) == 1

    def test_uint32_block_type(self):
        """Test compilation of block type similar to wavelet matrix."""
        # This tests the BlockT struct pattern used in wavelet matrix
        # We use two separate uint32 arrays instead of a struct since CuPy
        # handles structured arrays differently from NumPy for field access
        kernel_code = r"""
        #include <cub/warp/warp_scan.cuh>

        struct __align__(8) BlockT {
            unsigned int nsum;
            unsigned int nbit;
        };

        extern "C" __global__ void test_block_type(
            unsigned int* nsum_out,
            unsigned int* nbit_out,
            const unsigned int* values,
            int n
        ) {
            const int tid = threadIdx.x;
            if (tid >= n) return;

            typedef cub::WarpScan<unsigned int> WarpScan;
            __shared__ typename WarpScan::TempStorage temp_storage;

            unsigned int val = values[tid];
            unsigned int exclusive_sum;
            WarpScan(temp_storage).ExclusiveSum(val, exclusive_sum);

            // Simulate bit counting
            unsigned int bits = __ballot_sync(0xFFFFFFFF, val > 0);

            nsum_out[tid] = exclusive_sum;
            nbit_out[tid] = bits;
        }
        """
        kernel = cp.RawKernel(
            kernel_code, "test_block_type", options=CUB_COMPILE_OPTIONS
        )

        n = 32
        nsum_out = cp.zeros(n, dtype=cp.uint32)
        nbit_out = cp.zeros(n, dtype=cp.uint32)
        values = cp.arange(n, dtype=cp.uint32)

        kernel((1,), (32,), (nsum_out, nbit_out, values, n))
        cp.cuda.Device().synchronize()

        # Verify nsum is exclusive prefix sum
        expected_nsum = cp.zeros(n, dtype=cp.uint32)
        expected_nsum[1:] = cp.cumsum(values[:-1])
        cp.testing.assert_array_equal(nsum_out, expected_nsum)

        # Verify nbit has all bits set (all values > 0 except first)
        # values[0] = 0, so bit 0 is not set
        expected_bits = 0xFFFFFFFE  # All bits except bit 0
        assert int(nbit_out[0]) == expected_bits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
