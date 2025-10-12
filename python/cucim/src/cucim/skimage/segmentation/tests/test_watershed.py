"""Tests for watershed segmentation."""

import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from cupyx.scipy import ndimage as ndi

from cucim.skimage.segmentation import watershed


class TestWatershed:
    """Test watershed segmentation function."""

    def test_watershed_simple_2d(self):
        """Test basic watershed segmentation on a simple 2D image."""
        # Create a simple image with two peaks
        image = cp.zeros((10, 10), dtype=cp.float32)
        image[2, 2] = 10
        image[7, 7] = 10
        image = ndi.gaussian_filter(image, sigma=1.0)

        # Create markers at the peaks
        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[2, 2] = 1
        markers[7, 7] = 2

        # Apply watershed (use negative image so peaks are valleys)
        labels = watershed(-image, markers)

        # Check that we got two distinct regions
        assert labels.shape == (10, 10)
        assert cp.max(labels) == 2
        assert cp.min(labels) >= 1

        # Check that markers are preserved
        assert labels[2, 2] == 1
        assert labels[7, 7] == 2

        # Check that all pixels are labeled
        assert cp.all(labels > 0)

    def test_watershed_connectivity_4(self):
        """Test watershed with 4-connectivity."""
        # Create a cross-shaped structure
        image = cp.ones((7, 7), dtype=cp.float32)
        image[3, :] = 0  # horizontal bar
        image[:, 3] = 0  # vertical bar

        # Place markers in four quadrants
        markers = cp.zeros((7, 7), dtype=cp.int32)
        markers[1, 1] = 1  # top-left
        markers[1, 5] = 2  # top-right
        markers[5, 1] = 3  # bottom-left
        markers[5, 5] = 4  # bottom-right

        labels = watershed(image, markers, connectivity=1)

        # Check that we got four distinct regions
        assert cp.max(labels) == 4

        # Check that markers are preserved
        assert labels[1, 1] == 1
        assert labels[1, 5] == 2
        assert labels[5, 1] == 3
        assert labels[5, 5] == 4

    def test_watershed_connectivity_8(self):
        """Test watershed with 8-connectivity."""
        # Create a simple diagonal structure
        image = cp.ones((7, 7), dtype=cp.float32)
        for i in range(7):
            image[i, i] = 0

        markers = cp.zeros((7, 7), dtype=cp.int32)
        markers[1, 5] = 1
        markers[5, 1] = 2

        labels = watershed(image, markers, connectivity=2)

        # Check that we got two regions
        assert cp.max(labels) == 2
        assert labels[1, 5] == 1
        assert labels[5, 1] == 2

    def test_watershed_with_mask(self):
        """Test watershed with a binary mask."""
        # Create image
        image = cp.ones((10, 10), dtype=cp.float32)
        image[3:7, 3:7] = 0

        # Create markers
        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[5, 5] = 1

        # Create mask (only process center region)
        mask = cp.zeros((10, 10), dtype=cp.bool_)
        mask[2:8, 2:8] = True

        labels = watershed(image, markers, mask=mask)

        # Check that only masked region is labeled
        assert labels[5, 5] == 1
        assert labels[0, 0] == 0
        assert labels[9, 9] == 0

        # Check that some pixels in masked region are labeled
        assert cp.sum(labels[2:8, 2:8] > 0) > 1

    def test_watershed_multiple_markers(self):
        """Test watershed with multiple markers."""
        # Create a gradient image
        y, x = cp.ogrid[:20, :20]
        image = cp.sqrt((x - 5) ** 2 + (y - 5) ** 2).astype(cp.float32)
        image += cp.sqrt((x - 15) ** 2 + (y - 15) ** 2).astype(cp.float32)

        # Create three markers
        markers = cp.zeros((20, 20), dtype=cp.int32)
        markers[5, 5] = 1
        markers[15, 15] = 2
        markers[10, 10] = 3

        labels = watershed(image, markers)

        # Check that we got three regions
        assert cp.max(labels) == 3

        # Check that markers are preserved
        assert labels[5, 5] == 1
        assert labels[15, 15] == 2
        assert labels[10, 10] == 3

    def test_watershed_flat_image(self):
        """Test watershed on a flat image (all pixels have same value)."""
        # Flat image
        image = cp.ones((10, 10), dtype=cp.float32)

        # Two markers
        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[2, 2] = 1
        markers[7, 7] = 2

        labels = watershed(image, markers)

        # Should still segment into two regions
        assert cp.max(labels) == 2
        assert labels[2, 2] == 1
        assert labels[7, 7] == 2

    def test_watershed_single_marker(self):
        """Test watershed with a single marker (should label everything)."""
        image = cp.random.random((10, 10)).astype(cp.float32)

        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[5, 5] = 1

        labels = watershed(image, markers)

        # All pixels should be labeled with 1
        assert cp.all(labels == 1)

    def test_watershed_gradient_image(self):
        """Test watershed on a gradient magnitude image."""
        # Create synthetic image with two objects
        image = cp.zeros((30, 30), dtype=cp.float32)
        image[5:15, 5:15] = 1
        image[16:26, 16:26] = 1

        # Compute gradient magnitude
        gradient = ndi.gaussian_gradient_magnitude(image, sigma=1.0)

        # Create markers
        markers = cp.zeros((30, 30), dtype=cp.int32)
        markers[10, 10] = 1
        markers[21, 21] = 2

        labels = watershed(gradient, markers)

        # Should get two regions
        assert cp.max(labels) == 2
        assert labels[10, 10] == 1
        assert labels[21, 21] == 2

    def test_watershed_markers_dtype_conversion(self):
        """Test that markers are properly converted to int32."""
        image = cp.random.random((10, 10)).astype(cp.float32)

        # Test with different dtypes
        for dtype in [cp.uint8, cp.int16, cp.int64]:
            markers = cp.zeros((10, 10), dtype=dtype)
            markers[3, 3] = 1
            markers[7, 7] = 2

            labels = watershed(image, markers)

            assert labels.dtype == cp.int32
            assert cp.max(labels) == 2

    def test_watershed_image_dtype_conversion(self):
        """Test that image is properly converted to float32."""
        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[3, 3] = 1
        markers[7, 7] = 2

        # Test with different dtypes
        for dtype in [cp.uint8, cp.int32, cp.float64]:
            image = cp.random.random((10, 10)).astype(dtype)

            labels = watershed(image, markers)

            assert labels.dtype == cp.int32
            assert cp.max(labels) == 2

    # Error handling tests

    def test_watershed_no_markers_error(self):
        """Test that ValueError is raised when markers is None."""
        image = cp.random.random((10, 10)).astype(cp.float32)

        with pytest.raises(ValueError, match="markers must be provided"):
            watershed(image, markers=None)

    def test_watershed_shape_mismatch_markers(self):
        """Test that ValueError is raised for shape mismatch."""
        image = cp.random.random((10, 10)).astype(cp.float32)
        markers = cp.zeros((8, 8), dtype=cp.int32)

        with pytest.raises(ValueError, match="markers shape"):
            watershed(image, markers)

    def test_watershed_shape_mismatch_mask(self):
        """Test that ValueError is raised for mask shape mismatch."""
        image = cp.random.random((10, 10)).astype(cp.float32)
        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[5, 5] = 1
        mask = cp.ones((8, 8), dtype=cp.bool_)

        with pytest.raises(ValueError, match="mask shape"):
            watershed(image, markers, mask=mask)

    def test_watershed_negative_markers(self):
        """Test that negative markers are supported (like scikit-image)."""
        image = cp.random.random((10, 10)).astype(cp.float32)
        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[3, 3] = -1  # Negative marker (e.g., for background)
        markers[7, 7] = 1  # Positive marker

        # Should work without error
        labels = watershed(image, markers)

        # Check that both negative and positive labels are present
        assert labels[3, 3] == -1
        assert labels[7, 7] == 1

        # Check that we have both label types in the result
        unique_labels = cp.unique(labels)
        assert cp.any(unique_labels < 0)  # Has negative labels
        assert cp.any(unique_labels > 0)  # Has positive labels

    def test_watershed_invalid_connectivity(self):
        """Test that ValueError is raised for invalid connectivity."""
        image = cp.random.random((10, 10)).astype(cp.float32)
        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[5, 5] = 1

        with pytest.raises(ValueError, match="connectivity must be 1 or 2"):
            watershed(image, markers, connectivity=3)

    def test_watershed_3d_not_implemented(self):
        """Test that NotImplementedError is raised for 3D images."""
        image = cp.random.random((10, 10, 10)).astype(cp.float32)
        markers = cp.zeros((10, 10, 10), dtype=cp.int32)
        markers[5, 5, 5] = 1

        with pytest.raises(NotImplementedError, match="Only 2D"):
            watershed(image, markers)

    def test_watershed_compactness_not_implemented(self):
        """Test that NotImplementedError is raised for compactness."""
        image = cp.random.random((10, 10)).astype(cp.float32)
        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[5, 5] = 1

        with pytest.raises(NotImplementedError, match="compactness"):
            watershed(image, markers, compactness=0.5)

    def test_watershed_line_not_implemented(self):
        """Test that NotImplementedError is raised for watershed_line."""
        image = cp.random.random((10, 10)).astype(cp.float32)
        markers = cp.zeros((10, 10), dtype=cp.int32)
        markers[5, 5] = 1

        with pytest.raises(NotImplementedError, match="watershed_line"):
            watershed(image, markers, watershed_line=True)

    def test_watershed_convergence(self):
        """Test that watershed converges (doesn't run forever)."""
        # Large image to test convergence
        image = cp.random.random((100, 100)).astype(cp.float32)

        # Multiple markers
        markers = cp.zeros((100, 100), dtype=cp.int32)
        markers[10, 10] = 1
        markers[50, 50] = 2
        markers[90, 90] = 3

        # Should complete without timing out
        labels = watershed(image, markers)

        # Check basic properties
        assert labels.shape == (100, 100)
        assert cp.max(labels) == 3
        assert cp.min(labels) >= 1

    def test_watershed_empty_markers(self):
        """Test watershed with no markers (all zeros)."""
        image = cp.random.random((10, 10)).astype(cp.float32)
        markers = cp.zeros((10, 10), dtype=cp.int32)

        # With no markers, all pixels should remain unlabeled (0)
        labels = watershed(image, markers)

        assert cp.all(labels == 0)

    def test_watershed_all_markers(self):
        """Test watershed where every pixel is a marker."""
        image = cp.random.random((5, 5)).astype(cp.float32)
        markers = cp.arange(1, 26, dtype=cp.int32).reshape(5, 5)

        # Every pixel is already labeled, should remain unchanged
        labels = watershed(image, markers)

        assert_array_equal(labels, markers)

    def test_watershed_numpy_input(self):
        """Test that watershed works with NumPy arrays as input."""
        import numpy as np

        # NumPy inputs
        image = np.random.random((10, 10)).astype(np.float32)
        markers = np.zeros((10, 10), dtype=np.int32)
        markers[3, 3] = 1
        markers[7, 7] = 2

        # Should convert and work
        labels = watershed(image, markers)

        # Result should be CuPy array
        assert isinstance(labels, cp.ndarray)
        assert labels.dtype == cp.int32
        assert cp.max(labels) == 2


class TestWatershedEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_watershed_small_image(self):
        """Test watershed on very small images."""
        # 2x2 image
        image = cp.array([[1, 2], [3, 4]], dtype=cp.float32)
        markers = cp.array([[1, 0], [0, 2]], dtype=cp.int32)

        labels = watershed(image, markers)

        assert labels.shape == (2, 2)
        assert labels[0, 0] == 1
        assert labels[1, 1] == 2

    def test_watershed_single_pixel(self):
        """Test watershed on 1x1 image."""
        image = cp.array([[5.0]], dtype=cp.float32)
        markers = cp.array([[1]], dtype=cp.int32)

        labels = watershed(image, markers)

        assert labels.shape == (1, 1)
        assert labels[0, 0] == 1

    def test_watershed_rectangular_image(self):
        """Test watershed on non-square rectangular images."""
        # Tall image
        image = cp.random.random((50, 10)).astype(cp.float32)
        markers = cp.zeros((50, 10), dtype=cp.int32)
        markers[10, 5] = 1
        markers[40, 5] = 2

        labels = watershed(image, markers)

        assert labels.shape == (50, 10)
        assert cp.max(labels) == 2

        # Wide image
        image = cp.random.random((10, 50)).astype(cp.float32)
        markers = cp.zeros((10, 50), dtype=cp.int32)
        markers[5, 10] = 1
        markers[5, 40] = 2

        labels = watershed(image, markers)

        assert labels.shape == (10, 50)
        assert cp.max(labels) == 2

    def test_watershed_border_markers(self):
        """Test watershed with markers at image borders."""
        image = cp.random.random((10, 10)).astype(cp.float32)
        markers = cp.zeros((10, 10), dtype=cp.int32)

        # Place markers at corners
        markers[0, 0] = 1
        markers[0, 9] = 2
        markers[9, 0] = 3
        markers[9, 9] = 4

        labels = watershed(image, markers)

        assert cp.max(labels) == 4
        assert labels[0, 0] == 1
        assert labels[0, 9] == 2
        assert labels[9, 0] == 3
        assert labels[9, 9] == 4
