/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>

#include <cstdint>

namespace py = pybind11;

namespace cucim::skimage
{

void segmented_radix_sort_keys_uint8(std::uintptr_t keys_in,
                                     std::uintptr_t keys_out,
                                     std::uintptr_t segment_offsets,
                                     std::int64_t num_items,
                                     std::int64_t num_segments,
                                     std::uintptr_t stream_ptr);
void segmented_radix_sort_keys_ranges_uint8(std::uintptr_t keys_in,
                                            std::uintptr_t keys_out,
                                            std::uintptr_t begin_offsets,
                                            std::uintptr_t end_offsets,
                                            std::int64_t num_items,
                                            std::int64_t num_segments,
                                            std::uintptr_t stream_ptr);

void segmented_radix_sort_keys_uint16(std::uintptr_t keys_in,
                                      std::uintptr_t keys_out,
                                      std::uintptr_t segment_offsets,
                                      std::int64_t num_items,
                                      std::int64_t num_segments,
                                      std::uintptr_t stream_ptr);
void segmented_radix_sort_keys_ranges_uint16(std::uintptr_t keys_in,
                                             std::uintptr_t keys_out,
                                             std::uintptr_t begin_offsets,
                                             std::uintptr_t end_offsets,
                                             std::int64_t num_items,
                                             std::int64_t num_segments,
                                             std::uintptr_t stream_ptr);

void segmented_radix_sort_keys_float32(std::uintptr_t keys_in,
                                       std::uintptr_t keys_out,
                                       std::uintptr_t segment_offsets,
                                       std::int64_t num_items,
                                       std::int64_t num_segments,
                                       std::uintptr_t stream_ptr);
void segmented_radix_sort_keys_ranges_float32(std::uintptr_t keys_in,
                                              std::uintptr_t keys_out,
                                              std::uintptr_t begin_offsets,
                                              std::uintptr_t end_offsets,
                                              std::int64_t num_items,
                                              std::int64_t num_segments,
                                              std::uintptr_t stream_ptr);

void segmented_radix_sort_keys_float64(std::uintptr_t keys_in,
                                       std::uintptr_t keys_out,
                                       std::uintptr_t segment_offsets,
                                       std::int64_t num_items,
                                       std::int64_t num_segments,
                                       std::uintptr_t stream_ptr);
void segmented_radix_sort_keys_ranges_float64(std::uintptr_t keys_in,
                                              std::uintptr_t keys_out,
                                              std::uintptr_t begin_offsets,
                                              std::uintptr_t end_offsets,
                                              std::int64_t num_items,
                                              std::int64_t num_segments,
                                              std::uintptr_t stream_ptr);

} // namespace cucim::skimage

PYBIND11_MODULE(_cucim_skimage_cpp_ext, m)
{
    m.doc() = "Optional compiled helpers for cucim.skimage.";
    m.def("segmented_radix_sort_keys_uint8", &cucim::skimage::segmented_radix_sort_keys_uint8, py::arg("keys_in"),
          py::arg("keys_out"), py::arg("segment_offsets"), py::arg("num_items"), py::arg("num_segments"),
          py::arg("stream_ptr") = 0,
          "Sort uint8 keys independently within each segment using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_uint16", &cucim::skimage::segmented_radix_sort_keys_uint16, py::arg("keys_in"),
          py::arg("keys_out"), py::arg("segment_offsets"), py::arg("num_items"), py::arg("num_segments"),
          py::arg("stream_ptr") = 0,
          "Sort uint16 keys independently within each segment using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_float32", &cucim::skimage::segmented_radix_sort_keys_float32, py::arg("keys_in"),
          py::arg("keys_out"), py::arg("segment_offsets"), py::arg("num_items"), py::arg("num_segments"),
          py::arg("stream_ptr") = 0,
          "Sort float32 keys independently within each segment using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_float64", &cucim::skimage::segmented_radix_sort_keys_float64, py::arg("keys_in"),
          py::arg("keys_out"), py::arg("segment_offsets"), py::arg("num_items"), py::arg("num_segments"),
          py::arg("stream_ptr") = 0,
          "Sort float64 keys independently within each segment using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_ranges_uint8", &cucim::skimage::segmented_radix_sort_keys_ranges_uint8,
          py::arg("keys_in"), py::arg("keys_out"), py::arg("begin_offsets"), py::arg("end_offsets"),
          py::arg("num_items"), py::arg("num_segments"), py::arg("stream_ptr") = 0,
          "Sort selected uint8 key ranges using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_ranges_uint16", &cucim::skimage::segmented_radix_sort_keys_ranges_uint16,
          py::arg("keys_in"), py::arg("keys_out"), py::arg("begin_offsets"), py::arg("end_offsets"),
          py::arg("num_items"), py::arg("num_segments"), py::arg("stream_ptr") = 0,
          "Sort selected uint16 key ranges using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_ranges_float32", &cucim::skimage::segmented_radix_sort_keys_ranges_float32,
          py::arg("keys_in"), py::arg("keys_out"), py::arg("begin_offsets"), py::arg("end_offsets"),
          py::arg("num_items"), py::arg("num_segments"), py::arg("stream_ptr") = 0,
          "Sort selected float32 key ranges using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_ranges_float64", &cucim::skimage::segmented_radix_sort_keys_ranges_float64,
          py::arg("keys_in"), py::arg("keys_out"), py::arg("begin_offsets"), py::arg("end_offsets"),
          py::arg("num_items"), py::arg("num_segments"), py::arg("stream_ptr") = 0,
          "Sort selected float64 key ranges using cub::DeviceSegmentedRadixSort.");
}
