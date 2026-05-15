/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace cucim::skimage
{

struct Point
{
    double row;
    double col;

    bool operator==(const Point& other) const
    {
        return row == other.row && col == other.col;
    }
};

constexpr std::size_t invalid_index = std::numeric_limits<std::size_t>::max();

struct Node
{
    Point point;
    long long key = 0;
    std::size_t next = invalid_index;
};

struct Contour
{
    std::size_t head = invalid_index;
    std::size_t tail = invalid_index;
    std::size_t size = 0;
    bool active = true;
};

struct ContourAssembly
{
    std::vector<Node> nodes;
    std::vector<Contour> contours;
};

std::string hello()
{
    return "hello from cucim.skimage._cucim_skimage_cpp_ext";
}

ContourAssembly build_contour_assembly(py::array_t<double, py::array::c_style | py::array::forcecast> segments,
                                       py::array_t<long long, py::array::c_style | py::array::forcecast> segment_keys)
{
    py::buffer_info info = segments.request();
    if (info.ndim != 3 || info.shape[1] != 2 || info.shape[2] != 2)
    {
        throw py::value_error("segments must have shape (n_segments, 2, 2)");
    }

    const auto n_segments = static_cast<std::size_t>(info.shape[0]);
    const double* data = static_cast<const double*>(info.ptr);
    py::buffer_info key_info = segment_keys.request();
    if (key_info.ndim != 2 || key_info.shape[0] != info.shape[0] || key_info.shape[1] != 2)
    {
        throw py::value_error("segment_keys must have shape (n_segments, 2)");
    }
    const long long* key_data = static_cast<const long long*>(key_info.ptr);

    ContourAssembly assembly;
    std::vector<Node>& nodes = assembly.nodes;
    std::vector<Contour>& contours = assembly.contours;
    std::unordered_map<long long, std::size_t> starts;
    std::unordered_map<long long, std::size_t> ends;
    nodes.reserve(2 * n_segments);
    contours.reserve(n_segments);
    starts.max_load_factor(0.7F);
    ends.max_load_factor(0.7F);
    starts.reserve(n_segments);
    ends.reserve(n_segments);

    auto add_node = [&nodes](const Point& point, long long key) {
        nodes.push_back(Node{ point, key });
        return nodes.size() - 1;
    };

    auto append_node = [&nodes, &add_node](Contour& contour, const Point& point, long long key) {
        const std::size_t node_index = add_node(point, key);
        nodes[contour.tail].next = node_index;
        contour.tail = node_index;
        ++contour.size;
    };

    auto prepend_node = [&nodes, &add_node](Contour& contour, const Point& point, long long key) {
        const std::size_t node_index = add_node(point, key);
        nodes[node_index].next = contour.head;
        contour.head = node_index;
        ++contour.size;
    };

    std::size_t current_index = 0;
    for (std::size_t i = 0; i < n_segments; ++i)
    {
        const double* segment = data + 4 * i;
        Point from_point{ segment[0], segment[1] };
        Point to_point{ segment[2], segment[3] };
        const long long from_key = key_data[2 * i];
        const long long to_key = key_data[2 * i + 1];

        // Ignore degenerate segments. This matches scikit-image's behavior for
        // vertices exactly on the requested level.
        if (from_point == to_point)
        {
            continue;
        }

        bool has_tail = false;
        bool has_head = false;
        std::size_t tail_index = 0;
        std::size_t head_index = 0;

        auto tail_iter = starts.find(to_key);
        if (tail_iter != starts.end())
        {
            has_tail = true;
            tail_index = tail_iter->second;
            starts.erase(tail_iter);
        }

        auto head_iter = ends.find(from_key);
        if (head_iter != ends.end())
        {
            has_head = true;
            head_index = head_iter->second;
            ends.erase(head_iter);
        }

        if (has_tail && has_head)
        {
            Contour& tail = contours[tail_index];
            Contour& head = contours[head_index];

            if (tail_index == head_index)
            {
                append_node(head, to_point, to_key);
            }
            else if (tail_index > head_index)
            {
                nodes[head.tail].next = tail.head;
                head.tail = tail.tail;
                head.size += tail.size;
                tail.active = false;
                starts[nodes[head.head].key] = head_index;
                ends[nodes[head.tail].key] = head_index;
            }
            else
            {
                starts.erase(nodes[head.head].key);
                nodes[head.tail].next = tail.head;
                tail.head = head.head;
                tail.size += head.size;
                head.active = false;
                starts[nodes[tail.head].key] = tail_index;
                ends[nodes[tail.tail].key] = tail_index;
            }
        }
        else if (!has_tail && !has_head)
        {
            const std::size_t from_node = add_node(from_point, from_key);
            const std::size_t to_node = add_node(to_point, to_key);
            nodes[from_node].next = to_node;
            contours.push_back(Contour{ from_node, to_node, 2, true });
            starts[from_key] = current_index;
            ends[to_key] = current_index;
            ++current_index;
        }
        else if (!has_head)
        {
            Contour& tail = contours[tail_index];
            prepend_node(tail, from_point, from_key);
            starts[from_key] = tail_index;
        }
        else
        {
            Contour& head = contours[head_index];
            append_node(head, to_point, to_key);
            ends[to_key] = head_index;
        }
    }

    return assembly;
}

std::pair<std::size_t, std::size_t> count_active_contours(const ContourAssembly& assembly)
{
    std::size_t active_contours = 0;
    std::size_t total_points = 0;
    for (const Contour& contour : assembly.contours)
    {
        if (contour.active)
        {
            ++active_contours;
            total_points += contour.size;
        }
    }
    return { active_contours, total_points };
}

// Assemble marching-squares output segments into ordered contours.
//
// Input is a C-contiguous float64 NumPy array with shape (n_segments, 2, 2).
// Each segment stores two endpoints in row/column coordinates:
//     segments[i, 0, :] == from_point
//     segments[i, 1, :] == to_point
//
// segment_keys is a C-contiguous int64 NumPy array with shape (n_segments, 2).
// Each key is a topological ID for the source grid edge containing the
// corresponding endpoint. The function links endpoints by these integer keys,
// preserving the same traversal-order behavior as scikit-image's Python
// implementation. It returns a Python list of NumPy arrays, where each array
// has shape (n_points, 2) and contains one open or closed contour in
// row/column order.
py::list assemble_contours(py::array_t<double, py::array::c_style | py::array::forcecast> segments,
                           py::array_t<long long, py::array::c_style | py::array::forcecast> segment_keys)
{
    ContourAssembly assembly = build_contour_assembly(segments, segment_keys);
    std::vector<Node>& nodes = assembly.nodes;
    std::vector<Contour>& contours = assembly.contours;
    const auto counts = count_active_contours(assembly);
    const std::size_t active_contours = counts.first;

    py::list output(active_contours);
    std::size_t output_index = 0;
    for (const Contour& contour : contours)
    {
        if (!contour.active)
        {
            continue;
        }
        py::array_t<double> contour_array({ static_cast<py::ssize_t>(contour.size), static_cast<py::ssize_t>(2) });
        py::buffer_info contour_info = contour_array.request();
        double* contour_data = static_cast<double*>(contour_info.ptr);
        std::size_t j = 0;
        for (std::size_t node_index = contour.head; node_index != invalid_index; node_index = nodes[node_index].next)
        {
            const Point& point = nodes[node_index].point;
            contour_data[2 * j] = point.row;
            contour_data[2 * j + 1] = point.col;
            ++j;
        }
        output[output_index] = std::move(contour_array);
        ++output_index;
    }
    return output;
}

// Return contours in packed form:
//
//     points:  float64 array, shape (total_points, 2)
//     offsets: int64 array, shape (n_contours + 1,)
//
// Contour i is points[offsets[i]:offsets[i + 1]]. This avoids allocating one
// NumPy array per contour and gives callers a single contiguous coordinate
// array that can be transferred to another backend in one operation.
py::tuple assemble_contours_packed(py::array_t<double, py::array::c_style | py::array::forcecast> segments,
                                   py::array_t<long long, py::array::c_style | py::array::forcecast> segment_keys)
{
    ContourAssembly assembly = build_contour_assembly(segments, segment_keys);
    std::vector<Node>& nodes = assembly.nodes;
    std::vector<Contour>& contours = assembly.contours;
    const auto [active_contours, total_points] = count_active_contours(assembly);

    py::array_t<double> points(
        std::vector<py::ssize_t>{ static_cast<py::ssize_t>(total_points), static_cast<py::ssize_t>(2) },
        std::vector<py::ssize_t>{ static_cast<py::ssize_t>(2 * sizeof(double)),
                                  static_cast<py::ssize_t>(sizeof(double)) });
    py::array_t<long long> offsets(std::vector<py::ssize_t>{ static_cast<py::ssize_t>(active_contours + 1) },
                                   std::vector<py::ssize_t>{ static_cast<py::ssize_t>(sizeof(long long)) });
    py::buffer_info points_info = points.request();
    py::buffer_info offsets_info = offsets.request();
    double* points_data = static_cast<double*>(points_info.ptr);
    long long* offsets_data = static_cast<long long*>(offsets_info.ptr);

    std::size_t output_index = 0;
    std::size_t point_index = 0;
    offsets_data[0] = 0;
    for (const Contour& contour : contours)
    {
        if (!contour.active)
        {
            continue;
        }
        for (std::size_t node_index = contour.head; node_index != invalid_index; node_index = nodes[node_index].next)
        {
            const Point& point = nodes[node_index].point;
            points_data[2 * point_index] = point.row;
            points_data[2 * point_index + 1] = point.col;
            ++point_index;
        }
        ++output_index;
        offsets_data[output_index] = static_cast<long long>(point_index);
    }
    return py::make_tuple(std::move(points), std::move(offsets));
}

} // namespace cucim::skimage

PYBIND11_MODULE(_cucim_skimage_cpp_ext, m)
{
    m.doc() = "Optional compiled helpers for cucim.skimage.";
    m.def("hello", &cucim::skimage::hello, "Return a test string from the C++ extension.");
    m.def("assemble_contours", &cucim::skimage::assemble_contours,
          "Assemble marching-squares line segments into contours.");
    m.def("assemble_contours_packed", &cucim::skimage::assemble_contours_packed,
          "Assemble marching-squares line segments into a packed points/offsets representation.");
}
