/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PYCUCIM_CUFILE_PYDOC_H
#define PYCUCIM_CUFILE_PYDOC_H

#include "../macros.h"

namespace cucim::filesystem::doc::CuFileDriver
{

// CuFileDriver(int fd, bool no_gds = false, bool use_mmap = false, const char* file_path = "");
PYDOC(CuFileDriver, R"doc(
Constructor of CuFileDriver.

Args:
    fd: A file descriptor (in `int` type) which is available through `os.open()` method.
    no_gds: If True, use POSIX APIs only even when GDS can be supported for the file.
    use_mmap: Use memory-mapped I/O with a read-only file descriptor. The backing file must retain
        its size while mapped. Truncating it can cause SIGBUS on a subsequent read. Defaults to `False`.
    file_path: A file path for the file descriptor, as `str`, `bytes`, or `os.PathLike`. It would retrieve the absolute file path of the file descriptor if not specified.
)doc")

// ssize_t pread(void* buf, size_t count, off_t file_offset, off_t buf_offset = 0);
PYDOC(pread, R"doc(
Reads up to `count` bytes from the file driver at offset `file_offset` (from the start of the file) into the buffer
`buf` starting at offset `buf_offset`. The file offset is not changed.

Args:
    buf: A buffer where read bytes are stored. Buffer can be either in CPU memory or (CUDA) GPU memory.
    count: The number of bytes to read.
    file_offset: An offset from the start of the file.
    buf_offset: An offset from the start of the buffer. Default value is 0.
Returns:
    The number of bytes read if succeed, -1 otherwise.
)doc")

// ssize_t pwrite(const void* buf, size_t count, off_t file_offset, off_t buf_offset = 0);
PYDOC(pwrite, R"doc(
Writes up to `count` bytes from the buffer `buf` starting at offset `buf_offset` to the file at `file_offset`.
The file offset is not changed.

Args:
    buf: A buffer where write bytes come from. Buffer can be either in CPU memory or (CUDA) GPU memory.
    count: The number of bytes to write.
    file_offset: An offset from the start of the file.
    buf_offset: An offset from the start of the buffer. Default value is 0.
Returns:
    The number of bytes written, including a partial count if a later operation fails, or -1 if the write fails
    before writing any bytes.
)doc")

// bool close();
PYDOC(close, R"doc(
Closes opened file if not closed.
)doc")

} // namespace cucim::filesystem::doc::CuFileDriver

#endif // PYCUCIM_CUFILE_PYDOC_H
