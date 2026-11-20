/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/filesystem/cufile_driver.h"

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "cucim/util/cuda.h"
#include "cucim/util/platform.h"
#include "cufile_stub.h"

#define ALIGN_UP(x, align_to) (((uint64_t)(x) + ((uint64_t)(align_to)-1)) & ~((uint64_t)(align_to)-1))
#define ALIGN_DOWN(x, align_to) ((uint64_t)(x) & ~((uint64_t)(align_to)-1))


namespace cucim::filesystem
{
static constexpr unsigned int PAGE_SIZE = 4096;
static constexpr uint64_t DEFAULT_MAX_CACHE_SIZE = 128 << 20; // 128MiB
static CuFileStub s_cufile_stub;
static CuFileDriverInitializer s_cufile_initializer;
thread_local static CuFileDriverCache s_cufile_cache;
Mutex CuFileDriver::driver_mutex_;


static std::string get_fd_path(int fd)
{
    pid_t pid = getpid();
    ssize_t file_path_len = 0;

    char real_path[PATH_MAX];

    std::string src_path = fmt::format("/proc/{}/fd/{}", pid, fd);

    if ((file_path_len = readlink(src_path.c_str(), real_path, PATH_MAX - 1)) > 0)
    {
        real_path[file_path_len] = '\0';
    }
    else
    {
        throw std::runtime_error(fmt::format("Cannot get the real path from process entries ({})", strerror(errno)));
    }

    return std::string(real_path);
}

static int get_file_flags(const char* flags)
{
    int file_flags = -1;
    if (flags == nullptr || flags[0] == '\0')
    {
        return -1;
    }
    switch (flags[0])
    {
    case 'r':
        file_flags = O_RDONLY;
        if (flags[1] == '+')
        {
            file_flags = O_RDWR;
        }
        break;
    case 'w':
        file_flags = O_RDWR | O_CREAT | O_TRUNC;
        break;
    case 'a':
        file_flags = O_RDWR | O_CREAT;
        break;
    default:
        return -1;
    }


    file_flags |= O_CLOEXEC;

    return file_flags;
}

bool is_gds_available()
{
    return static_cast<bool>(s_cufile_initializer) && !cucim::util::is_in_wsl();
}

std::shared_ptr<CuFileDriver> open(const char* file_path, const char* flags, mode_t mode)
{
    bool use_o_direct = true;
    bool no_gds = false;
    bool use_mmap = false;
    int file_flags = get_file_flags(flags);

    for (const char* ch = (flags[1] == '+' ? &flags[2] : &flags[1]); *ch; ch++)
        switch (*ch)
        {
        case 'n':
            use_o_direct = false;
            break;
        case 'p':
            no_gds = true;
            break;
        case 'm':
            use_mmap = true;
            break;
        }
    if (use_o_direct)
    {
        file_flags |= O_DIRECT;
    }

    if (file_flags < 0)
    {
        return std::shared_ptr<CuFileDriver>();
    }

    FileHandleType file_type = (file_flags & O_DIRECT ? FileHandleType::kPosixODirect : FileHandleType::kPosix);

    int fd = ::open(file_path, file_flags, mode);
    if (fd < 0)
    {
        if (errno == ENOENT)
        {
            throw std::invalid_argument(fmt::format("File '{}' doesn't exist!", file_path));
        }
        if (file_type == FileHandleType::kPosix)
        {
            throw std::invalid_argument(fmt::format("File '{}' cannot be open!", file_path));
        }
        else // if kFileHandlePosixODirect
        {
            file_flags &= ~O_DIRECT;
            fd = ::open(file_path, file_flags, mode);
            fmt::print(
                stderr, "The file {} doesn't support O_DIRECT. Trying to open the file without O_DIRECT\n", file_path);
            if (fd < 0)
            {
                throw std::invalid_argument(fmt::format("File '{}' cannot be open!", file_path));
            }
            file_type = FileHandleType::kPosix; // POSIX
        }
    }

    const auto handle = std::make_shared<CuFileDriver>(fd, no_gds, use_mmap, file_path);

    // Set ownership to the file descriptor
    if (handle->handle_)
    {
        handle->handle_->own_fd = true;
    }

    return handle;
}

std::shared_ptr<CuFileDriver> open(int fd, bool no_gds, bool use_mmap)
{
    return std::make_shared<CuFileDriver>(fd, no_gds, use_mmap, nullptr);
}

CuFileDriver::CuFileDriver(int fd, bool no_gds, bool use_mmap, const char* file_path)
{
    if (file_path == nullptr || *file_path == '\0')
    {
        file_path_ = get_fd_path(fd);
    }
    else
    {
        file_path_ = file_path;
    }

    struct stat st;
    fstat(fd, &st);
    file_size_ = st.st_size;

    int flags;
    // Note: the following method cannot detect flags such as O_EXCL and O_TRUNC.
    flags = fcntl(fd, F_GETFL);
    if (flags < 0)
    {
        throw std::runtime_error(fmt::format("[Error] fcntl failed for fd {} ({})", fd, std::strerror(errno)));
    }
    file_flags_ = flags;

    FileHandleType file_type = (flags & O_DIRECT) ? FileHandleType::kPosixODirect : FileHandleType::kPosix;
    // Copy file path (Allocated memory would be freed at close() method.)
    char* file_path_cstr = static_cast<char*>(cucim_malloc(file_path_.size() + 1));
    memcpy(file_path_cstr, file_path_.c_str(), file_path_.size());
    file_path_cstr[file_path_.size()] = '\0';
    handle_ = std::make_shared<CuCIMFileHandle>(fd, nullptr, file_type, const_cast<char*>(file_path_cstr), this,
                                                static_cast<uint64_t>(st.st_dev), static_cast<uint64_t>(st.st_ino),
                                                static_cast<int64_t>(st.st_mtim.tv_nsec), false);

    CUfileError_t status;
    CUfileDescr_t cf_descr{}; // It is important to set zero!

    if ((file_type == FileHandleType::kPosixODirect || file_type == FileHandleType::kGPUDirect) && !no_gds &&
        !use_mmap && s_cufile_initializer)
    {
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&handle_->cufile, &cf_descr);
        if (status.err == CU_FILE_SUCCESS)
        {
            handle_->type = FileHandleType::kGPUDirect;
        }
        else
        {
            fmt::print(
                stderr,
                "[Error] cuFileHandleRegister fd: {} ({}), status: {}. Would work with cuCIM's compatibility mode.\n",
                fd, file_path_, cufileop_status_error(status.err));
        }
    }
    else if (use_mmap)
    {
        if (flags & (O_RDWR || O_WRONLY))
        {
            throw std::runtime_error(
                fmt::format("[Error] Memory-mapped IO for writable file descriptor is not supported!"));
        }

        mmap_ptr_ = mmap((void*)0, file_size_, PROT_READ, MAP_SHARED, fd, 0);
        if (mmap_ptr_ != MAP_FAILED)
        {
            handle_->type = FileHandleType::kMemoryMapped;
        }
        else
        {
            mmap_ptr_ = nullptr;
            throw std::runtime_error(fmt::format("[Error] failed to call mmap ({})", std::strerror(errno)));
        }
    }
}

bool close(const std::shared_ptr<CuFileDriver>& fd)
{
    return fd->close();
}
ssize_t pread(const std::shared_ptr<CuFileDriver>& fd, void* buf, size_t count, off_t file_offset, off_t buf_offset)
{
    if (fd != nullptr)
    {
        return fd->pread(buf, count, file_offset, buf_offset);
    }
    else
    {
        fmt::print(stderr, "fd (CuFileDriver) is null!\n");
        return -1;
    }
}
ssize_t pwrite(const std::shared_ptr<CuFileDriver>& fd, const void* buf, size_t count, off_t file_offset, off_t buf_offset)
{
    if (fd != nullptr)
    {
        return fd->pwrite(buf, count, file_offset, buf_offset);
    }
    else
    {
        fmt::print(stderr, "fd (CuFileDriver) is null!\n");
        return -1;
    }
}

bool discard_page_cache(const char* file_path)
{
    int fd = ::open(file_path, O_RDONLY);
    if (fd < 0)
    {
        return false;
    }
    if (::fdatasync(fd) < 0)
    {
        return false;
    }
    if (::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) < 0)
    {
        return false;
    }
    if (::close(fd) < 0)
    {
        return false;
    }
    return true;
}


CuFileDriverInitializer::CuFileDriverInitializer() : max_host_cache_size_(DEFAULT_MAX_CACHE_SIZE)
{
    // Initialize libcufile library
    s_cufile_stub.load();

    CUfileError_t status = cuFileDriverOpen();
    if (status.err == CU_FILE_SUCCESS)
    {
        is_available_ = true;
        CUfileDrvProps_t props;

        status = cuFileDriverGetProperties(&props);
        if (status.err == CU_FILE_SUCCESS)
        {
            // kb -> bytes
            max_device_cache_size_ = static_cast<uint64_t>(props.max_device_cache_size) << 10;
            // POSIX staging needs a nonzero cache even when cuFile reports no device cache.
            if (max_device_cache_size_ != 0)
            {
                max_host_cache_size_ = max_device_cache_size_;
            }
        }
        else
        {
            fmt::print(stderr, "cuFileDriverGetProperties() failed!\n");
        }
        // fmt::print(stderr, "CuFileDriver opened!\n");
    }
    else
    {
        is_available_ = false;
        max_device_cache_size_ = DEFAULT_MAX_CACHE_SIZE;

        // fmt::print(stderr, "[Warning] CuFileDriver cannot be open. Falling back to use POSIX file IO APIs.\n");
    }
}
CuFileDriverInitializer::~CuFileDriverInitializer()
{
    if (is_available_)
    {
        CUfileError_t status = cuFileDriverClose();
        if (status.err != CU_FILE_SUCCESS)
        {
            fmt::print(stderr, "Unable to close cuFileDriver ({})\n", cufileop_status_error(status.err));
        }
        else
        {
            // fmt::print(stderr, "CuFileDriver closed!\n");
        }
        is_available_ = false;
    }

    // Close cufile stub
    s_cufile_stub.unload();
}

CuFileDriverCache::CuFileDriverCache()
{
}
void* CuFileDriverCache::device_cache()
{
    if (device_cache_)
    {
        return device_cache_aligned_;
    }
    else
    {
        cudaError_t cuda_status;
        unsigned int cache_size = s_cufile_initializer.max_device_cache_size();
        CUDA_TRY(cudaMalloc(&device_cache_, PAGE_SIZE + cache_size));
        if (cuda_status)
        {
            throw std::bad_alloc();
        }
        device_cache_aligned_ = reinterpret_cast<void*>(ALIGN_UP(device_cache_, PAGE_SIZE));
        CUfileError_t status = cuFileBufRegister(device_cache_aligned_, cache_size, 0);
        if (status.err != CU_FILE_SUCCESS)
        {
            CUDA_TRY(cudaFree(device_cache_));
            device_cache_ = nullptr;
            device_cache_aligned_ = nullptr;
            if (cuda_status)
            {
                throw std::bad_alloc();
            }
            throw std::runtime_error("Failed to call cuFileBufRegister()!");
        }

        return device_cache_aligned_;
    }
}
void* CuFileDriverCache::host_cache()
{
    if (host_cache_)
    {
        return host_cache_aligned_;
    }
    else
    {
        if (posix_memalign(&host_cache_, PAGE_SIZE, s_cufile_initializer.max_host_cache_size()))
        {
            throw std::bad_alloc();
        }
        host_cache_aligned_ = host_cache_;

        return host_cache_aligned_;
    }
}
CuFileDriverCache::~CuFileDriverCache()
{

    if (device_cache_)
    {
        cudaError_t cuda_status;
        CUfileError_t status = cuFileBufDeregister(device_cache_aligned_);
        if (status.err != CU_FILE_SUCCESS)
        {
            fmt::print(stderr, "Failed on cuFileBufDeregister()! (status: {})\n", cufileop_status_error(status.err));
        }
        CUDA_TRY(cudaFree(device_cache_));
        if (cuda_status)
        {
            fmt::print(stderr, "Failed on cudaFree()!\n");
        }
        device_cache_ = nullptr;
        device_cache_aligned_ = nullptr;
    }
    if (host_cache_)
    {
        free(host_cache_);
        host_cache_ = nullptr;
        host_cache_aligned_ = nullptr;
    }
}
ssize_t CuFileDriver::pread(void* buf, size_t count, off_t file_offset, off_t buf_offset) const
{
    if (file_flags_ == -1)
    {
        fmt::print(stderr, "File is not open yet.\n");
        return -1;
    }
    if ((file_flags_ & O_ACCMODE) == O_WRONLY)
    {
        fmt::print(stderr, "The file is open with write-only mode!\n");
        return -1;
    }

    cudaError_t cuda_status;

    cudaPointerAttributes attributes;
    cudaMemoryType memory_type;

    FileHandleType file_type = handle_->type;

    CUDA_TRY(cudaPointerGetAttributes(&attributes, buf));
    if (cuda_status)
    {
        //        if (cuda_status == cudaErrorInvalidValue)
        //        {
        //            attributes.type = cudaMemoryTypeDevice;
        //        }
        //        else
        //        {
        return -1;
        //        }
    }
    memory_type = attributes.type;

    bool staged_copy_issued = false;
    auto copy_to_device = [&](void* dest, const void* src, size_t size) {
        staged_copy_issued |= size != 0;
        CUDA_TRY(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
        return cuda_status == cudaSuccess;
    };
    auto finish_read = [&](ssize_t result) -> ssize_t {
        if (staged_copy_issued)
        {
            // Finish default-stream copies even if a later read or copy failed.
            const int saved_errno = errno;
            CUDA_TRY(cudaStreamSynchronize(nullptr));
            errno = saved_errno;
            if (cuda_status)
            {
                return -1;
            }
        }
        return result;
    };

    auto copy_from_host = [&](uint8_t* dest, const uint8_t* src, size_t size) {
        if (size == 0)
        {
            return true;
        }
        if (memory_type != cudaMemoryTypeUnregistered)
        {
            return copy_to_device(dest, src, size);
        }
        memcpy(dest, src, size);
        return true;
    };

    auto report_read_error = [&](off_t offset) {
        const int saved_errno = errno;
        fmt::print(stderr, "Cannot read the file content at offset {}! ({})\n", offset, std::strerror(saved_errno));
        errno = saved_errno;
        return ssize_t{ -1 };
    };

    auto read_staged = [&](uint8_t* dest, size_t size, off_t offset) -> ssize_t {
        const size_t cache_size = s_cufile_initializer.max_host_cache_size();
        auto* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());
        size_t copied = 0;
        while (copied < size)
        {
            const size_t chunk_size = std::min(cache_size, size - copied);
            const off_t chunk_offset = offset + static_cast<off_t>(copied);
            const ssize_t read_cnt = ::pread(handle_->fd, cache_buf, chunk_size, chunk_offset);
            if (read_cnt < 0)
            {
                return report_read_error(chunk_offset);
            }
            if (read_cnt == 0)
            {
                break;
            }
            if (!copy_from_host(dest + copied, cache_buf, read_cnt))
            {
                return -1;
            }
            copied += read_cnt;
            if (static_cast<size_t>(read_cnt) < chunk_size)
            {
                break;
            }
        }
        return copied;
    };

    // The aligned read may contain bytes before the range requested by the caller.
    auto copy_available = [&](uint8_t* dest, const uint8_t* src, ssize_t read_cnt, off_t read_offset, size_t skip,
                              size_t size) -> ssize_t {
        if (read_cnt < 0)
        {
            return report_read_error(read_offset);
        }
        const auto available = static_cast<size_t>(read_cnt);
        const size_t copied = available > skip ? std::min(size, available - skip) : 0;
        return copy_from_host(dest, src + skip, copied) ? static_cast<ssize_t>(copied) : -1;
    };

    auto read_page = [&](uint8_t* dest, size_t size, off_t offset, size_t skip) -> ssize_t {
        uint8_t page[PAGE_SIZE * 2];
        auto* aligned_page = reinterpret_cast<uint8_t*>(ALIGN_UP(page, PAGE_SIZE));
        const ssize_t read_cnt = ::pread(handle_->fd, aligned_page, PAGE_SIZE, offset);
        return copy_available(dest, aligned_page, read_cnt, offset, skip, size);
    };

    auto* output_buf = static_cast<uint8_t*>(buf) + buf_offset;
    if (file_type == FileHandleType::kPosix)
    {
        if (memory_type != cudaMemoryTypeUnregistered)
        {
            return count == 0 ? 0 : finish_read(read_staged(output_buf, count, file_offset));
        }
        return ::pread(handle_->fd, output_buf, count, file_offset);
    }
    if (file_type == FileHandleType::kMemoryMapped)
    {
        if (file_offset < 0)
        {
            errno = EINVAL;
            return -1;
        }
        // The mapping covers the file size at open time; reading past it would fault.
        const auto offset = static_cast<size_t>(file_offset);
        if (offset >= file_size_)
        {
            return 0;
        }
        const size_t available = std::min(count, file_size_ - offset);
        return finish_read(copy_from_host(output_buf, static_cast<uint8_t*>(mmap_ptr_) + offset, available) ?
                               static_cast<ssize_t>(available) :
                               -1);
    }
    if (memory_type == cudaMemoryTypeUnregistered || file_type == FileHandleType::kPosixODirect)
    {
        if (file_offset < 0)
        {
            errno = EINVAL;
            return -1;
        }
        if (count == 0)
        {
            return 0;
        }

        const bool is_aligned = reinterpret_cast<uintptr_t>(output_buf) % PAGE_SIZE == 0 && file_offset % PAGE_SIZE == 0;
        if (is_aligned)
        {
            const size_t block_size = ALIGN_DOWN(count, PAGE_SIZE);
            if (block_size > 0)
            {
                const ssize_t read_cnt = memory_type == cudaMemoryTypeUnregistered ?
                                             ::pread(handle_->fd, output_buf, block_size, file_offset) :
                                             read_staged(output_buf, block_size, file_offset);
                if (read_cnt < 0 || static_cast<size_t>(read_cnt) < block_size)
                {
                    return finish_read(read_cnt);
                }
            }

            const size_t tail_size = count - block_size;
            const ssize_t tail_read = tail_size ? read_page(output_buf + block_size, tail_size,
                                                            file_offset + static_cast<off_t>(block_size), 0) :
                                                  0;
            return finish_read(tail_read < 0 ? -1 : static_cast<ssize_t>(block_size) + tail_read);
        }

        const off_t first_page = ALIGN_DOWN(file_offset, PAGE_SIZE);
        const size_t page_offset = file_offset - first_page;
        const size_t aligned_size = ALIGN_UP(page_offset + count, PAGE_SIZE);
        if (aligned_size <= s_cufile_initializer.max_host_cache_size())
        {
            auto* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());
            const ssize_t read_cnt = ::pread(handle_->fd, cache_buf, aligned_size, first_page);
            return finish_read(copy_available(output_buf, cache_buf, read_cnt, first_page, page_offset, count));
        }

        const size_t head_size = std::min(count, PAGE_SIZE - page_offset);
        const ssize_t head_read = read_page(output_buf, head_size, first_page, page_offset);
        if (head_read < 0 || static_cast<size_t>(head_read) < head_size)
        {
            return finish_read(head_read);
        }

        const size_t body_size = ALIGN_DOWN(count - head_size, PAGE_SIZE);
        if (body_size > 0)
        {
            const ssize_t body_read = read_staged(output_buf + head_size, body_size, first_page + PAGE_SIZE);
            if (body_read < 0 || static_cast<size_t>(body_read) < body_size)
            {
                return finish_read(body_read < 0 ? -1 : head_read + body_read);
            }
        }

        const size_t tail_size = count - head_size - body_size;
        const ssize_t tail_read = tail_size ? read_page(output_buf + head_size + body_size, tail_size,
                                                        first_page + PAGE_SIZE + static_cast<off_t>(body_size), 0) :
                                              0;
        return finish_read(tail_read < 0 ? -1 : head_read + static_cast<ssize_t>(body_size) + tail_read);
    }
    if (file_type == FileHandleType::kGPUDirect)
    {
        (void*)s_cufile_cache.device_cache(); // Lazy initialization

        const ssize_t read_cnt = cuFileRead(handle_->cufile, output_buf, count, file_offset, 0);
        if (read_cnt < 0)
        {
            fmt::print(stderr, "Failed to read file with cuFileRead().\n");
            return -1;
        }
        return read_cnt;
    }

    return 0;
}
ssize_t CuFileDriver::pwrite(const void* buf, size_t count, off_t file_offset, off_t buf_offset)
{
    if (file_flags_ == -1)
    {
        fmt::print(stderr, "File is not open yet.\n");
        return -1;
    }
    if ((file_flags_ & O_ACCMODE) == O_RDONLY)
    {
        fmt::print(stderr, "The file is open with read-only mode!\n");
        return -1;
    }

    cudaError_t cuda_status;
    ssize_t total_write_cnt = 0;

    cudaPointerAttributes attributes;
    cudaMemoryType memory_type;

    FileHandleType file_type = handle_->type;

    CUDA_TRY(cudaPointerGetAttributes(&attributes, buf));
    if (cuda_status)
    {
        return -1;
    }
    memory_type = attributes.type;

    auto finish_write = [&](bool failed = false) {
        if (total_write_cnt > 0)
        {
            // close() truncates O_DIRECT padding to this logical size.
            file_size_ = std::max(file_size_, file_offset + static_cast<size_t>(total_write_cnt));
        }
        return failed && total_write_cnt == 0 ? -1 : total_write_cnt;
    };

    // Stop at the first error or short write. With O_DIRECT, the next offset after a short write may be
    // unaligned. Preserve the bytes written by earlier chunks if a later operation fails.
    auto write_staged = [&](const uint8_t* src, size_t size, off_t offset) -> ssize_t {
        const size_t cache_size = s_cufile_initializer.max_host_cache_size();
        auto* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());
        size_t written = 0;
        while (written < size)
        {
            const size_t chunk_size = std::min(cache_size, size - written);
            if (memory_type == cudaMemoryTypeUnregistered)
            {
                memcpy(cache_buf, src + written, chunk_size);
            }
            else
            {
                CUDA_TRY(cudaMemcpy(cache_buf, src + written, chunk_size, cudaMemcpyDeviceToHost));
                if (cuda_status)
                {
                    return written > 0 ? static_cast<ssize_t>(written) : -1;
                }
            }
            const off_t chunk_offset = offset + static_cast<off_t>(written);
            const ssize_t write_cnt = ::pwrite(handle_->fd, cache_buf, chunk_size, chunk_offset);
            if (write_cnt < 0)
            {
                fmt::print(
                    stderr, "Cannot write the file content at offset {}! ({})\n", chunk_offset, std::strerror(errno));
                return written > 0 ? static_cast<ssize_t>(written) : -1;
            }
            written += write_cnt;
            if (static_cast<size_t>(write_cnt) < chunk_size)
            {
                break;
            }
        }
        return static_cast<ssize_t>(written);
    };

    // The number of bytes in [skip, skip + size) of a block that a write of write_cnt bytes covered.
    auto covered = [](ssize_t write_cnt, size_t skip, size_t size) -> size_t {
        const auto block_written = static_cast<size_t>(write_cnt);
        return block_written > skip ? std::min(size, block_written - skip) : 0;
    };

    // Set when a write stops early, so that the later parts of the request are not written.
    bool short_write = false;

    if (file_type == FileHandleType::kPosix)
    {
        if (memory_type != cudaMemoryTypeUnregistered)
        {
            total_write_cnt = write_staged(static_cast<const uint8_t*>(buf) + buf_offset, count, file_offset);
            if (total_write_cnt < 0)
            {
                return -1;
            }
        }
        else
        {
            total_write_cnt = ::pwrite(handle_->fd, reinterpret_cast<const char*>(buf) + buf_offset, count, file_offset);
        }
    }
    else if (file_type == FileHandleType::kMemoryMapped)
    {
        fmt::print(stderr, "[Error] pwrite() is not supported for Memory-mapped IO file type!\n");
        return -1;
    }
    else if (memory_type == cudaMemoryTypeUnregistered || handle_->type == FileHandleType::kPosixODirect)
    {
        uint64_t buf_align = (reinterpret_cast<uint64_t>(buf) + buf_offset) % PAGE_SIZE;
        bool is_aligned = (buf_align == 0) && ((file_offset % PAGE_SIZE) == 0);

        if (is_aligned)
        {
            ssize_t write_cnt;
            size_t block_write_size = ALIGN_DOWN(count, PAGE_SIZE);

            if (block_write_size > 0)
            {
                if (memory_type == cudaMemoryTypeUnregistered)
                {
                    write_cnt = ::pwrite(
                        handle_->fd, reinterpret_cast<const char*>(buf) + buf_offset, block_write_size, file_offset);
                    if (write_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot write the file content block! ({})\n", std::strerror(errno));
                        return -1;
                    }
                }
                else
                {
                    write_cnt =
                        write_staged(static_cast<const uint8_t*>(buf) + buf_offset, block_write_size, file_offset);
                    if (write_cnt < 0)
                    {
                        return -1;
                    }
                }
                total_write_cnt += write_cnt;
                short_write = static_cast<size_t>(write_cnt) < block_write_size;
            }

            size_t remaining = count - block_write_size;
            if (remaining && !short_write)
            {
                uint8_t internal_buf[PAGE_SIZE * 2]{};
                uint8_t* internal_buf_pos =
                    reinterpret_cast<uint8_t*>(ALIGN_UP(static_cast<uint8_t*>(internal_buf), PAGE_SIZE));

                const off_t tail_offset = file_offset + block_write_size;
                // Read the remaining block (size of PAGE_SIZE)
                ssize_t read_cnt;
                read_cnt = ::pread(handle_->fd, internal_buf_pos, PAGE_SIZE, tail_offset);
                if (read_cnt < 0)
                {
                    fmt::print(stderr, "Cannot read the remaining file content block! ({})\n", std::strerror(errno));
                    return finish_write(true);
                }
                // Overwrite a buffer to write, to the intermediate remaining block (internal_buf_pos)
                if (memory_type == cudaMemoryTypeUnregistered)
                {
                    memcpy(internal_buf_pos, reinterpret_cast<const uint8_t*>(buf) + buf_offset + block_write_size,
                           remaining);
                }
                else
                {
                    CUDA_TRY(cudaMemcpy(internal_buf_pos,
                                        reinterpret_cast<const uint8_t*>(buf) + buf_offset + block_write_size,
                                        remaining, cudaMemcpyDeviceToHost));
                    if (cuda_status)
                    {
                        return finish_write(true);
                    }
                }
                // Write the constructed block
                write_cnt = ::pwrite(handle_->fd, internal_buf_pos, PAGE_SIZE, tail_offset);
                if (write_cnt < 0)
                {
                    fmt::print(stderr, "Cannot write the remaining file content! ({})\n", std::strerror(errno));
                    return finish_write(true);
                }

                total_write_cnt += covered(write_cnt, 0, remaining);
            }
        }
        else
        {
            uint64_t cache_size = s_cufile_initializer.max_host_cache_size();
            uint8_t* cache_buf = static_cast<uint8_t*>(s_cufile_cache.host_cache());

            off_t file_start_offset = ALIGN_DOWN(file_offset, PAGE_SIZE);
            off_t end_offset = count + file_offset;
            off_t end_boundary_offset = ALIGN_UP(end_offset, PAGE_SIZE);
            size_t large_block_size = end_boundary_offset - file_start_offset;
            off_t page_offset = file_offset - file_start_offset;
            const uint8_t* input_buf = static_cast<const uint8_t*>(buf) + buf_offset;

            if (large_block_size <= cache_size) // Optimize if bytes to write is less than cache_size
            {
                memset(cache_buf, 0, PAGE_SIZE);
                ssize_t read_cnt = ::pread(handle_->fd, cache_buf, PAGE_SIZE, file_start_offset);
                if (read_cnt < 0)
                {
                    fmt::print(
                        stderr, "Cannot read the head part of the file content block! ({})\n", std::strerror(errno));
                    return -1;
                }
                if (large_block_size > PAGE_SIZE)
                {
                    read_cnt = ::pread(handle_->fd, cache_buf + large_block_size - PAGE_SIZE, PAGE_SIZE,
                                       end_boundary_offset - PAGE_SIZE);
                    if (read_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot read the tail part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }
                }

                if (memory_type == cudaMemoryTypeUnregistered)
                {
                    memcpy(cache_buf + page_offset, input_buf, count);
                }
                else
                {
                    CUDA_TRY(cudaMemcpy(cache_buf + page_offset, input_buf, count, cudaMemcpyDeviceToHost));
                    if (cuda_status)
                    {
                        return -1;
                    }
                }

                // Write the constructed block
                ssize_t write_cnt = ::pwrite(handle_->fd, cache_buf, large_block_size, file_start_offset);
                if (write_cnt < 0)
                {
                    fmt::print(stderr, "Cannot write the file content block! ({})\n", std::strerror(errno));
                    return -1;
                }

                total_write_cnt += covered(write_cnt, page_offset, count);
            }
            else
            {
                off_t overflow_offset = page_offset + count;
                size_t header_size = (overflow_offset > PAGE_SIZE) ? PAGE_SIZE - page_offset : count;
                size_t tail_size = (overflow_offset > PAGE_SIZE) ? end_offset - ALIGN_DOWN(end_offset, PAGE_SIZE) : 0;
                uint64_t body_remaining_size = count - header_size - tail_size;
                off_t write_offset = file_start_offset;

                size_t bytes_to_copy;
                ssize_t read_cnt;
                ssize_t write_cnt;

                uint8_t internal_buf[PAGE_SIZE * 2]{};
                uint8_t* internal_buf_pos =
                    reinterpret_cast<uint8_t*>(ALIGN_UP(static_cast<uint8_t*>(internal_buf), PAGE_SIZE));
                // Handle the head part of the file content
                if (header_size)
                {
                    read_cnt = ::pread(handle_->fd, internal_buf_pos, PAGE_SIZE, write_offset);
                    if (read_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot read the head part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }
                    // Overwrite the region to write
                    bytes_to_copy = header_size;
                    if (memory_type == cudaMemoryTypeUnregistered)
                    {
                        memcpy(internal_buf_pos + page_offset, input_buf, bytes_to_copy);
                    }
                    else
                    {
                        CUDA_TRY(cudaMemcpy(
                            internal_buf_pos + page_offset, input_buf, bytes_to_copy, cudaMemcpyDeviceToHost));
                        if (cuda_status)
                        {
                            return -1;
                        }
                    }

                    // Write the constructed block
                    write_cnt = ::pwrite(handle_->fd, internal_buf_pos, PAGE_SIZE, write_offset);
                    if (write_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot write the head part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return -1;
                    }
                    const size_t header_written = covered(write_cnt, page_offset, header_size);
                    input_buf += header_written;
                    write_offset += write_cnt;

                    total_write_cnt += header_written;
                    short_write = header_written < header_size;
                }

                // Copy n * PAGE_SIZE bytes
                if (body_remaining_size > 0 && !short_write)
                {
                    write_cnt = write_staged(input_buf, body_remaining_size, write_offset);
                    if (write_cnt < 0)
                    {
                        return finish_write(true);
                    }
                    write_offset += write_cnt;
                    input_buf += write_cnt;

                    total_write_cnt += write_cnt;
                    short_write = static_cast<size_t>(write_cnt) < body_remaining_size;
                }

                // Handle the tail part of the file content
                if (tail_size && !short_write)
                {
                    memset(internal_buf_pos, 0, PAGE_SIZE);
                    read_cnt = ::pread(handle_->fd, internal_buf_pos, PAGE_SIZE, write_offset);
                    if (read_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot read the tail part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return finish_write(true);
                    }
                    // Overwrite the region to write
                    bytes_to_copy = tail_size;
                    if (memory_type == cudaMemoryTypeUnregistered)
                    {
                        memcpy(internal_buf_pos, input_buf, bytes_to_copy);
                    }
                    else
                    {
                        CUDA_TRY(cudaMemcpy(internal_buf_pos, input_buf, bytes_to_copy, cudaMemcpyDeviceToHost));
                        if (cuda_status)
                        {
                            return finish_write(true);
                        }
                    }

                    // Write the constructed block
                    write_cnt = ::pwrite(handle_->fd, internal_buf_pos, PAGE_SIZE, write_offset);
                    if (write_cnt < 0)
                    {
                        fmt::print(stderr, "Cannot write the tail part of the file content block! ({})\n",
                                   std::strerror(errno));
                        return finish_write(true);
                    }
                    total_write_cnt += covered(write_cnt, 0, tail_size);
                }
            }
        }
    }
    else if (file_type == FileHandleType::kGPUDirect)
    {
        (void*)s_cufile_cache.device_cache(); // Lazy initialization

        ssize_t write_cnt =
            cuFileWrite(handle_->cufile, reinterpret_cast<const char*>(buf) + buf_offset, count, file_offset, 0);
        if (write_cnt < 0)
        {
            fmt::print(stderr, "[cuFile Error] {}\n", CUFILE_ERRSTR(write_cnt));
            return -1;
        }
        total_write_cnt += write_cnt;
    }
    return finish_write();
}
bool CuFileDriver::close()
{
    if (handle_ && handle_->cufile)
    {
        cuFileHandleDeregister(handle_->cufile);
    }
    if (mmap_ptr_)
    {
        int err = munmap(mmap_ptr_, file_size_);
        if (err < 0)
        {
            fmt::print(stderr, "[Error] Cannot call munmap() ({})\n", std::strerror(errno));
        }
        mmap_ptr_ = nullptr;
    }
    if (handle_ && handle_->fd != -1)
    {
        // If block write was used
        if ((file_flags_ & O_RDWR) &&
            (handle_->type == FileHandleType::kGPUDirect || handle_->type == FileHandleType::kPosixODirect))
        {
            // Truncate file assuming that `file_size_` is up to date during pwrite() calls
            int err = ::ftruncate(handle_->fd, file_size_);
            if (err < 0)
            {
                fmt::print(stderr, "[Error] Cannot resize the file {} to {} ({})\n", handle_->path, file_size_,
                           std::strerror(errno));
            }
        }
        handle_ = nullptr;
    }
    file_path_.clear();
    file_size_ = 0;
    file_flags_ = -1;
    return true;
}

filesystem::Path CuFileDriver::path() const
{
    return file_path_;
}

CuFileDriver::~CuFileDriver()
{
    close();
}

} // namespace cucim::filesystem
