#pragma once

#include <cstddef>
#include <stdexcept>
#include <iostream>


template<class T>
class managed_allocator
{
  public:
    using value_type = T;

    managed_allocator(int device)
      : device_(device)
    {}

    managed_allocator()
      : managed_allocator(0)
    {}

    managed_allocator(const managed_allocator&) = default;

    template<class U>
    managed_allocator(const managed_allocator<U>& other)
      : device_(other.device())
    {}

    bool operator==(const managed_allocator& other) const
    {
      return device_ == other.device_;
    }

    bool operator!=(const managed_allocator& other) const
    {
      return !(*this == other);
    }

    value_type* allocate(std::size_t n) const
    {
      int old_device = set_device(device_);

      value_type* result = nullptr;
      if(auto error = cudaMallocManaged(&result, n * sizeof(value_type)))
      {
        throw std::runtime_error("managed_allocator::allocate: CUDA error after cudaMallocManaged: " + std::string(cudaGetErrorString(error)));
      }

      set_device(old_device);

      return result;
    }

    void deallocate(value_type* ptr, std::size_t) const
    {
      int old_device = set_device(device_);

      if(auto error = cudaFree(ptr))
      {
        std::cerr << "managed_allocator::deallocate: CUDA error after cudaFree: " << cudaGetErrorString(error) << std::endl;
        std::terminate();
      }

      set_device(old_device);
    }

    int device() const
    {
      return device_;
    }

  private:
    static int set_device(int new_device)
    {
      int old_device = -1;
      if(auto error = cudaGetDevice(&old_device))
      {
        throw std::runtime_error("managed_allocator::set_device: CUDA error after cudaGetDevice: " + std::string(cudaGetErrorString(error)));
      }

      if(auto error = cudaSetDevice(new_device))
      {
        throw std::runtime_error("managed_allocator::set_device: CUDA error after cudaSetDevice: " + std::string(cudaGetErrorString(error)));
      }

      return old_device;
    }

    int device_;
};

