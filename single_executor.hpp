#pragma once

#include <stdexcept>
#include <string>


namespace detail
{
namespace single_executor_detail
{


template<class Function>
__global__ void kernel(Function f)
{
  f();
}


} // end single_executor_detail
} // end detail


class single_executor
{
  public:
    inline single_executor(cudaStream_t s)
      : stream_(s)
    {}

    template<class Function>
    void execute(Function f) const
    {
      const void* kernel_ptr = &detail::single_executor_detail::kernel<Function>;
      if(auto error = cudaLaunchKernel(kernel_ptr, dim3{1}, dim3{1}, {&f}, 0, stream_))
      {
        throw std::runtime_error("single_executor::execute: CUDA error after cudaLaunchKernel: " + std::string(cudaGetErrorString(error)));
      }
    }

    inline void wait() const
    {
      // synchronize with everything submitted to the stream 
      if(auto error = cudaStreamSynchronize(stream_))
      {
        throw std::runtime_error("single_executor::wait: CUDA error after cudaStreamSynchronize: " + std::string(cudaGetErrorString(error)));
      }
    }

    constexpr bool operator==(const single_executor& other) const
    {
      return stream_ == other.stream_;
    }

    constexpr bool operator!=(const single_executor& other) const
    {
      return !(*this == other);
    }

  private:
    cudaStream_t stream_;
};

