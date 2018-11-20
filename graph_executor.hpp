#pragma once

#include <grid_index.hpp>
#include <basic_sender.hpp>
#include <kernel_sender.hpp>
#include <stdexcept>
#include <vector>


namespace detail
{


template<class Function>
__global__ void bulk_kernel(Function f)
{
  // create the index
  grid_index idx{blockIdx, threadIdx};

  // invoke f
  f(idx);
}


} // end detail


class graph_executor
{
  public:
    graph_executor(cudaStream_t s)
      : stream_(s)
    {}

    template<class Function, class Sender>
    kernel_sender bulk_then_execute(Function f, grid_index shape, Sender& predecessor) const
    {
      // we need to capture this array by value into the lambda below
      // so that it becomes a member of the lambda
      void* kernel_params[] = {nullptr};

      auto node_parameters_function = [=]() mutable
      {
        // XXX &f is an address of a member of this lambda because we captured f by value
        kernel_params[0] = (void*)&f;

        cudaKernelNodeParams result
        {
          reinterpret_cast<void*>(&detail::bulk_kernel<Function>),
          shape[0], // gridDim
          shape[1], // blockDim
          0,

          // XXX note that we're returning addresses which point into this lambda object here
          kernel_params,
          nullptr
        };

        return result;
      };

      return {stream(), node_parameters_function, std::move(predecessor)};
    }

    cudaStream_t stream() const
    {
      return stream_;
    }

  private:
    cudaStream_t stream_;
};

