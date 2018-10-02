#include <iostream>
#include <cassert>
#include <bulk_graph_executor.hpp>
#include <void_sender.hpp>

__managed__ unsigned int result;

int main()
{
  cudaStream_t stream{};
  if(auto error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("CUDA error after cudaStreamCreate(): " + std::string(cudaGetErrorString(error)));
  }

  {
    // test with
    // * empty outer shared object
    // * empty inner shared object

    bulk_graph_executor ex(stream);

    void_sender start;

    grid_index shape_a{dim3(10), dim3(10)};

    result = 0;

    auto task_a = ex.bulk_then_execute(
      [] __device__ (grid_index idx)
      {
        dim3 block_idx = idx[0];
        dim3 thread_idx = idx[1];

        unsigned int my_contribution = block_idx.x ^ thread_idx.x;

        atomicXor(&result, my_contribution);
      },
      shape_a,
      start
    );

    grid_index shape_b{dim3(10), dim3(10)};

    auto task_b = ex.bulk_then_execute(
      [] __device__ (grid_index idx)
      {
        dim3 block_idx = idx[0];
        dim3 thread_idx = idx[1];

        unsigned int my_contribution = block_idx.x ^ thread_idx.x;

        atomicXor(&result, my_contribution);
      },
      shape_b,
      task_a
    );

    task_b.submit();

    task_b.sync_wait();

    // compute the expected result
    unsigned int expected_result = 0;
    for(unsigned int i = 0; i < shape_a[0].x; ++i)
    {
      for(unsigned int j = 0; j < shape_a[1].x; ++j)
      {
        expected_result ^= (i ^ j);
      }
    }

    for(unsigned int i = 0; i < shape_b[0].x; ++i)
    {
      for(unsigned int j = 0; j < shape_b[1].x; ++j)
      {
        expected_result ^= (i ^ j);
      }
    }

    assert(expected_result == result);
  }

  if(auto error = cudaStreamDestroy(stream))
  {
    throw std::runtime_error("CUDA error after cudaStreamDestroy(): " + std::string(cudaGetErrorString(error)));
  }

  std::cout << "OK" << std::endl;
  
  return 0;
}

