#include <iostream>
#include <cassert>
#include <graph_executor.hpp>

__managed__ unsigned int result;

int main()
{
  {
    // test with
    // * empty outer shared object
    // * empty inner shared object

    bulk_graph_executor ex;

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

    // XXX should clean up all of this and give the executor a stream
    cudaStream_t stream{};
    if(auto error = cudaStreamCreate(&stream))
    {
      throw std::runtime_error("CUDA error after cudaStreamCreate: " + std::string(cudaGetErrorString(error)));
    }

    task_b.submit(stream);

    // XXX should implement .sync_wait() or whatever instead of explicit stream synchronization
    if(auto error = cudaStreamSynchronize(stream))
    {
      throw std::runtime_error("CUDA error after cudaStreamSynchronize: " + std::string(cudaGetErrorString(error)));
    }

    if(auto error = cudaStreamDestroy(stream))
    {
      throw std::runtime_error("CUDA error after cudaStreamDestroy: " + std::string(cudaGetErrorString(error)));
    }

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

  std::cout << "OK" << std::endl;
  
  return 0;
}

