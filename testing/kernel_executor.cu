#include <kernel_executor.hpp>
#include <iostream>
#include <cassert>

__managed__ unsigned int result;

struct empty {};

int main()
{
  cuda_context ctx;

  cudaStream_t stream{};
  if(auto error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("CUDA error after cudaStreamCreate(): " + std::string(cudaGetErrorString(error)));
  }

  {
    // test with
    // * non-empty outer shared object
    // * non-empty inner shared object

    grid_index shape{dim3(10), dim3(10)};

    result = 0;

    kernel_executor ex(ctx, stream);

    ex.bulk_execute(
      [] __device__ (grid_index idx, int& outer_shared, int& inner_shared)
      {
        dim3 block_idx = idx[0];
        dim3 thread_idx = idx[1];

        unsigned int my_contribution = block_idx.x ^ thread_idx.x ^ outer_shared ^ inner_shared;

        atomicXor(&result, my_contribution);
      },
      shape,
      [] __host__ __device__ { return 7; },
      [] __host__ __device__ { return 13; }
    );

    // wait for execution to complete
    ex.wait();

    // compute the expected result
    unsigned int expected_result = 0;
    for(unsigned int i = 0; i < 10; ++i)
    {
      for(unsigned int j = 0; j < 10; ++j)
      {
        expected_result ^= (i ^ j ^ 7 ^ 13);
      }
    }

    assert(expected_result == result);
  }

  {
    // test with
    // * empty outer shared object
    // * non-empty inner shared object

    grid_index shape{dim3(10), dim3(10)};

    result = 0;

    kernel_executor ex(ctx, stream);

    ex.bulk_execute(
      [] __device__ (grid_index idx, empty outer_shared, int& inner_shared)
      {
        dim3 block_idx = idx[0];
        dim3 thread_idx = idx[1];

        unsigned int my_contribution = block_idx.x ^ thread_idx.x ^ inner_shared;

        atomicXor(&result, my_contribution);
      },
      shape,
      [] __host__ __device__ { return empty{}; },
      [] __host__ __device__ { return 13; }
    );

    // wait for execution to complete
    ex.wait();

    // compute the expected result
    unsigned int expected_result = 0;
    for(unsigned int i = 0; i < 10; ++i)
    {
      for(unsigned int j = 0; j < 10; ++j)
      {
        expected_result ^= (i ^ j ^ 13);
      }
    }

    assert(expected_result == result);
  }

  {
    // test with
    // * non-empty outer shared object
    // * empty inner shared object

    grid_index shape{dim3(10), dim3(10)};

    result = 0;

    kernel_executor ex(ctx, stream);

    ex.bulk_execute(
      [] __device__ (grid_index idx, int& outer_shared, empty&)
      {
        dim3 block_idx = idx[0];
        dim3 thread_idx = idx[1];

        unsigned int my_contribution = block_idx.x ^ thread_idx.x ^ outer_shared;

        atomicXor(&result, my_contribution);
      },
      shape,
      [] __host__ __device__ { return 7; },
      [] __host__ __device__ { return empty{}; }
    );

    // wait for execution to complete
    ex.wait();

    // compute the expected result
    unsigned int expected_result = 0;
    for(unsigned int i = 0; i < 10; ++i)
    {
      for(unsigned int j = 0; j < 10; ++j)
      {
        expected_result ^= (i ^ j ^ 7);
      }
    }

    assert(expected_result == result);
  }

  {
    // test with
    // * empty outer shared object
    // * empty inner shared object

    grid_index shape{dim3(10), dim3(10)};

    result = 0;

    kernel_executor ex(ctx, stream);

    ex.bulk_execute(
      [] __device__ (grid_index idx, empty, empty)
      {
        dim3 block_idx = idx[0];
        dim3 thread_idx = idx[1];

        unsigned int my_contribution = block_idx.x ^ thread_idx.x;

        atomicXor(&result, my_contribution);
      },
      shape,
      [] __host__ __device__ { return empty{}; },
      [] __host__ __device__ { return empty{}; }
    );

    // wait for execution to complete
    ex.wait();

    // compute the expected result
    unsigned int expected_result = 0;
    for(unsigned int i = 0; i < 10; ++i)
    {
      for(unsigned int j = 0; j < 10; ++j)
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

