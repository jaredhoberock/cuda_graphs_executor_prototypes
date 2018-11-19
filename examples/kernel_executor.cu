// $ nvcc -std=c++11 --expt-extended-lambda -I.. kernel_executor.cu
#include <iostream>
#include <kernel_executor.hpp>

__global__ void my_kernel()
{
  printf("Hello, world from my_kernel!\n");
}

struct empty {};

int main()
{
  // create a cuda_context to own memory allocations
  cuda_context ctx;

  // create a CUDA stream to launch kernels on
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // create a kernel_executor from the context and stream
  kernel_executor ex(ctx, stream);

  // grid_index encapsulates a kernel launch configuration into a single object
  // this shape describes a single thread block containing a single thread
  grid_index shape{dim3(1), dim3(1)};

  // kernel_executor can launch kernels directly through a non-standard interface
  // this interface may be useful for porting legacy code
  ex.bulk_execute_global_function(my_kernel, shape);


  // kernel_executor::bulk_execute is the standard interface

  // the standard interface receives callable objects such as lambda functions
  auto kernel = [] __device__ (grid_index idx, int& grid_shared, int& block_shared)
  {
    printf("Hello, world from lambda function!\n");

    // idx identifies this lambda function's invocation in the grid of threads
    // it is equivalent to blockIdx & threadIdx
    int block_idx = idx[0].x;
    int thread_idx = idx[1].x;

    // shared objects may be used for communication during a bulk_execute

    // grid_shared is a single object which is shared among all the threads in the grid
    // there is one such grid_shared object per bulk_execute
    
    // block_shared is a single object which is shared among all threads within a thread block
    // there is one such block_shared object for each thread block created by a bulk_execute
  };

  ex.bulk_execute(
    kernel,                                // the callable describing the kernel to execute
    shape,                                 // the shape of the kernel to execute
    [] __host__ __device__ { return 42; }, // a factory function to create the initial state of the grid shared object
                                           // this function is invoked a single time per bulk_execute 
    [] __host__ __device__ { return 13; }  // a factory function to create the initial state of each block shared object
                                           // this function is invoked a single time for each thread block created by the bulk_execute
  );

  // if a kernel does not need a shared object, it is most efficient for its
  // factory function to return an empty type which the kernel function may ignore:

  ex.bulk_execute(
    [] __device__ (grid_index idx, empty, empty) { /* no-op */ },
    shape,
    [] __host__ __device__ { return empty(); },
    [] __host__ __device__ { return empty(); }
  );

  // block the main thread until all work created on the stream have completed
  ex.wait();

  // destroy the CUDA stream
  cudaStreamDestroy(stream);

  std::cout << "OK" << std::endl;
}

