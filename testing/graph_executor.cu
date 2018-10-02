#include <iostream>
#include <cassert>
#include <graph_executor.hpp>
#include <void_sender.hpp>

int main()
{
  cudaStream_t stream{};
  if(auto error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("CUDA error after cudaStreamCreate(): " + std::string(cudaGetErrorString(error)));
  }

  {
    // test A then B

    void_sender start;

    graph_executor ex(stream);

    auto task_a = ex.then_execute(
      [] __host__ __device__ ()
      {
        printf("Task A\n");
      },
      start
    );

    auto task_b = ex.then_execute(
      [] __host__ __device__ ()
      {
        printf("Task B\n");
      },
      task_a
    );

    task_b.submit();

    task_b.sync_wait();
  }

  if(auto error = cudaStreamDestroy(stream))
  {
    throw std::runtime_error("CUDA error after cudaStreamDestroy(): " + std::string(cudaGetErrorString(error)));
  }

  std::cout << "OK" << std::endl;
  
  return 0;
}

