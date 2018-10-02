#include <iostream>
#include <cassert>
#include <graph_executor.hpp>
#include <void_sender.hpp>

int main()
{
  {
    // test A then B

    void_sender start;

    graph_executor ex;

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

    cudaStream_t stream{};
    if(auto error = cudaStreamCreate(&stream))
    {
      throw std::runtime_error("CUDA error after cudaStreamCreate: " + std::string(cudaGetErrorString(error)));
    }

    task_b.submit(stream);

    if(auto error = cudaStreamSynchronize(stream))
    {
      throw std::runtime_error("CUDA error after cudaStreamSynchronize: " + std::string(cudaGetErrorString(error)));
    }

    if(auto error = cudaStreamDestroy(stream))
    {
      throw std::runtime_error("CUDA error after cudaStreamDestroy: " + std::string(cudaGetErrorString(error)));
    }
  }

  std::cout << "OK" << std::endl;
  
  return 0;
}

