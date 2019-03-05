// $ nvcc -c -std=c++11 --expt-extended-lambda -I../.. test.cu -o test.o

#include <cassert>
#include <thrust/device_vector.h>
#include "daxpy.hpp"

void test(size_t n)
{
  // create resources
  cudaStream_t stream;
  if(cudaError_t error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("test: CUDA error after cudaStreamCreate: " + std::string(cudaGetErrorString(error)));
  }

  thrust::device_vector<double> x(n, 1);
  thrust::device_vector<double> y(n, 2);
  double a = 2;

  graph_executor ex(stream);
  any_sender task = daxpy(ex, n, a, x.data().get(), y.data().get());

  task.submit();
  task.sync_wait();

  if(cudaError_t error = cudaStreamDestroy(stream))
  {
    throw std::runtime_error("test: CUDA error after cudaStreamDestroy: " + std::string(cudaGetErrorString(error)));
  }

  thrust::device_vector<double> reference(n, 4);
  assert(reference == y);
}

