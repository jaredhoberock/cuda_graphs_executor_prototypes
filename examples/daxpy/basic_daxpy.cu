// $ nvcc -std=c++11 -I../.. basic_daxpy.cu -o basic_daxpy
#include <cassert>
#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>

__global__ void daxpy_kernel(int n, double a, const double* x, double* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    y[i] = a * x[i] + y[i];
  }
}

void daxpy(cudaStream_t s, int n, double a, const double* x, double* y)
{
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  daxpy_kernel<<<num_blocks, block_size, 0, s>>>(n, a, x, y);
}

void test(size_t n)
{
  thrust::device_vector<double> x(n, 1);
  thrust::device_vector<double> y(n, 2);
  double a = 2;

  daxpy(0, n, a, x.data().get(), y.data().get());

  if(cudaError_t error = cudaDeviceSynchronize())
  {
    throw std::runtime_error("test: CUDA error after cudaDeviceSynchronize: " + std::string(cudaGetErrorString(error)));
  }

  thrust::device_vector<double> reference(n, 4);
  assert(reference == y);
}

double measure_bandwidth(size_t n, size_t num_trials = 100)
{
  thrust::device_vector<double> x(n, 1);
  thrust::device_vector<double> y(n, 2);
  double a = 2;

  cudaStream_t stream;
  if(cudaError_t error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("measure_bandwidth: CUDA error after cudaStreamCreate: " + std::string(cudaGetErrorString(error)));
  }

  // time trials
  auto start = std::chrono::high_resolution_clock().now();
  {
    for(size_t i = 0; i < num_trials; ++i)
    {
      daxpy(stream, n, a, x.data().get(), y.data().get());
    }

    if(cudaError_t error = cudaStreamSynchronize(stream))
    {
      throw std::runtime_error("measure_bandwidth: CUDA error after cudaStreamSynchronize: " + std::string(cudaGetErrorString(error)));
    }
  }
  auto end = std::chrono::high_resolution_clock().now();

  // compute mean GB/s
  size_t mean_nanoseconds = (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) / num_trials).count();
  double mean_seconds = double(mean_nanoseconds) / 1000000000;
  size_t num_bytes = 2 * n * sizeof(double);
  double mean_bytes_per_second = double(num_bytes) / mean_seconds;
  double mean_gigabytes_per_second = mean_bytes_per_second / 1000000000;

  if(cudaError_t error = cudaStreamDestroy(stream))
  {
    throw std::runtime_error("measure_bandwidth: CUDA error after cudaStreamDestroy: " + std::string(cudaGetErrorString(error)));
  }

  return mean_gigabytes_per_second;
}

int main(int argc, char** argv)
{
  size_t n = 1 << 25;
  if(argc > 1)
  {
    n = std::atoi(argv[1]);
  }

  // first test for correctness
  test(n);

  double bandwidth = measure_bandwidth(n);

  std::clog << n << ", " << bandwidth << std::endl;
  std::cout << "Basic DAXPY bandwidth: " << bandwidth << " GB/s" << std::endl;
  std::cout << "OK" << std::endl;

  return 0;
}

