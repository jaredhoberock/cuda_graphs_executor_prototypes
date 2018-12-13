// $ nvcc -std=c++11 --expt-extended-lambda -I../.. kernel_executor_daxpy.cu -o kernel_executor_daxpy

#include <thrust/device_vector.h>
#include <kernel_executor.hpp>

struct empty {};

void daxpy(const kernel_executor& ex, int n, double a, const double* x, double* y)
{
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;

  grid_index shape(num_blocks, block_size);

  ex.bulk_execute(
    [=] __device__ (grid_index idx, empty&, empty&)
    {
      int block_idx = idx[0].x;
      int thread_idx = idx[1].x;
      int i = block_idx * block_size + thread_idx;

      if(i < n)
      {
        y[i] = a * x[i] + y[i];
      }
    },
    shape,
    [] __host__ __device__ { return empty(); },
    [] __host__ __device__ { return empty(); }
  );
}

void test(size_t n)
{
  // create resources
  cuda_context ctx;
  cudaStream_t stream;
  if(cudaError_t error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("test: CUDA error after cudaStreamCreate: " + std::string(cudaGetErrorString(error)));
  }

  thrust::device_vector<double> x(n, 1);
  thrust::device_vector<double> y(n, 2);
  double a = 2;

  kernel_executor ex(ctx, stream);
  daxpy(ex, n, a, x.data().get(), y.data().get());
  ex.wait();

  if(cudaError_t error = cudaStreamDestroy(stream))
  {
    throw std::runtime_error("test: CUDA error after cudaStreamDestroy: " + std::string(cudaGetErrorString(error)));
  }

  thrust::device_vector<double> reference(n, 4);
  assert(reference == y);
}

double measure_bandwidth(size_t n, size_t num_trials = 100)
{
  thrust::device_vector<double> x(n, 1);
  thrust::device_vector<double> y(n, 2);
  double a = 2;

  // create resources
  cuda_context ctx;

  cudaStream_t stream;
  if(cudaError_t error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("measure_bandwidth: CUDA error after cudaStreamCreate: " + std::string(cudaGetErrorString(error)));
  }

  kernel_executor ex(ctx, stream);

  // time trials
  auto start = std::chrono::high_resolution_clock().now();
  {
    for(size_t i = 0; i < num_trials; ++i)
    {
      daxpy(ex, n, a, x.data().get(), y.data().get());
    }

    ex.wait();
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
  std::cout << "Kernel Executor DAXPY bandwidth: " << bandwidth << " GB/s" << std::endl;
  std::cout << "OK" << std::endl;

  return 0;
}

