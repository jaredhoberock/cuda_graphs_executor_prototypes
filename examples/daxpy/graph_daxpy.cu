// $ nvcc -std=c++11 -I../.. graph_daxpy.cu -o graph_daxpy
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

__global__ void hello_world(int n, double a)
{
  if(blockIdx.x == 0 and threadIdx.x == 0) 
  {
    printf("Hello world!\n");
    printf("n: %d\n", n);
    printf("a: %f\n", a);
  }
}

cudaGraph_t make_daxpy_graph(int n, double a, const double* x, double* y)
{
  cudaGraph_t graph{};
  if(cudaError_t error = cudaGraphCreate(&graph, 0))
  {
    throw std::runtime_error("make_daxpy_graph: CUDA error after cudaGraphCreate: " + std::string(cudaGetErrorString(error)));
  }

  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;

  // introduce a kernel node
  void* kernel_params[] = {&n, &a, &x, &y};

  cudaKernelNodeParams params
  {
    reinterpret_cast<void*>(daxpy_kernel),
    dim3(num_blocks),
    dim3(block_size),
    0,
    kernel_params,
    nullptr
  };

  cudaGraphNode_t node{};
  if(cudaError_t error = cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params))
  {
    throw std::runtime_error("make_daxpy_graph: CUDA error after cudaGraphAddKernelNode: " + std::string(cudaGetErrorString(error)));
  }

  return graph;
}

void test(size_t n)
{
  thrust::device_vector<double> x(n, 1);
  thrust::device_vector<double> y(n, 2);
  double a = 2;

  // make the graph
  cudaGraph_t graph = make_daxpy_graph(n, a, x.data().get(), y.data().get());

  // instantiate the graph
  cudaGraphExec_t executable_graph{};
  if(cudaError_t error = cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0))
  {
    throw std::runtime_error("test: CUDA error after cudaGraphInstantiate: " + std::string(cudaGetErrorString(error)));
  }

  // create a stream
  cudaStream_t stream{};
  if(cudaError_t error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("test: CUDA error after cudaStreamCreate: " + std::string(cudaGetErrorString(error)));
  }

  // launch the graph
  if(cudaError_t error = cudaGraphLaunch(executable_graph, stream))
  {
    throw std::runtime_error("test: CUDA error after cudaGraphLaunch: " + std::string(cudaGetErrorString(error)));
  }

  // wait
  if(cudaError_t error = cudaStreamSynchronize(stream))
  {
    throw std::runtime_error("test: CUDA error after cudaStreamSynchronize: " + std::string(cudaGetErrorString(error)));
  }

  // destroy resources
  if(cudaError_t error = cudaStreamDestroy(stream))
  {
    throw std::runtime_error("test: CUDA error after cudaStreamDestroy: " + std::string(cudaGetErrorString(error)));
  }
  
  if(cudaError_t error = cudaGraphDestroy(graph))
  {
    throw std::runtime_error("test: CUDA error after cudaGraphDestroy: " + std::string(cudaGetErrorString(error)));
  }

  if(cudaError_t error = cudaGraphExecDestroy(executable_graph))
  {
    throw std::runtime_error("test: CUDA error after cudaGraphExecDestroy: " + std::string(cudaGetErrorString(error)));
  }

  // check the result
  thrust::device_vector<double> reference(n, 4);
  assert(reference == y);
}

double measure_bandwidth(size_t n, size_t num_trials = 100)
{
  thrust::device_vector<double> x(n, 1);
  thrust::device_vector<double> y(n, 2);
  double a = 2;

  // make a stream
  cudaStream_t stream;
  if(cudaError_t error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("measure_bandwidth: CUDA error after cudaStreamCreate: " + std::string(cudaGetErrorString(error)));
  }

  // make the graph
  cudaGraph_t graph = make_daxpy_graph(n, a, x.data().get(), y.data().get());

  // instantiate the graph
  cudaGraphExec_t executable_graph;
  if(cudaError_t error = cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0))
  {
    throw std::runtime_error("test: CUDA error after cudaGraphInstantiate: " + std::string(cudaGetErrorString(error)));
  }

  // time trials
  auto start = std::chrono::high_resolution_clock().now();
  {
    for(size_t i = 0; i < num_trials; ++i)
    {
      cudaGraphLaunch(executable_graph, stream);
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

  if(cudaError_t error = cudaGraphExecDestroy(executable_graph))
  {
    throw std::runtime_error("measure_bandwidth: CUDA error after cudaGraphExecDestroy: " + std::string(cudaGetErrorString(error)));
  }

  if(cudaError_t error = cudaGraphDestroy(graph))
  {
    throw std::runtime_error("measure_bandwidth: CUDA error after cudaGraphDestroy: " + std::string(cudaGetErrorString(error)));
  }

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
  std::cout << "Graph DAXPY bandwidth: " << bandwidth << " GB/s" << std::endl;
  std::cout << "OK" << std::endl;

  return 0;
}


