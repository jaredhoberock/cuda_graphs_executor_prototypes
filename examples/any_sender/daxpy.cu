// $ nvcc -c -std=c++11 --expt-extended-lambda -I../.. daxpy.cu -o daxpy.o

#include "daxpy.hpp"
#include <void_sender.hpp>

any_sender daxpy(const graph_executor& ex, int n, double a, const double* x, double* y)
{
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;

  void_sender root_node;

  grid_index shape(num_blocks, block_size);

  return ex.bulk_then_execute(
    [=] __device__ (grid_index idx)
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
    root_node
  );
}

