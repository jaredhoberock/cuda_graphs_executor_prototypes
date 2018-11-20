# Introduction

This repository contains executors which target both traditional CUDA kernel launch and CUDA graphs.

Executors hide platform-specific details of work creation and present C++ programmers with a uniform interface for creating work.

# `kernel_executor`

`kernel_executor` eagerly launches CUDA kernels:

```c++
#include <iostream>
#include <kernel_executor.hpp>

__global__ void my_kernel()
{
  printf("Hello, world from my_kernel!\n");
}

struct empty {};

int main()
{
  // Create a cuda_context to garbage collect asynchronous memory allocations.
  cuda_context ctx;

  // Create a CUDA stream on which to launch kernels.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Create a kernel_executor from the context and stream.
  kernel_executor ex(ctx, stream);

  // grid_index encapsulates a kernel launch configuration into a single object.
  // This shape describes a single thread block containing a single thread.
  grid_index shape{dim3(1), dim3(1)};

  // kernel_executor can launch kernels directly through a non-standard interface.
  // This interface may be useful for porting legacy code which launches CUDA kernels
  // directly through the triple chevron syntax.
  ex.bulk_execute_global_function(my_kernel, shape);


  // kernel_executor::bulk_execute is the standard interface

  // The standard interface receives callable objects such as lambda functions.
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
    // the grid_shared object is located in global memory
    
    // block_shared is a single object which is shared among all threads within a thread block
    // there is one such block_shared object for each thread block created by a bulk_execute
    // the block_shared object is located in __shared__ memory
  };

  ex.bulk_execute(
    kernel,                                // The callable describing the kernel to execute.
    shape,                                 // The shape of the kernel to execute.
    [] __host__ __device__ { return 42; }, // A factory function to create the initial state of the grid shared object.
                                           // This function is invoked a single time per call to bulk_execute.
    [] __host__ __device__ { return 13; }  // A factory function to create the initial state of each block shared object.
                                           // This function is invoked a single time for each thread block created by bulk_execute.
  );

  // If a kernel does not need a shared object, it is most efficient for its
  // factory function to return an empty type which the kernel function may ignore.

  // For example:
  ex.bulk_execute(
    [] __device__ (grid_index idx, empty, empty) { /* no-op */ },
    shape,
    [] __host__ __device__ { return empty(); },
    [] __host__ __device__ { return empty(); }
  );

  // Block the main thread until all work created on the stream have completed.
  ex.wait();

  // Destroy the CUDA stream.
  cudaStreamDestroy(stream);

  std::cout << "OK" << std::endl;
}
```

# `graph_executor`

'graph_executor` constructs a graph of dependent kernels, which may be launched lazily:

```c++
#include <iostream>
#include <array>
#include <graph_executor.hpp>
#include <void_sender.hpp>
#include <when_all.hpp>

int main()
{
  // Create a CUDA stream to launch kernels on.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Create a graph_executor from the stream.
  graph_executor ex(stream);

  // graph_executor works by constructing a graph of dependent kernels.
  // Nodes in the graph are represented as "Senders", which are an [experimental
  // C++ proposal](https://wg21.link/P1194) under consideration for future standardization.

  // This example program will create a simple diamond-shaped graph structure:
  // 
  //       root_node
  //       /      \
  //     node_a  node_b
  //       \      /
  //        node_c
  //
  // In this graph, the direction of time flows from the top to the bottom.
  // The execution of work at node_a and node_b depend on the completion of root_node's work,
  // while node_c depends on the completion of work at node_b and node_c.

  // To begin describing the graph, we define a root node with a "void_sender".
  // A void_sender depends on nothing, and does no actual work. It is a no-op.
  void_sender root_node;

  // Nodes A, B, and C will contain actual kernel launches. To keep things simple,
  // we'll launch a single thread at each of these nodes
  grid_index shape{dim3(1), dim3(1)};

  // Nodes representing kernel launches are created using graph_executor::bulk_then_execute,
  // which defines a CUDA kernel launch dependent on another node.

  // Node A is represented with a "kernel_sender". This is a sender type which corresponds to a kernel launch.
  kernel_sender node_a = ex.bulk_then_execute(
    // The function object defining the kernel is bulk_then_execute's first parameter
    // We've used an inline __device__ lambda function here, but we could have also used a C++ functor.
    [] __device__ (grid_index idx)
    {
      dim3 block_idx = idx[0];
      dim3 thread_idx = idx[1];

      printf("Hello, world from thread (%d, %d) of Node A!\n", block_idx.x, thread_idx.x);
    },

    // The shape of the kernel launch is the second parameter.
    shape,

    // The sender on which this kernel launch depends is the third parameter.
    root_node
  );

  // Node B is defined similarly.
  kernel_sender node_b = ex.bulk_then_execute(
    [] __device__ (grid_index idx)
    {
      dim3 block_idx = idx[0];
      dim3 thread_idx = idx[1];

      printf("Hello, world from thread (%d, %d) of Node B!\n", block_idx.x, thread_idx.x);
    },
    shape,
    root_node
  );

  // Node C depends on both Nodes A & B. However, kernels launched through bulk_then_execute
  // only receive a single sender object as a dependency.
  // To represent multiple dependencies, we can create a "join_sender" with when_all.

  // when_all receives an executor and a collection of senders as parameters.

  // We can use std::array as our collection, but any type with .begin() and .end() iterators will do.
  // Since kernel_sender is a move-only type, we need to move them into std::array's constructor:
  std::array<kernel_sender,2> nodes_a_and_b{std::move(node_a), std::move(node_b)};

  // Create a sender corresponding to both Nodes A & B's completion with when_all.
  join_sender when_nodes_a_and_b = when_all(ex, nodes_a_and_b);

  // Now, we can define Node C similarly to Nodes A & B:
  kernel_sender node_c = ex.bulk_then_execute(
    [] __device__ (grid_index idx)
    {
      dim3 block_idx = idx[0];
      dim3 thread_idx = idx[1];

      printf("Hello, world from thread (%d, %d) of Node C!\n", block_idx.x, thread_idx.x);
    },
    shape,
    when_nodes_a_and_b
  );

  // At this point, we have described our entire computation as a graph.
  // However, nothing will happen until we submit the graph for execution.

  // Submit the graph for execution by calling .submit() on the terminal node in the graph.
  node_c.submit();

  // While the graph executes, we can do other work asynchronously on the host.
  std::cout << "Graph submitted for execution." << std::endl;

  // To synchronize with the completion of the graph, call .sync_wait() on the graph's terminal node.
  node_c.sync_wait();

  // At this point, the hello world messages should have been printed to the terminal. Because they are
  // independent of one another in the graph, Nodes A & B's messages may be output in any order.
  // However, because Node C depended on both Nodes A & B, Node C's message should follow both
  // Nodes A & B's messages.

  // While the process of defining this simple graph may seem complex, we anticipate that conveniences
  // will simplify it in the future.

  // Finally, destroy the CUDA stream.
  cudaStreamDestroy(stream);

  std::cout << "OK" << std::endl;

}
```

