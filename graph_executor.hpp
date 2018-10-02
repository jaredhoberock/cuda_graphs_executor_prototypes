#pragma once

#include <basic_sender.hpp>
#include <stdexcept>


namespace detail
{


template<class Function>
__global__ void single_kernel(Function f)
{
  f();
}


} // end detail


template<class,class> class single_sender;


class graph_executor
{
  public:
    graph_executor(cudaStream_t s)
      : stream_(s)
    {}

    template<class Function, class Sender>
    single_sender<Function,Sender> then_execute(Function f, Sender& sender) const;

    cudaStream_t stream() const
    {
      return stream_;
    }

  private:
    cudaStream_t stream_;
};


template<class Function, class Sender>
class single_sender : private basic_sender<single_sender<Function,Sender>, graph_executor, Function, Sender>
{
  private:
    using super_t = basic_sender<single_sender<Function,Sender>, graph_executor, Function, Sender>;

  public:
    using super_t::executor;
    using super_t::submit;
    using super_t::sync_wait;

  private:
    using super_t::predecessor;
    using super_t::function;

    // friend super_t so it can access insert() and downcasts
    friend super_t;

    // friend graph_executor so it can access the private ctor
    friend class graph_executor;

    // friend single_sender so it can access insert()
    template<class,class> friend class single_sender;

    using super_t::super_t;

    // this function transliterates the chain of predecessors into a graph
    cudaGraphNode_t insert(cudaGraph_t g) const
    {
      // insert the predecessor
      cudaGraphNode_t predecessor_node = predecessor().insert(g);

      // introduce a new kernel node
      cudaGraphNode_t result_node{};
      void* kernel_params[] = {reinterpret_cast<void*>(const_cast<Function*>(&function()))};
      cudaKernelNodeParams node_params
      {
        reinterpret_cast<void*>(&detail::single_kernel<Function>),
        dim3{1},
        dim3{1},
        0,
        kernel_params,
        nullptr
      };

      if(auto error = cudaGraphAddKernelNode(&result_node, g, &predecessor_node, 1, &node_params))
      {
        throw std::runtime_error("single_sender::insert: CUDA error after cudaGraphAddKernelNode: " + std::string(cudaGetErrorString(error)));
      }

      return result_node;
    }
};

template<class Function, class Sender>
single_sender<Function,Sender> graph_executor::then_execute(Function f, Sender& sender) const
{
  return {*this, f, std::move(sender)};
}

