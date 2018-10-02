#pragma once

#include <grid_index.hpp>
#include <basic_sender.hpp>

namespace detail
{


template<class Function>
__global__ void bulk_kernel(Function f)
{
  // create the index
  grid_index idx{blockIdx, threadIdx};

  // invoke f
  f(idx);
}


} // end detail


template<class,class> class bulk_sender;


class bulk_graph_executor
{
  public:
    bulk_graph_executor(cudaStream_t s)
      : stream_(s)
    {}

    template<class Function, class Sender>
    bulk_sender<Function,Sender> bulk_then_execute(Function f, grid_index shape, Sender& sender) const;

    cudaStream_t stream() const
    {
      return stream_;
    }

  private:
    cudaStream_t stream_;
};


template<class Function, class Sender>
class bulk_sender : private basic_sender<bulk_sender<Function,Sender>, bulk_graph_executor, Function, Sender>
{
  private:
    using super_t = basic_sender<bulk_sender<Function,Sender>, bulk_graph_executor, Function, Sender>;

  public:
    void sync_wait() const
    {
      // XXX should keep a cudaEvent_t member to avoid synchronizing the whole stream
      if(auto error = cudaStreamSynchronize(super_t::executor().stream()))
      {
        throw std::runtime_error("bulk_sender::sync_wait: CUDA error after cudaStreamSynchronize: " + std::string(cudaGetErrorString(error)));
      }
    }

    using super_t::executor;
    using super_t::submit;
    using super_t::sync_wait;

  private:
    using super_t::predecessor;
    using super_t::function;

    // friend super_t so it can access insert() and downcasts
    friend super_t;

    // friend bulk_graph_executor so it can access the private ctor
    friend class bulk_graph_executor;

    // friend bulk_sender so it can access insert()
    template<class,class> friend class bulk_sender;

    bulk_sender(const bulk_graph_executor& executor, Function f, grid_index shape, Sender&& predecessor)
      : super_t(executor, f, std::move(predecessor)),
        shape_(shape)
    {}

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
        reinterpret_cast<void*>(&detail::bulk_kernel<Function>),
        shape_[0],
        shape_[1],
        0,
        kernel_params,
        nullptr
      };

      if(auto error = cudaGraphAddKernelNode(&result_node, g, &predecessor_node, 1, &node_params))
      {
        throw std::runtime_error("bulk_sender::insert: CUDA error after cudaGraphAddKernelNode: " + std::string(cudaGetErrorString(error)));
      }

      return result_node;
    }

    grid_index shape_;
};


template<class Function, class Sender>
bulk_sender<Function,Sender> bulk_graph_executor::bulk_then_execute(Function f, grid_index shape, Sender& sender) const
{
  return {*this, f, shape, std::move(sender)};
}

