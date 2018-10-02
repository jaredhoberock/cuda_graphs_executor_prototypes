#pragma once

#include <grid_index.hpp>

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


inline void launch(cudaStream_t stream, cudaGraph_t graph)
{
   // instantiate the graph
   cudaGraphExec_t executable_graph{};
   if(auto error = cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0))
   {
     throw std::runtime_error("detail::submit: CUDA error after cudaGraphInstantiate: " + std::string(cudaGetErrorString(error)));
   }

   // launch the graph
   if(auto error = cudaGraphLaunch(executable_graph, stream))
   {
     throw std::runtime_error("detail::submit: CUDA error after cudaGraphLaunch: " + std::string(cudaGetErrorString(error)));
   }

   // delete the graph instance
   if(auto error = cudaGraphExecDestroy(executable_graph))
   {
     throw std::runtime_error("detail::submit: CUDA error after cudaGraphExecDestroy: " + std::string(cudaGetErrorString(error)));
   }
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
class bulk_sender
{
  public:
    const bulk_graph_executor& executor() const
    {
      return executor_;
    }

    void submit()
    {
      // create a new graph
      cudaGraph_t graph = make_graph();

      // launch the graph
      detail::launch(executor().stream(), graph);

      // destroy the graph
      if(auto error = cudaGraphDestroy(graph))
      {
        throw std::runtime_error("bulk_sender::submit: CUDA error after cudaGraphDestroy: " + std::string(cudaGetErrorString(error)));
      }
    }

    void sync_wait() const
    {
      // XXX should keep a cudaEvent_t member to avoid synchronizing the whole stream
      if(auto error = cudaStreamSynchronize(executor().stream()))
      {
        throw std::runtime_error("bulk_sender::sync_wait: CUDA error after cudaStreamSynchronize: " + std::string(cudaGetErrorString(error)));
      }
    }

  private:
    friend class bulk_graph_executor;
    template<class,class> friend class bulk_sender;

    bulk_sender(const bulk_graph_executor& executor, Function f, grid_index shape, Sender&& predecessor)
      : executor_(executor),
        function_(f),
        predecessor_(std::move(predecessor)),
        shape_(shape)
    {}

    // this function transliterates the chain of predecessors into a graph
    cudaGraphNode_t insert(cudaGraph_t g) const
    {
      // insert the predecessor
      cudaGraphNode_t predecessor_node = predecessor_.insert(g);

      // introduce a new kernel node
      cudaGraphNode_t result_node{};
      void* kernel_params[] = {reinterpret_cast<void*>(const_cast<Function*>(&function_))};
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

    cudaGraph_t make_graph() const
    {
      // create a new graph
      cudaGraph_t graph{};
      if(auto error = cudaGraphCreate(&graph, 0))
      {
        throw std::runtime_error("CUDA error after cudaGraphCreate: " + std::string(cudaGetErrorString(error)));
      }

      // insert into the graph
      insert(graph);

      // return the graph
      return graph;
    }

    bulk_graph_executor executor_;
    Function function_;
    grid_index shape_;
    Sender predecessor_;
};


template<class Function, class Sender>
bulk_sender<Function,Sender> bulk_graph_executor::bulk_then_execute(Function f, grid_index shape, Sender& sender) const
{
  return {*this, f, shape, std::move(sender)};
}

