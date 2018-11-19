#pragma once

#include <stdexcept>
#include <string>

template<class CudaGraphExecutor>
class basic_sender
{
  public:
    const CudaGraphExecutor& executor() const
    {
      return executor_;
    }

    void submit()
    {
      // create a new graph
      cudaGraph_t graph = make_graph();

      // launch the graph
      launch(graph);

      // destroy the graph
      if(auto error = cudaGraphDestroy(graph))
      {
        throw std::runtime_error("sender::submit: CUDA error after cudaGraphDestroy: " + std::string(cudaGetErrorString(error)));
      }
    }

    void sync_wait() const
    {
      if(!event_)
      {
        throw std::runtime_error("sender::sync_wait: invalid state");
      }

      if(auto error = cudaEventSynchronize(event_))
      {
        throw std::runtime_error("sender::sync_wait: CUDA error after cudaEventSynchronize: " + std::string(cudaGetErrorString(error)));
      }
    }

  private:
    sender(const CudaGraphExecutor& executor)
      : executor_(executor),
        event_{}
    {}

    virtual cudaGraphNode_t insert(cudaGraph_t g) const = 0;

    cudaGraph_t make_graph() const
    {
      // create a new graph
      cudaGraph_t graph{};
      if(auto error = cudaGraphCreate(&graph, 0))
      {
        throw std::runtime_error("sender::make_graph: CUDA error after cudaGraphCreate: " + std::string(cudaGetErrorString(error)));
      }

      // insert into the graph
      insert(graph);

      // return the graph
      return graph;
    }

    void launch(cudaGraph_t graph)
    {
      // instantiate the graph
      cudaGraphExec_t executable_graph{};
      if(auto error = cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0))
      {
        throw std::runtime_error("sender::launch: CUDA error after cudaGraphInstantiate: " + std::string(cudaGetErrorString(error)));
      }
      
      // launch the graph
      if(auto error = cudaGraphLaunch(executable_graph, executor().stream()))
      {
        throw std::runtime_error("sender::launch: CUDA error after cudaGraphLaunch: " + std::string(cudaGetErrorString(error)));
      }

      // create an event
      if(auto error = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming))
      {
        throw std::runtime_error("sender::launch: CUDA error after cudaEventCreateWithFlags: " + std::string(cudaGetErrorString(error)));
      }

      // record an event
      if(auto error = cudaEventRecord(event_, executor().stream()))
      {
        throw std::runtime_error("sender::launch: CUDA error after cudaEventRecord: " + std::string(cudaGetErrorString(error)));
      }
      
      // delete the graph instance
      if(auto error = cudaGraphExecDestroy(executable_graph))
      {
        throw std::runtime_error("sender::launch: CUDA error after cudaGraphExecDestroy: " + std::string(cudaGetErrorString(error)));
      }
    }

    CudaGraphExecutor executor_;
    cudaEvent_t event_;
};

