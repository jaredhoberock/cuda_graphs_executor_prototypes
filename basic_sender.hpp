#pragma once

#include <stdexcept>
#include <iostream>

template<class Derived>
class basic_sender
{
  public:
    void submit()
    {
      // insert into the graph
      if(!created)
      {
        if(auto error = cudaGraphCreate(&graph, 0))
        {
          throw std::runtime_error("CUDA error after cudaGraphCreate: " + std::string(cudaGetErrorString(error)));
        }
        derived().insert(graph);
      }

      // launch the graph
      launch(graph);

    }

    void sync_wait() const
    {
      if(!event_)
      {
        throw std::runtime_error("basic_sender::sync_wait: invalid state");
      }

      if(auto error = cudaEventSynchronize(event_))
      {
        throw std::runtime_error("basic_sender::sync_wait: CUDA error after cudaEventSynchronize: " + std::string(cudaGetErrorString(error)));
      }
    }

  protected:
    basic_sender() 
      : event_{},
        instantiated{false},
        created{false}
    {}
    basic_sender(const basic_sender&)
      : event_{},
        instantiated{false},
        created{false}
    {}

    basic_sender(cudaStream_t stream)
      : stream_(stream),
        event_{},
        instantiated{false},
        created{false}
    {}

    ~basic_sender()
    {

      if(instantiated)
      {
        // delete the graph instance
        if(auto error = cudaGraphExecDestroy(executable_graph))
        {
          std::cerr << "basic_sender::launch: CUDA error after cudaGraphExecDestroy: " + std::string(cudaGetErrorString(error)) << std::endl;
          std::terminate();
        }
      }

      // destroy the graph
      if(created)
      {
        if(auto error = cudaGraphDestroy(graph))
        {
          std::cerr << "basic_sender::submit: CUDA error after cudaGraphDestroy: " + std::string(cudaGetErrorString(error)) << std::endl;
          std::terminate();
        }
      }
    }

  private:
    void launch(cudaGraph_t graph)
    {
      // instantiate the graph
      if ( !instantiated ) 
      {
        if(auto error = cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0))
        {
          throw std::runtime_error("basic_sender::launch: CUDA error after cudaGraphInstantiate: " + std::string(cudaGetErrorString(error)));
        }
        instantiated = true;
      }
      
      // launch the graph
      if(auto error = cudaGraphLaunch(executable_graph, stream_))
      {
        throw std::runtime_error("basic_sender::launch: CUDA error after cudaGraphLaunch: " + std::string(cudaGetErrorString(error)));
      }

      // create an event
      if(auto error = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming))
      {
        throw std::runtime_error("basic_sender::launch: CUDA error after cudaEventCreateWithFlags: " + std::string(cudaGetErrorString(error)));
      }

      // record an event
      if(auto error = cudaEventRecord(event_, stream_))
      {
        throw std::runtime_error("basic_sender::launch: CUDA error after cudaEventRecord: " + std::string(cudaGetErrorString(error)));
      }
    }

    const Derived& derived() const
    {
      return static_cast<const Derived&>(*this);
    }

    cudaStream_t stream_;
    cudaEvent_t event_;
    bool instantiated;
    bool created;
    cudaGraph_t graph;
    cudaGraphExec_t executable_graph{};
};

