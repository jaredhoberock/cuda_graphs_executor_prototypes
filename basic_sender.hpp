#pragma once

#include <stdexcept>
#include <iostream>

template<class Derived>
class basic_sender
{
  public:
    void submit()
    {
      // create the graph if it does not yet exist
      if(!graph_)
      {
        if(auto error = cudaGraphCreate(&graph_, 0))
        {
          throw std::runtime_error("basic_sender::submit: CUDA error after cudaGraphCreate: " + std::string(cudaGetErrorString(error)));
        }

        derived().insert(graph_);
      }

      // launch the graph
      launch(graph_);
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
      : stream_{},
        event_{},
        graph_{},
        executable_graph_{}
    {}

    basic_sender(basic_sender&& other)
      : basic_sender()
    {
      std::swap(stream_, other.stream_);
      std::swap(event_, other.event_);
      std::swap(graph_, other.graph_);
      std::swap(executable_graph_, other.executable_graph_);
    }

    basic_sender(cudaStream_t stream)
      : stream_(stream),
        event_{},
        graph_{},
        executable_graph_{}
    {}

    ~basic_sender()
    {
      if(executable_graph_)
      {
        // destroy the graph instance
        if(auto error = cudaGraphExecDestroy(executable_graph_))
        {
          std::cerr << "basic_sender::~basic_sender: CUDA error after cudaGraphExecDestroy: " + std::string(cudaGetErrorString(error)) << std::endl;
          std::terminate();
        }
      }

      // destroy the graph if it exists
      if(graph_)
      {
        if(auto error = cudaGraphDestroy(graph_))
        {
          std::cerr << "basic_sender::~basic_sender: CUDA error after cudaGraphDestroy: " + std::string(cudaGetErrorString(error)) << std::endl;
          std::terminate();
        }
      }
    }

  private:
    void launch(cudaGraph_t graph)
    {
      // instantiate the graph if it is not already instantiated
      if(!executable_graph_)
      {
        if(auto error = cudaGraphInstantiate(&executable_graph_, graph_, nullptr, nullptr, 0))
        {
          throw std::runtime_error("basic_sender::launch: CUDA error after cudaGraphInstantiate: " + std::string(cudaGetErrorString(error)));
        }
      }
      
      // launch the graph
      if(auto error = cudaGraphLaunch(executable_graph_, stream_))
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
    cudaGraph_t graph_;
    cudaGraphExec_t executable_graph_{};
};

