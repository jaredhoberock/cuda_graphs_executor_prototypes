#pragma once

template<class Derived, class CudaGraphExecutor, class Function, class Sender>
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
        throw std::runtime_error("basic_sender::submit: CUDA error after cudaGraphDestroy: " + std::string(cudaGetErrorString(error)));
      }
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
    basic_sender(const CudaGraphExecutor& executor, Function function, Sender&& predecessor)
      : executor_(executor),
        function_(function),
        predecessor_(std::move(predecessor)),
        event_{}
    {}

    const Function& function() const
    {
      return function_;
    }

    const Sender& predecessor() const
    {
      return predecessor_;
    }

  private:
    cudaGraph_t make_graph() const
    {
      // create a new graph
      cudaGraph_t graph{};
      if(auto error = cudaGraphCreate(&graph, 0))
      {
        throw std::runtime_error("CUDA error after cudaGraphCreate: " + std::string(cudaGetErrorString(error)));
      }

      // insert into the graph
      derived().insert(graph);

      // return the graph
      return graph;
    }

    void launch(cudaGraph_t graph)
    {
       // instantiate the graph
       cudaGraphExec_t executable_graph{};
       if(auto error = cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0))
       {
         throw std::runtime_error("basic_sender::launch: CUDA error after cudaGraphInstantiate: " + std::string(cudaGetErrorString(error)));
       }
    
       // launch the graph
       if(auto error = cudaGraphLaunch(executable_graph, executor().stream()))
       {
         throw std::runtime_error("basic_sender::launch: CUDA error after cudaGraphLaunch: " + std::string(cudaGetErrorString(error)));
       }

       // create an event
       if(auto error = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming))
       {
         throw std::runtime_error("basic_sender::launch: CUDA error after cudaEventCreateWithFlags: " + std::string(cudaGetErrorString(error)));
       }

       // record an event
       if(auto error = cudaEventRecord(event_, executor().stream()))
       {
         throw std::runtime_error("basic_sender::launch: CUDA error after cudaEventRecord: " + std::string(cudaGetErrorString(error)));
       }
    
       // delete the graph instance
       if(auto error = cudaGraphExecDestroy(executable_graph))
       {
         throw std::runtime_error("basic_sender::launch: CUDA error after cudaGraphExecDestroy: " + std::string(cudaGetErrorString(error)));
       }
    }

    const Derived& derived() const
    {
      return static_cast<const Derived&>(*this);
    }

    CudaGraphExecutor executor_;
    Function function_;
    Sender predecessor_;
    cudaEvent_t event_;
};

