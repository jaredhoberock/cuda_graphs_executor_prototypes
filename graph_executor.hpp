#pragma once


namespace detail
{


template<class Function>
__global__ void kernel(Function f)
{
  f();
}


} // end detail


class graph_executor;


template<class Function, class Sender>
class single_sender
{
  public:
    void submit(cudaStream_t stream)
    {
      // create a new graph
      cudaGraph_t graph = make_graph();

      // instantiate the graph
      cudaGraphExec_t executable_graph{};
      if(auto error = cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0))
      {
        throw std::runtime_error("CUDA error after cudaGraphInstantiate: " + std::string(cudaGetErrorString(error)));
      }

      // launch the graph
      if(auto error = cudaGraphLaunch(executable_graph, stream))
      {
        throw std::runtime_error("CUDA error after cudaGraphLaunch: " + std::string(cudaGetErrorString(error)));
      }

      // delete the graph instance
      if(auto error = cudaGraphExecDestroy(executable_graph))
      {
        throw std::runtime_error("CUDA error after cudaGraphExecDestroy: " + std::string(cudaGetErrorString(error)));
      }

      // delete the graph
      if(auto error = cudaGraphDestroy(graph))
      {
        throw std::runtime_error("CUDA error after cudaGraphDestroy: " + std::string(cudaGetErrorString(error)));
      }
    }

  private:
    friend class graph_executor;
    template<class,class> friend class single_sender;

    single_sender(Function f, Sender&& predecessor)
      : function_(f),
        predecessor_(std::move(predecessor))
    {}

    // this function transliterates the chain of predecessor into a graph
    cudaGraphNode_t insert(cudaGraph_t g) const
    {
      // insert the predecessor
      cudaGraphNode_t predecessor_node = predecessor_.insert(g);

      // introduce a new kernel node
      cudaGraphNode_t result_node{};
      void* kernel_params[] = {reinterpret_cast<void*>(const_cast<Function*>(&function_))};
      cudaKernelNodeParams node_params
      {
        reinterpret_cast<void*>(&detail::kernel<Function>),
        dim3{1},
        dim3{1},
        0,
        kernel_params,
        nullptr
      };

      if(auto error = cudaGraphAddKernelNode(&result_node, g, &predecessor_node, 1, &node_params))
      {
        throw std::runtime_error("CUDA error after cudaGraphAddKernelNode: " + std::string(cudaGetErrorString(error)));
      }

      // discard the predecessor node
      // XXX not sure if this is best practice or not
      //     seems like a leak

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
      // discard the returned node
      // XXX not sure if this is best practice or not
      //     seems like a leak
      insert(graph);

      // return the graph
      return graph;
    }

    static void invoke_host_function(void* user_data)
    {
      // cast the user data into a Function
      Function* function = reinterpret_cast<Function*>(user_data);

      // invoke the function
      (*function)();

      // delete function
      delete function;
    }

    Function function_;
    Sender predecessor_;
};


class void_sender
{
  public:
    cudaGraphNode_t insert(cudaGraph_t g) const
    {
      // create an empty node
      cudaGraphNode_t result{};
      if(auto error = cudaGraphAddEmptyNode(&result, g, nullptr, 0))
      {
        throw std::runtime_error("CUDA error after cudaGraphAddEmptyNode: " + std::string(cudaGetErrorString(error)));
      }

      // return the node
      return result;
    }
};


class graph_executor
{
  public:
    template<class Function, class Sender>
    single_sender<Function,Sender> then_execute(Function f, Sender& sender) const
    {
      return single_sender<Function,Sender>(f, std::move(sender));
    }
};

