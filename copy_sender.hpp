#pragma once

#include <basic_sender.hpp>
#include <any_sender.hpp>
#include <stdexcept>
#include <functional>
#include <iostream>


class copy_sender : public basic_sender<copy_sender>
{
  private:
    using super_t = basic_sender<copy_sender>;

    // friend these types so they can access insert()
    friend super_t;
    friend any_sender;

    std::function<cudaMemcpy3DParms()> node_params_function_;
    any_sender predecessor_;

  public:
    template<class NodeParamsFunction>
    copy_sender(cudaStream_t stream, NodeParamsFunction node_params_function, any_sender&& predecessor)
      : super_t(stream),
        node_params_function_(node_params_function),
        predecessor_(std::move(predecessor))
    {}

  protected:
    cudaGraphNode_t insert(cudaGraph_t g) const
    {
      // insert the predecessor
      cudaGraphNode_t predecessor_node = predecessor_.insert(g);

      // generate the node parameters
      cudaMemcpy3DParms node_params = node_params_function_();

      // introduce a new kernel node
      cudaGraphNode_t result_node{};
      if(auto error = cudaGraphAddMemcpyNode(&result_node, g, &predecessor_node, 1, &node_params))
      {
        throw std::runtime_error("copy_sender::insert: CUDA error after cudaGraphAddMemcpyNode: " + std::string(cudaGetErrorString(error)));
      }

      return result_node;
    }
};

