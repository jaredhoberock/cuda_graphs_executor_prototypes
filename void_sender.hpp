#pragma once

#include <stdexcept>
#include <string>

class void_sender
{
  public:
    cudaGraphNode_t insert(cudaGraph_t g) const
    {
      // create an empty node
      cudaGraphNode_t result{};
      if(auto error = cudaGraphAddEmptyNode(&result, g, nullptr, 0))
      {
        throw std::runtime_error("void_sender::insert: CUDA error after cudaGraphAddEmptyNode: " + std::string(cudaGetErrorString(error)));
      }

      // return the node
      return result;
    }
};

