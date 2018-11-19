#pragma once

#include <stdexcept>
#include <string>
#include <basic_sender.hpp>
#include <any_sender.hpp>

class void_sender : public basic_sender<void_sender>
{
  public:
    void_sender() = default;
    void_sender(const void_sender&) = default;
    void_sender(void_sender&&) = default;

  protected:
    using super_t = basic_sender<void_sender>;

    // friend these types so they can access insert()
    friend super_t;
    friend any_sender;

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

