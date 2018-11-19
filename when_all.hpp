#pragma once

#include <basic_sender.hpp>
#include <any_sender.hpp>
#include <iterator>
#include <vector>
#include <algorithm>


class join_sender : public basic_sender<join_sender>
{
  private:
    using super_t = basic_sender<join_sender>;

    // friend these types so they can access insert()
    friend super_t;
    friend any_sender;

    std::vector<any_sender> predecessors_;

  public:
    template<class SenderIterator>
    join_sender(cudaStream_t stream, SenderIterator predecessors_begin, SenderIterator predecessors_end)
      : super_t(stream),
        predecessors_(std::make_move_iterator(predecessors_begin), std::make_move_iterator(predecessors_end))
    {}

  private:
    // this function transliterates the DAG of senders into a graph
    cudaGraphNode_t insert(cudaGraph_t g) const
    {
      // insert the predecessors
      std::vector<cudaGraphNode_t> predecessor_nodes(predecessors_.size());
      std::transform(predecessors_.begin(), predecessors_.end(), predecessor_nodes.begin(), [g](const any_sender& predecessor)
      {
        return predecessor.insert(g);
      });

      // introduce a new empty node
      cudaGraphNode_t result_node{};
      if(auto error = cudaGraphAddEmptyNode(&result_node, g, predecessor_nodes.data(), predecessor_nodes.size()))
      {
        throw std::runtime_error("join_sender::insert: CUDA error after cudaGraphAddEmptyNode: " + std::string(cudaGetErrorString(error)));
      }

      return result_node;
    }
};


template<class CudaGraphExecutor, class SenderIterator>
join_sender when_all(const CudaGraphExecutor& ex, SenderIterator predecessors_first, SenderIterator predecessors_last)
{
  return {ex.stream(), predecessors_first, predecessors_last};
}

template<class CudaGraphExecutor, class SenderRange>
join_sender when_all(const CudaGraphExecutor& ex, SenderRange& predecessors)
{
  return when_all(ex, predecessors.begin(), predecessors.end());
}

