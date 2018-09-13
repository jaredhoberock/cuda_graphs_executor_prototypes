#pragma once

#include <stdexcept>
#include <vector>
#include <utility>


class cuda_context
{
  public:
    ~cuda_context()
    {
      wait();
    }

    template<class Function>
    void invoke_after(cudaEvent_t dependency, Function&& f)
    {
      std::function<void()> g = std::forward<Function>(f);
      outstanding_work_.push_back(std::make_pair(dependency, g));
    }

    void wait()
    {
      for(auto& task : outstanding_work_)
      {
        // wait for the event to complete
        if(auto error = cudaEventSynchronize(task.first))
        {
          throw std::runtime_error("cuda_context::wait: CUDA error after cudaEventSynchronize: " + std::string(cudaGetErrorString(error)));
        }

        // invoke the function
        task.second();

        // destroy the event
        if(auto error = cudaEventDestroy(task.first))
        {
          throw std::runtime_error("cuda_context::wait: CUDA error after cudaEventDestroy: " + std::string(cudaGetErrorString(error)));
        }
      }

      outstanding_work_.clear();
    }

  private:
    // XXX should probably be protected by a mutex or something
    std::vector<std::pair<cudaEvent_t, std::function<void()>>> outstanding_work_;
};

