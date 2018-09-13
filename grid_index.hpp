#pragma once

class grid_index
{
  public:
    __host__ __device__
    grid_index(const dim3& block_index, const dim3& thread_index)
      : block_and_thread_index_{block_index, thread_index}
    {}

    __host__ __device__
    grid_index()
      : grid_index(dim3(), dim3())
    {}

    __host__ __device__
    dim3& operator[](int i)
    {
      return block_and_thread_index_[i];
    }

    __host__ __device__
    const dim3& operator[](int i) const
    {
      return block_and_thread_index_[i];
    }

  private:
    dim3 block_and_thread_index_[2];
};

