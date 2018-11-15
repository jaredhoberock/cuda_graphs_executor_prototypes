#pragma once

#include <type_traits>
#include <memory>
#include <thread>

#include <managed_allocator.hpp>
#include <grid_index.hpp>
#include <cuda_context.hpp>

#define __KERNEL_EXECUTOR_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr

namespace detail
{
namespace basic_kernel_executor_detail
{

// this function object is a "deleter" which destroys and
// deallocates its argument using the stored allocator
template<class Allocator>
struct delete_with_allocator
{
  using pointer = typename std::allocator_traits<
    Allocator
  >::pointer;

  delete_with_allocator(const Allocator& allocator)
    : allocator_(allocator)
  {}

  void operator()(pointer ptr)
  {
    // destroy the object
    std::allocator_traits<Allocator>::destroy(allocator_, ptr);

    // deallocate the storage
    allocator_.deallocate(ptr, 1);
  }

  Allocator allocator_;
};


template<class Empty>
struct empty_ptr : public Empty
{
  __host__ __device__
  const Empty& operator*() const
  {
    return *this;
  }

  __host__ __device__
  Empty& operator*()
  {
    return *this;
  }

  // mimic this part of unique_ptr<Empty>'s interface
  __host__ __device__
  empty_ptr<Empty> get() const
  {
    return *this;
  }
};


template<class T,
         __KERNEL_EXECUTOR_REQUIRES(
           std::is_trivially_destructible<T>::value
         )>
__device__ void synchronize_block_and_destroy_if_nontrivial(T&)
{
  // nothing needs to be done because T has a trivial destructor
}


template<class T,
         __KERNEL_EXECUTOR_REQUIRES(
           !std::is_trivially_destructible<T>::value
         )>
__device__ void synchronize_block_and_destroy_if_nontrivial(T& object)
{
  __syncthreads();

  // have thread 0 destroy the object
  if(threadIdx.x == 0 and threadIdx.y == 0 and threadIdx.z == 0)
  {
    object.~T();
  }
}


template<class Factory,
         __KERNEL_EXECUTOR_REQUIRES(
           std::is_empty<
             typename std::result_of<Factory()>::type
           >::value
         )>
__device__ typename std::result_of<Factory()>::type*
  make_inner_shared_object_and_synchronize_block(Factory)
{
  // the inner shared object is empty, so there's no need to create one
  return nullptr;
}


template<class Factory,
         __KERNEL_EXECUTOR_REQUIRES(
           !std::is_empty<
             typename std::result_of<Factory()>::type
           >::value
         )>
__device__ typename std::result_of<Factory()>::type*
  make_inner_shared_object_and_synchronize_block(Factory factory)
{
  using result_type = typename std::result_of<Factory()>::type;

  // create static __shared__ storage for the result of the factory
  __shared__ typename std::aligned_storage<sizeof(result_type)>::type result_storage;

  // have thread 0 call the factory
  if(threadIdx.x == 0 and threadIdx.y == 0 and threadIdx.z == 0)
  {
    new(&result_storage) result_type(factory());
  }

  __syncthreads();

  // note that we return a reference from this function
  return reinterpret_cast<result_type*>(&result_storage);
}


template<class Function, class Pointer, class InnerFactory>
__global__ void kernel(Function f, Pointer outer_shared_ptr, InnerFactory inner_shared_factory)
{
  // create the index
  grid_index idx{blockIdx, threadIdx};

  // create the inner shared object
  auto* inner_shared_object_ptr = make_inner_shared_object_and_synchronize_block(inner_shared_factory);
  auto& inner_shared_object = *inner_shared_object_ptr;

  // get the outer shared object
  auto& outer_shared_object = *outer_shared_ptr;

  // invoke f
  f(idx, outer_shared_object, inner_shared_object);

  // destroy the inner shared object if necessary
  synchronize_block_and_destroy_if_nontrivial(inner_shared_object);
}


} // end basic_kernel_executor_detail
} // end detail


template<class Allocator>
class basic_kernel_executor
{
  public:
    basic_kernel_executor(cuda_context& context, cudaStream_t stream, const Allocator& allocator = Allocator())
      : context_(context),
        stream_(stream),
        allocator_(allocator)
    {}

    template<class GlobalFunctionPtr, class... Args>
    void bulk_execute_global_function(GlobalFunctionPtr kernel_ptr, grid_index shape, const Args&... args) const
    {
      // pack parameter addresses into an array
      void* parameter_array[] = {const_cast<Args*>(&args)...};

      // launch the kernel
      if(auto error = cudaLaunchKernel(reinterpret_cast<void*>(kernel_ptr), shape[0], shape[1], parameter_array, 0, stream_))
      {
        throw std::runtime_error("basic_kernel_executor::bulk_execute_global_function: CUDA error after cudaLaunchKernel: " + std::string(cudaGetErrorString(error)));
      }
    }

    template<class Function, class OuterFactory, class InnerFactory>
    void bulk_execute(Function f, grid_index shape, OuterFactory outer_shared_factory, InnerFactory inner_shared_factory) const
    {
      // allocate the outer shared parameter
      auto outer_shared_object_ptr = make_outer_shared_object(outer_shared_factory);

      // instantiate the kernel
      void* kernel_ptr = reinterpret_cast<void*>(&detail::basic_kernel_executor_detail::kernel<Function, decltype(outer_shared_object_ptr.get()), InnerFactory>);

      // execute the kernel
      bulk_execute_global_function(kernel_ptr, shape, f, outer_shared_object_ptr.get(), inner_shared_factory);

      // destroy the outer shared parameter after the kernel completes
      schedule_outer_shared_object_for_deletion(std::move(outer_shared_object_ptr));
    }

    void wait() const
    {
      if(auto error = cudaStreamSynchronize(stream_))
      {
        throw std::runtime_error("basic_kernel_executor::wait: CUDA error after cudaStreamSynchronize: " + std::string(cudaGetErrorString(error)));
      }
    }

    bool operator==(const basic_kernel_executor& other) const
    {
      return (stream_ == other.stream_) and (allocator_ == other.allocator_);
    }

    bool operator!=(const basic_kernel_executor& other) const
    {
      return !(*this == other);
    }

  private:
    template<class T, class Deleter>
    void schedule_outer_shared_object_for_deletion(std::unique_ptr<T,Deleter> ptr) const
    {
      // tell the context to destroy the outer shared parameter after the kernel completes
      // XXX if the outer shared object is empty, there's no need to do this
      cudaEvent_t event{};
      if(auto error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming))
      {
        throw std::runtime_error("basic_kernel_executor::bulk_execute: CUDA error after cudaEventCreateWithFlags: " + std::string(cudaGetErrorString(error)));
      }

      // decompose the unique_ptr so that the context can use a std::function to store the task
      T* raw_ptr = ptr.release();
      Deleter deleter = ptr.get_deleter();

      // delete the pointer after the event completes 
      // the context will destroy the event
      context_.invoke_after(event, [=]() mutable
      {
        deleter(raw_ptr);
      });
    }

    template<class T>
    void schedule_outer_shared_object_for_deletion(detail::basic_kernel_executor_detail::empty_ptr<T>) const
    {
      // because T is an empty type, there is nothing to destroy or deallocate
    }

    template<class Factory,
             // figure out the type returned by the factory
             class Result = typename std::result_of<Factory()>::type,
             // only enable this overload if Result is a non-empty type
             __KERNEL_EXECUTOR_REQUIRES(!std::is_empty<Result>::value),
             // rebind Allocator to the appropriate type
             class ResultAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<Result>,
             // figure out the appropriate type of Deleter
             class Deleter = detail::basic_kernel_executor_detail::delete_with_allocator<ResultAllocator>
            >
    std::unique_ptr<Result,Deleter>
      make_outer_shared_object(Factory factory) const
    {
      // make an allocator for the result 
      ResultAllocator result_allocator = allocator_;

      // allocate the result into a unique_ptr
      std::unique_ptr<Result, Deleter> result(result_allocator.allocate(1), Deleter(result_allocator));

      // construct the result using the result of the factory
      std::allocator_traits<ResultAllocator>::construct(result_allocator, result.get(), factory());     

      return result;
    }

    template<class Factory,
             // figure out the type returned by the factory
             class Result = typename std::result_of<Factory()>::type,
             // only enable this overload if Result is an empty type
             __KERNEL_EXECUTOR_REQUIRES(std::is_empty<Result>::value)
            >
    detail::basic_kernel_executor_detail::empty_ptr<Result>
      make_outer_shared_object(Factory) const
    {
      return detail::basic_kernel_executor_detail::empty_ptr<Result>{};
    }

    cuda_context& context_;
    cudaStream_t stream_;
    Allocator allocator_;
};


using kernel_executor = basic_kernel_executor<managed_allocator<int>>;

