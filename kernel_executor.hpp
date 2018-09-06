#pragma once

#include <type_traits>
#include <memory>

namespace detail
{
namespace basic_kernel_executor_detail
{


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
           std::is_empty<typename std::result_of<Factory()>::type>
         )>
__device__ decltype(Factory()) synchronize_block_and_make_inner_shared_object(Factory)
{
  using result_type = typename std::result_of<Factory()>::type;

  // the result of the Factory is an empty type
  // so there's no reason to actually call the factory or put its result into __shared__ memory
  return result_type{};
}


template<class Factory,
         __KERNEL_EXECUTOR_REQUIRES(
           !std::is_empty<typename std::result_of<Factory()>::type>
         )>
__device__ decltype(Factory())& make_inner_shared_object_and_synchronize_block(Factory factory)
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
  return *reinterpret_cast<result_type*>(&result_storage);
}


template<class Index, class Function, class Pointer, class InnerFactory>
__global__ void kernel(Function f, Pointer outer_shared_ptr, InnerFactory inner_shared_factory)
{
  // create the index
  Index idx(blockIdx, threadIdx);

  // create the inner shared object
  auto& inner_shared_object = make_inner_shared_object_and_synchronize_block(inner_shared_factory);

  // get the outer shared object
  auto& outer_shared_object = *outer_shared_ptr;

  // invoke f
  f(idx, outer_shared_object, inner_shared_object);

  // destroy the inner shared object if necessary
  synchronize_block_and_destroy_if_nontrivial(inner_shared_object);
}


} // end basic_kernel_executor_detail
} // end detail


template<class Shape, class Allocator>
class basic_kernel_executor
{
  public:
    basic_kernel_executor(cudaStream_t stream, const Allocator& allocator = Allocator())
      : stream_(stream),
        allocator_(allocator)
    {}

    template<class Function, class OuterFactory, class InnerFactory>
    void bulk_execute(Function f, Shape shape, OuterFactory outer_shared_factory, InnerFactory inner_shared_factory) const
    {
      // allocate the outer shared parameter
      auto outer_shared_object_ptr = make_outer_shared_object(outer_shared_factory);

      // pack parameter addresses into an array
      void* parameter_array[] = {&f, &outer_shared_factory.get(), &inner_shared_factory};

      // launch the kernel
      if(auto error = cudaLaunchKernel(kernel_ptr, std::get<0>(shape), std::get<1>(shape), parameter_array, 0, stream_))
      {
        throw std::runtime_error("basic_kernel_executor::bulk_execute: CUDA error after cudaLaunchKernel: " + std::string(cudaGetErrorString(error)));
      }

      // create a stream callback to delete the outer shared parameter
      cudaStreamCallback_t callback = &delete_on_new_thread<outer_shared_object_ptr::element_type>;
      auto* raw_ptr = outer_shared_object_ptr.release();
      if(auto error = cudaStreamAddCallback(stream_, callback, raw_ptr, 0))
      {
        // put the raw pointer back into the unique_ptr
        outer_shared_object_ptr.reset(raw_ptr);

        throw std::runtime_error("basic_kernel_executor::bulk_execute: CUDA error after cudaStreamAddCallback: " + std::string(cudaGetErrorString(error)));
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
    // this function object is a "deleter" which destroys and
    // deallocates its argument using the stored allocator
    template<class Allocator>
    struct delete_with_allocator
    {
      using pointer = typename std::allocator_traits<
        Allocator
      >::pointer;

      void operator()(pointer ptr)
      {
        // destroy the object
        std::allocator_traits<Allocator>::destroy(allocator_, ptr);

        // deallocate the storage
        allocator_.deallocate(ptr);
      }

      Allocator allocator_;
    };

    template<class T>
    static void delete_on_new_thread(cudaStream_t, cudaError_t, void* raw_ptr)
    {
      // delete the pointer in a thread besides the one that is executing this stream callback
      std::thread new_thread([=raw_ptr]
      {
        // get the right type of allocator to deallocate the object
        // XXX this assumes that Allocator is default-constructible
        //     a way around this might be to pass a pointer to a 
        //     self-deallocating object to the callback
        //     the object could contain the appropriate allocator
        //     to do the deallocation
        std::allocator_traits<Allocator>::template rebind_alloc<T> allocator = Allocator{};

        // create the appropriate type of deleter
        delete_with_allocator<decltype(allocator)> deleter(allocator);

        // cast the raw_ptr to the appropriate type of pointer and delete it
        deleter(reinterpret_cast<T*>(raw_ptr));
      });

      // do not join with the thread
      new_thread.detach();
    }

    template<class Factory,
             // figure out the type returned by the factory
             class Result = typename std::result_of<Factory()>::type,
             // rebind Allocator to the appropriate type
             class ResultAllocator = std::allocator_traits<Allocator>::template rebind_alloc<Result>,
             // figure out the appropriate type of Deleter
             class Deleter = delete_with_allocator<ResultAllocator>
            >
    static std::unique_ptr<Result,Deleter>
      make_outer_shared_object(Factory factory)
    {
      // make an allocator for the result 
      ResultAllocator result_allocator = allocator_;

      // allocate the result into a unique_ptr
      std::unique_ptr<Result, Deleter> result(result_allocator.allocate(1), Deleter(result_allocator));

      // construct the result using the result of the factory
      result_allocator.construct(result.get(), factory());     

      return result;
    }

    cudaStream_t stream_;
    Allocator allocator_;
};

