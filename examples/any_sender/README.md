This example demonstrates how to use `any_sender` to pass any type of sender through API interfaces.

This repository contains many types of senders; for example, `kernel_sender` and `copy_sender`.

In many cases, we can write code that operates on senders generically with
templates. For example, the following function takes a `graph_executor` and any
type of sender and generically appends a kernel launch which depends on that sender:

   template<class Sender>
   kernel_sender append_kernel(const graph_executor& ex, Sender& sender)
   {
     f = ...
     grid_index shape = ...

     kernel_sender result = ex.bulk_then_execute(f, shape, sender);
     return result;
   }

However, there exist programming contexts in which template code is
inappropriate. For example, consider a program composed of multiple
separately-compiled objects. In such a setting, the functionality of these
objects cannot be exposed via function templates. Instead, it must be exposed
via a binary API composed of concrete functions.

To create a such binary interface, we need to transform function templates like
`append_kernel` into normal functions by replacing their `Sender` template
parameters with a special type of sender:

    any_sender append_kernel(const graph_executor& ex, any_sender& sender)
    {
      f = ...
      grid_index shape = ...

      kernel_sender result = ex.bulk_then_execute(f, shape, sender);
      return result;
    }

This is the purpose of `any_sender`, which can be used in non-template contexts
as a container for any other type of concrete sender. It can be used as a
function parameter for senders, and as a function result which "forgets" about
the specific type of sender which it had been assigned.

This directory contains a program which separates the `daxpy` example into three separately-compiled source files:

    * `main.cu`: Contains the `main` function which calls `test`.
    * `test.cu`: Contains the `test` function which calls `daxpy`.
    * `daxpy.cu`: Contains the `daxpy` function which returns an `any_sender`.

To build the program, compile these source files separately and link them together:

    $ nvcc -std=c++11 --expt-extended-lambda -I../.. main.cu test.cu daxpy.cu
    $ ./a.out 
    OK

