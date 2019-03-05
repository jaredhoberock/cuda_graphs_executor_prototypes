// $ nvcc -c -std=c++11 --expt-extended-lambda -I../.. main.cu -o main.o

#include <iostream>
#include "test.hpp"

int main(int argc, char** argv)
{
  size_t n = 1 << 25;
  if(argc > 1)
  {
    n = std::atoi(argv[1]);
  }

  // test for correctness
  test(n);

  std::cout << "OK" << std::endl;

  return 0;
}

