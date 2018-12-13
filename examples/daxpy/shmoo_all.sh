#!/bin/bash -v

./shmoo_daxpy.py basic_daxpy > basic_daxpy.csv
./shmoo_daxpy.py graph_daxpy > graph_daxpy.csv
./shmoo_daxpy.py kernel_executor_daxpy > kernel_executor_daxpy.csv
./shmoo_daxpy.py graph_executor_daxpy > graph_executor_daxpy.csv
