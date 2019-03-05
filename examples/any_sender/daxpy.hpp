#pragma once

#include <graph_executor.hpp>
#include <any_sender.hpp>

any_sender daxpy(const graph_executor& ex, int n, double a, const double* x, double* y);

