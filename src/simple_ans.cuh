#ifndef SIMPLE_ANS
#define SIMPLE_ANS 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <vector>


std::vector<unsigned int> encode(
    const std::vector<float> mu,
    const std::vector<float> sigma,
    const std::vector<int> symbols,
    const int left_inclusive_symbol,
    const int right_inclusive_symbol,
    const unsigned int precision
);


torch::Tensor decode(
    const torch::Tensor& mu,
    const torch::Tensor& sigma,
    const torch::Tensor& bitstream,

    const int left_inclusive_symbol,
    const int right_inclusive_symbol,
    const unsigned int precision
    
    );

std::tuple<torch::Tensor, torch::Tensor> calc_grid(
    const torch::Tensor& mu,
    const torch::Tensor& sigma,

    const int left_inclusive_symbol,
    const int right_inclusive_symbol,
    const unsigned int precision
    
    );

#endif /*  SIMPLE_ANS */

