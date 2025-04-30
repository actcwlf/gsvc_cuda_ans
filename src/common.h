#pragma once
#include "torch/extension.h"


struct QuantizedDistribution {
    int m_left_inclusive_symbol;
    int m_right_inclusive_symbol;
    unsigned int m_support_size;
    unsigned int m_precision;
    QuantizedDistribution() {}
    virtual std::tuple<torch::Tensor, torch::Tensor> cdf_pmf_per_symbol(const torch::Tensor & symbols) = 0;
    virtual std::tuple<torch::Tensor, torch::Tensor> cdf_pmf_grid() = 0;
};