#pragma once
// #include "ATen/core/TensorBody.h"
#include "torch/extension.h"
#include "common.h"

void __global__ gaussian_pmf_grid(
    const float * mu,     // size N
    const float * sigam,  // size N

    unsigned int * pmf_grid,  // size N x (rin_symbol - lin_symbol + 1)
    const unsigned int free_weight,
    const int lin_symbol, // left inclusive symbol
    const int rin_symbol, // right inclusive symbol
    const unsigned long max_weight,
    const unsigned int N // number of entry
);


void __global__ gaussian_pmf_per_symbol(
    const float * mu, const float * sigam, 
    const int * symbols, 
    unsigned int * pmf_array,
    const unsigned int precision,
    const unsigned int N
);


void __global__ gaussian_cdf_pmf_per_symbol(
    const float * mu,
    const float * sigam, 
    const int * symbols, 
    unsigned int * cdf_array,
    unsigned int * pmf_array,
    const unsigned int free_weight,
    const int lin_symbol, // left inclusive symbol
    const int rin_symbol, // right inclusive symbol
    const unsigned long max_weight,
    const unsigned int N
);

void __global__ gaussian_cdf_grid(
    const float * mu, const float * sigma, 
    unsigned int * cdf_grid,
    const unsigned int free_weight,
    const int lin_symbol, // left inclusive symbol
    const int rin_symbol, // right inclusive symbol
    const unsigned long max_weight,
    const unsigned int N // number of entry
);


struct QuantizedGaussian: public QuantizedDistribution {

    unsigned int m_free_weight;
    unsigned long m_weight_budget;
    const torch::Tensor m_mu;  // 当前仅支持向量 这里不能使用引用
    const torch::Tensor m_sigma;




    QuantizedGaussian(
        const torch::Tensor & mu,
        const torch::Tensor & sigma, 
        const int left_inclusive_symbol, 
        const int right_inclusive_symbol, 
        const unsigned int precision): m_mu(mu), m_sigma(sigma) {

        m_left_inclusive_symbol = left_inclusive_symbol;
        m_right_inclusive_symbol = right_inclusive_symbol;
        m_support_size = right_inclusive_symbol - left_inclusive_symbol + 1;
        m_precision = precision;
        m_weight_budget = 1l << precision;
        m_free_weight = m_weight_budget - m_support_size;
    }

    std::tuple<torch::Tensor, torch::Tensor> cdf_pmf_per_symbol(const torch::Tensor & symbols) override;
    
    //  {
    //     const unsigned int N = symbols.size(0);
    //     auto int_opts = symbols.options().dtype(torch::kInt32);
    //     torch::Tensor pmf = torch::full({N}, 0, int_opts);
    //     torch::Tensor cdf = torch::full({N}, 0, int_opts);
        
    //     const int block_size = 512;
    //     // const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    //     const int grid_size = (N - 1) / block_size + 1;

    //     gaussian_cdf_pmf_per_symbol<<<grid_size, block_size>>>(
    //         m_mu.contiguous().data_ptr<float>(),
    //         m_sigma.contiguous().data_ptr<float>(),
    //         symbols.contiguous().data_ptr<int>(),
    //         reinterpret_cast<unsigned int*>(cdf.contiguous().data_ptr<int>()),
    //         reinterpret_cast<unsigned int*>(pmf.contiguous().data_ptr<int>()),
    //         m_free_weight,
    //         m_left_inclusive_symbol,
    //         m_right_inclusive_symbol,
    //         m_weight_budget,
    //         N
    //     );

    //     return std::make_tuple(cdf, pmf);

    // }

    std::tuple<torch::Tensor, torch::Tensor> cdf_pmf_grid() override;
    
    //  {
    //     const unsigned int N = m_mu.size(0);
    //     const int block_size = 512;
    //     // const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    //     const int grid_size = (N - 1) / block_size + 1;

    //     auto int_opts = m_mu.options().dtype(torch::kInt32);

    //     torch::Tensor pmf_grid = torch::full({N, m_support_size}, 0, int_opts);
    //     torch::Tensor cdf_grid = torch::full({N, m_support_size + 1}, 0, int_opts);



    //     gaussian_cdf_grid<<<grid_size, block_size>>>(
    //         m_mu.contiguous().data_ptr<float>(),
    //         m_sigma.contiguous().data_ptr<float>(),
    //         reinterpret_cast<unsigned int*>(cdf_grid.contiguous().data_ptr<int>()),
    //         m_free_weight,
    //         m_left_inclusive_symbol,
    //         m_right_inclusive_symbol,
    //         m_weight_budget,
    //         N // number of entry
    //     );

    //     // pmf 应该可以直接从cdf中计算得出，待优化
    //     gaussian_pmf_grid<<<grid_size, block_size>>>(
    //         m_mu.contiguous().data_ptr<float>(),
    //         m_sigma.contiguous().data_ptr<float>(),
    //         reinterpret_cast<unsigned int*>(pmf_grid.contiguous().data_ptr<int>()),
    //         m_free_weight,
    //         m_left_inclusive_symbol,
    //         m_right_inclusive_symbol,
    //         m_weight_budget,
    //         N // number of entry
    //     );

    //     return std::make_tuple(cdf_grid, pmf_grid);
    // }
};