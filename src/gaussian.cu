#include <algorithm>
#include <cooperative_groups.h>
#include <iostream>
#include <cuda_fp16.h>
#include <cstdio>
#include <torch/extension.h>
#include "gaussian.cuh"

float __device__ gaussian_cdf(const float mu, const float sigma, const float x) {
    const float scaled_x = (x - mu) / (sigma) * rsqrtf(2);
    return 0.5 * ( 1 + erff(scaled_x));
}

float __device__ gaussian_icdf(const float mu, const float sigma, const float x) {
    return mu + sigma * erfinvf(2 * x - 1) * sqrtf(2);
}


void __global__ gaussian_cdf_grid(
    const float * mu, const float * sigma, 
    unsigned int * cdf_grid,
    const unsigned int free_weight,
    const int lin_symbol, // left inclusive symbol
    const int rin_symbol, // right inclusive symbol
    const unsigned long max_weight,
    const unsigned int N // number of entry
) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_t = blockDim.x * gridDim.x;
    const unsigned int total_symbol = (rin_symbol - lin_symbol + 1);

    while (tid < N * total_symbol)
    {   
        const unsigned entry = tid / total_symbol;
        const unsigned int symbol_idx = tid - entry * total_symbol;
        const int symbol = symbol_idx + lin_symbol;

        const float symbolf = static_cast<float>(symbol);
        unsigned int s_ub, s_lb;
        if (symbol == rin_symbol) {
            s_ub = free_weight; // TODO: probably overflow
        } else {
            auto ub = gaussian_cdf(mu[entry], sigma[entry], symbolf + 0.5);
            s_ub = static_cast<unsigned int>(ub * free_weight) ;
        }
        

        auto leaky_cdf = s_ub + symbol_idx + 1;
        cdf_grid[ entry * (total_symbol + 1) + 1 + symbol_idx] = leaky_cdf;

        // printf("tid=%d, symbol_idx=%d, symbol=%d, pmf=%x\n", tid, symbol_idx, symbol, pmf);

        tid += total_t;
    }

}

void __global__ gaussian_cdf_per_symbol(
    const float * mu, const float * sigam, 
    const int * symbols, 
    unsigned int * cdf_array,
    unsigned int precission,
    
    const unsigned int N
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_t = blockDim.x * gridDim.x;
    const unsigned int scale = 2l << (precission-1);
    while (tid < N)
    {
        auto cdf = gaussian_cdf(mu[tid], sigam[tid], static_cast<float>(symbols[tid]) + 0.5);
        cdf_array[tid] = static_cast<unsigned int>(cdf * scale);
        tid += total_t;
    }
}

void __global__ gaussian_pmf_grid(
    const float * mu,     // size N
    const float * sigam,  // size N

    unsigned int * pmf_grid,  // size N x (rin_symbol - lin_symbol + 1)
    const unsigned int free_weight,
    const int lin_symbol, // left inclusive symbol
    const int rin_symbol, // right inclusive symbol
    const unsigned long max_weight,
    const unsigned int N // number of entry
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_t = blockDim.x * gridDim.x;
    const unsigned int total_symbol = (rin_symbol - lin_symbol + 1);

    while (tid < N * total_symbol)
    {   
        const unsigned entry = tid / total_symbol;
        const unsigned int symbol_idx = tid - entry * total_symbol;
        const int symbol = symbol_idx + lin_symbol;

        const float symbolf = static_cast<float>(symbol);
        unsigned int s_ub, s_lb;
        if (symbol == rin_symbol) {
            s_ub = free_weight; // TODO: probably overflow
        } else {
            auto ub = gaussian_cdf(mu[entry], sigam[entry], symbolf + 0.5);
            s_ub = static_cast<unsigned int>(ub * free_weight) ;
        }
        
        if (symbol == lin_symbol) {
            s_lb = 0;
        } else {
            auto lb = gaussian_cdf(mu[entry], sigam[entry], symbolf - 0.5);
            s_lb = static_cast<unsigned int>(lb * free_weight);
        }
        auto pmf = s_ub - s_lb + 1;
        pmf_grid[ entry * total_symbol + symbol_idx] = pmf;

        // printf("tid=%d, symbol_idx=%d, symbol=%d, pmf=%x\n", tid, symbol_idx, symbol, pmf);

        tid += total_t;
    }
}

void __global__ gaussian_pmf_per_symbol(const float * mu, const float * sigam, 
    const int * symbols, 
    unsigned int * pmf_array,
    const unsigned int precision,
    const unsigned int N
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_t = blockDim.x * gridDim.x;
    const unsigned int scale = 2l << (precision-1);
    while (tid < N)
    {   
        const float symbolf = static_cast<float>(symbols[tid]);
        auto ub = gaussian_cdf(mu[tid], sigam[tid], symbolf + 0.5);
        const unsigned int s_ub = static_cast<unsigned int>(ub * scale);
        auto lb = gaussian_cdf(mu[tid], sigam[tid], symbolf - 0.5);
        const unsigned int s_lb = static_cast<unsigned int>(lb * scale);

        pmf_array[tid] = s_ub - s_lb;
        tid += total_t;
    }
}


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
) {
    // printf("enter gaussian_cdf_pmf_per_symbol\n");
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_t = blockDim.x * gridDim.x;
    while (tid < N)
    {   
        const int symbol = symbols[tid];
        // printf("encoding %d\n", symbol);
        const unsigned int symbol_idx = symbol - lin_symbol;
        const float symbolf = static_cast<float>(symbol);
        unsigned int s_ub, s_lb;
        if (symbol == rin_symbol) {
            s_ub = free_weight; // TODO: probably overflow
        } else {
            auto ub = gaussian_cdf(mu[tid], sigam[tid], symbolf + 0.5);
            s_ub = static_cast<unsigned int>(ub * free_weight) ;
        }
        
        if (symbol == lin_symbol) {
            s_lb = 0;
        } else {
            auto lb = gaussian_cdf(mu[tid], sigam[tid], symbolf - 0.5);
            s_lb = static_cast<unsigned int>(lb * free_weight);
        }
        // printf("s_lb %d s_ub %d\n", s_lb, s_ub);
        cdf_array[tid] = s_lb + symbol_idx;
        pmf_array[tid] = s_ub - s_lb + 1;
        tid += total_t;
    }
}

void __global__ gaussian_ppf_grid() {
    
}

std::tuple<torch::Tensor, torch::Tensor> QuantizedGaussian::cdf_pmf_per_symbol(const torch::Tensor & symbols) {
        const unsigned int N = symbols.size(0);
        // auto int_opts = symbols.options().dtype(torch::kInt32);

        torch::Device device(torch::kCUDA);
        torch::TensorOptions options(torch::kInt32);


        torch::Tensor pmf = torch::full({N}, 0, options.device(device));
        torch::Tensor cdf = torch::full({N}, 0, options.device(device));

        // std::cout <<"[cdf_pmf_per_symbol] cdf"  << cdf << std::endl;
        
        const int block_size = 16;
        // const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
        const int grid_size = (N - 1) / block_size + 1;

        // std::cout << "mu " <<  m_mu << std::endl;

        gaussian_cdf_pmf_per_symbol<<<grid_size, block_size>>>(
            m_mu.contiguous().data_ptr<float>(),
            m_sigma.contiguous().data_ptr<float>(),
            symbols.contiguous().data_ptr<int>(),
            reinterpret_cast<unsigned int*>(cdf.contiguous().data_ptr<int>()),
            reinterpret_cast<unsigned int*>(pmf.contiguous().data_ptr<int>()),
            m_free_weight,
            m_left_inclusive_symbol,
            m_right_inclusive_symbol,
            m_weight_budget,
            N
        );

        return std::make_tuple(cdf, pmf);

    }

    std::tuple<torch::Tensor, torch::Tensor> QuantizedGaussian::cdf_pmf_grid() {
        const unsigned int N = m_mu.size(0);
        const int block_size = 512;
        // const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
        const int grid_size = (N - 1) / block_size + 1;

        auto int_opts = m_mu.options().dtype(torch::kInt32);

        torch::Tensor pmf_grid = torch::full({N, m_support_size}, 0, int_opts);
        torch::Tensor cdf_grid = torch::full({N, m_support_size + 1}, 0, int_opts);



        gaussian_cdf_grid<<<grid_size, block_size>>>(
            m_mu.contiguous().data_ptr<float>(),
            m_sigma.contiguous().data_ptr<float>(),
            reinterpret_cast<unsigned int*>(cdf_grid.contiguous().data_ptr<int>()),
            m_free_weight,
            m_left_inclusive_symbol,
            m_right_inclusive_symbol,
            m_weight_budget,
            N // number of entry
        );

        // pmf 应该可以直接从cdf中计算得出，待优化
        gaussian_pmf_grid<<<grid_size, block_size>>>(
            m_mu.contiguous().data_ptr<float>(),
            m_sigma.contiguous().data_ptr<float>(),
            reinterpret_cast<unsigned int*>(pmf_grid.contiguous().data_ptr<int>()),
            m_free_weight,
            m_left_inclusive_symbol,
            m_right_inclusive_symbol,
            m_weight_budget,
            N // number of entry
        );

        return std::make_tuple(cdf_grid, pmf_grid);
    }