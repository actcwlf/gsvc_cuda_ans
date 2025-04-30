#include <algorithm>
#include <torch/extension.h>
#include <cooperative_groups.h>
// #include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>

#include "gaussian.cuh"

namespace cg = cooperative_groups;


const double EPSILON = 1.0e-15;
const double value_a = 1.23;
const double value_b = 2.34;
const double value_c = 3.57;


using word_t = unsigned int;
using symbol_t = unsigned int;



void __global__ encode_symbols(
    // const symbol_t* symbols,
    const unsigned int * pmf_symbols,
    const unsigned int * cdf_symbols,
    unsigned int * bitstream ,
    int * header,
    const unsigned int precision,
    const int N
) {
    const int tid = threadIdx.x;

    if (tid != 0) {
        return;
    }
    // unsigned int m_mask;
    unsigned long m_head = 0;
    // const unsigned int m_precision = 32;
    const unsigned int m_mask = static_cast<unsigned int>((1l << precision) - 1);
    int cursor = -1;
    for (int i = 0; i < N; i++) {
        const unsigned int pmf_symbol = pmf_symbols[i];
        const unsigned int cdf_symbol = cdf_symbols[i];
        if((m_head >> precision) >= pmf_symbol) {
            cursor += 1;
            bitstream[cursor] = static_cast<word_t>(m_head & m_mask);
            
            m_head = m_head >> precision;
        }

        auto z = m_head %  pmf_symbol + cdf_symbol;
        // printf("cuda z=%lx, pmf=%d, cdf=%d\n", z, pmf_symbol, cdf_symbol);
        m_head = m_head / pmf_symbol;
        m_head = m_head << precision | z;
    }

    while (m_head > 0) {
        cursor += 1;
        bitstream[cursor] = static_cast<word_t>(m_head & m_mask);
        
        m_head >>= precision;
    }
    header[0] = cursor;
}


void __global__ decode_symbols(
    const unsigned int * bitstream ,
    const unsigned int * cdf_grid, // N * len_pmf, where len_pmf = right_inclusive_symbol - left_inclusive_symbol + 1
    const unsigned int * pmf_grid, // N * len_pmf, where len_pmf = right_inclusive_symbol - left_inclusive_symbol + 1
    unsigned int * symbols,
    unsigned int support_size,
    const unsigned int precision,
    int cursor,
    unsigned int N
) {

    auto block = cg::this_thread_block();
    int tid = threadIdx.x;
    const int total_t = blockDim.x * gridDim.x;

    unsigned long m_head = 0;
    // const unsigned int m_precision = 32;
    const unsigned int m_mask = static_cast<unsigned int>((1l << precision) - 1);
    // int cursor = header[0];

    if (tid >= support_size) {
        return;
    }

    while (cursor >= 0  && (m_head >> precision == 0))
    {
        auto token = bitstream[cursor];
        cursor -= 1;
        m_head = m_head << precision | token;
    }

    __shared__ int current_decoded[1];
    __shared__ int residual_z[1];
    // if(tid == 0) {
    //     printf("total bs %d", N);
    // }

    bool done = false;
    int decoded_num = 0;
    for(int i = 0; i < N; i++) {
        // if (tid == 0) {
        //     printf("decoding %d\n", i);
        // }
        const unsigned int * cdf = cdf_grid + (i * (support_size + 1));
        const unsigned int * pmf = pmf_grid + (i * support_size);
        auto z = m_head & m_mask;
        m_head = m_head >> precision;
        bool succeed = false;
        int real_tid = tid;
        while(tid < support_size) {
            auto lb = cdf[tid];
            auto ub = cdf[tid+1];
            if ( (z >= lb) && (z < ub)) {
                current_decoded[0] = tid;
                residual_z[0] = z - lb;
                succeed = true; // succeed
                symbols[i] = tid;
                // printf("curr decoded %d, residual z %ld, m_head %ld\n", tid, z, m_head);
                break;
            } else {
                if (z < lb) {
                    break;
                }
                tid += total_t;
            }
        }
        tid = real_tid;


        block.sync();

        unsigned int symbol = current_decoded[0];
        unsigned int m_symbol = pmf[symbol];

        m_head = m_head * m_symbol + residual_z[0];
        if (succeed) {
            // printf("after m_head %ld\n", m_head);
            succeed = false;
        }
        if (((m_head >> precision) == 0) && (cursor >= 0)){
            auto token = bitstream[cursor];
            cursor -= 1;
            m_head = m_head << precision | token;
        }
        

        
    }
}



void __global__ add(const unsigned int * symbols, int * new_symbols, const int lin_symbol, const unsigned int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_t = blockDim.x * gridDim.x;
    while (tid < N)
    {
        new_symbols[tid] = symbols[tid] + lin_symbol;
        tid += total_t;
    }
}


std::vector<unsigned int> encode(
    const std::vector<float> mu,
    const std::vector<float> sigma,
    const std::vector<int> symbols,
    const int left_inclusive_symbol,
    const int right_inclusive_symbol,
    const unsigned int precision
    ) {

    const unsigned int support_size = right_inclusive_symbol - left_inclusive_symbol + 1;
    const unsigned long weight_budget = 1l << precision;
    const unsigned int N = symbols.size();
    const unsigned int free_weight = weight_budget - support_size;


    unsigned int *h_bs;

    float * d_mu;
    float * d_sigma;
    int * d_symbols;
    unsigned int * d_pmf;
    unsigned int * d_cdf;

    unsigned int *d_bs;
    int * d_header;


    cudaMallocHost(&h_bs, N * sizeof(unsigned int));

    cudaMalloc(&d_mu, N * sizeof(float));
    cudaMalloc(&d_sigma, N * sizeof(float));
    cudaMalloc(&d_symbols, N * sizeof(int));
    cudaMalloc(&d_pmf, N * sizeof(unsigned int));
    cudaMalloc(&d_cdf, N * sizeof(unsigned int));

    cudaMalloc(&d_bs,    N * sizeof(unsigned int));
    cudaMalloc(&d_header, sizeof(int));


    cudaMemcpy(d_mu, mu.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, sigma.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_symbols, symbols.data(), N * sizeof(int), cudaMemcpyHostToDevice);


    const int block_size = 512;
    // const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    const int grid_size = (N - 1) / block_size + 1;


    gaussian_cdf_pmf_per_symbol<<<grid_size, block_size>>>(
        d_mu,
        d_sigma,
        d_symbols,
        d_cdf,
        d_pmf,
        free_weight,
        left_inclusive_symbol,
        right_inclusive_symbol,
        weight_budget,
        N
    );

    // cudaDeviceSynchronize();

    encode_symbols<<<1, 1>>>(
        d_pmf,
        d_cdf,
        d_bs,
        d_header,
        precision,
        N
    );

    unsigned int bs_len;

    cudaMemcpy(&bs_len, d_header, sizeof(int), cudaMemcpyDeviceToHost); 

    bs_len = bs_len + 1;

    cudaMemcpy(h_bs, d_bs, bs_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);


    std::vector<unsigned int> bitstream(h_bs , h_bs + bs_len);

    cudaFreeHost(h_bs);
    cudaFree(d_mu);
    cudaFree(d_sigma);
    cudaFree(d_symbols);
    cudaFree(d_pmf);
    cudaFree(d_cdf);
    cudaFree(d_bs);
    cudaFree(d_header);

    return bitstream;

}


torch::Tensor decode(
    const torch::Tensor& mu,
    const torch::Tensor& sigma,
    const torch::Tensor& bitstream,

    const int left_inclusive_symbol,
    const int right_inclusive_symbol,
    const unsigned int precision
    
    ) {

    const unsigned int support_size = right_inclusive_symbol - left_inclusive_symbol + 1;
    const unsigned long weight_budget = 1l << precision;
    const unsigned int N = mu.size(0);
    const unsigned int free_weight = weight_budget - support_size;

    auto int_opts = mu.options().dtype(torch::kInt32);
    auto float_opts = mu.options().dtype(torch::kFloat32);
    // auto uint_opts = mu.options().dtype(torch::);

    torch::Tensor symbols = torch::full({N}, 0, int_opts);
    torch::Tensor pmf_grid = torch::full({N, support_size}, 0, int_opts);
    torch::Tensor cdf_grid = torch::full({N, support_size + 1}, 0, int_opts);





    const int block_size = 512;
    // const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    const int grid_size = (N - 1) / block_size + 1;


    gaussian_pmf_grid<<<grid_size, block_size>>>(
        mu.contiguous().data_ptr<float>(),
        sigma.contiguous().data_ptr<float>(),
        reinterpret_cast<unsigned int*>(pmf_grid.contiguous().data_ptr<int>()),
        free_weight,
        left_inclusive_symbol,
        right_inclusive_symbol,
        weight_budget,
        N // number of entry
    );

    gaussian_cdf_grid<<<grid_size, block_size>>>(
        mu.contiguous().data_ptr<float>(),
        sigma.contiguous().data_ptr<float>(),
        reinterpret_cast<unsigned int*>(cdf_grid.contiguous().data_ptr<int>()),
        free_weight,
        left_inclusive_symbol,
        right_inclusive_symbol,
        weight_budget,
        N // number of entry
    );

    decode_symbols<<<1, 512>>>(
        reinterpret_cast<unsigned int*>(bitstream.contiguous().data_ptr<int>()),
        reinterpret_cast<unsigned int*>(cdf_grid.contiguous().data_ptr<int>()),
        reinterpret_cast<unsigned int*>(pmf_grid.contiguous().data_ptr<int>()),
        reinterpret_cast<unsigned int*>(symbols.contiguous().data_ptr<int>()),
        support_size,
        precision,
        bitstream.size(0) - 1,
        N
    );

    symbols = symbols + left_inclusive_symbol;

    return symbols;
}

std::tuple<torch::Tensor, torch::Tensor> calc_grid(
    const torch::Tensor& mu,
    const torch::Tensor& sigma,

    const int left_inclusive_symbol,
    const int right_inclusive_symbol,
    const unsigned int precision
    
    ) {

    const unsigned int support_size = right_inclusive_symbol - left_inclusive_symbol + 1;
    const unsigned long weight_budget = 1l << precision;
    const unsigned int N = mu.size(0);
    const unsigned int free_weight = weight_budget - support_size;

    auto int_opts = mu.options().dtype(torch::kInt32);
    auto float_opts = mu.options().dtype(torch::kFloat32);
    // auto uint_opts = mu.options().dtype(torch::);

    torch::Tensor symbols = torch::full({N}, 0, int_opts);
    torch::Tensor pmf_grid = torch::full({N, support_size}, 0, int_opts);
    torch::Tensor cdf_grid = torch::full({N, support_size + 1}, 0, int_opts);





    const int block_size = 512;
    // const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    const int grid_size = (N - 1) / block_size + 1;


    gaussian_pmf_grid<<<grid_size, block_size>>>(
        mu.contiguous().data_ptr<float>(),
        sigma.contiguous().data_ptr<float>(),
        reinterpret_cast<unsigned int*>(pmf_grid.contiguous().data_ptr<int>()),
        free_weight,
        left_inclusive_symbol,
        right_inclusive_symbol,
        weight_budget,
        N // number of entry
    );

    gaussian_cdf_grid<<<grid_size, block_size>>>(
        mu.contiguous().data_ptr<float>(),
        sigma.contiguous().data_ptr<float>(),
        reinterpret_cast<unsigned int*>(cdf_grid.contiguous().data_ptr<int>()),
        free_weight,
        left_inclusive_symbol,
        right_inclusive_symbol,
        weight_budget,
        N // number of entry
    );


    return std::make_tuple(pmf_grid, cdf_grid);
}
