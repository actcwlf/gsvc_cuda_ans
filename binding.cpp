#include <ATen/ops/from_blob.h>
#include <c10/core/DeviceType.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/types.h>
#include <vector>
#include "torch/extension.h"
#include "simple_ans.cuh"

namespace py = pybind11;

template <typename T>
std::vector<T> from_numpy(const py::array_t<T> arr ) {
    auto buf = arr.request();
    auto ptr = static_cast<T *>(buf.ptr);
    std::vector<T> vec;
    vec.reserve(buf.shape[0]);
    for (size_t idx = 0; idx < buf.shape[0]; idx++){
        vec.push_back(ptr[idx]);
    }

    return vec;
}


template <typename T>
py::array_t<T> to_numpy(const std::vector<T> & v)
{
  //std::cout << "1d to_numpy, size " << v.size() << std::endl;;
  auto result = py::array_t<T>(v.size());
  T* buf = (T*)result.request().ptr;
  for (size_t i = 0; i < v.size(); ++i)
  {
    buf[i] = v[i];
  }
  return py::array_t<T>(std::move(result));
}


template <typename T>
py::array_t<T> to_numpy(const torch::Tensor & v)
{
  //std::cout << "1d to_numpy, size " << v.size() << std::endl;;
  auto result = py::array_t<T>(v.size(0));
  T* buf = (T*)result.request().ptr;
  for (size_t i = 0; i < v.size(0); ++i)
  {
    auto t = v.contiguous().data_ptr<T>();
    buf[i] = t[i];
  }
  return py::array_t<T>(std::move(result));
}


struct CudaAnsCoder {
    int m_left_inclusive_symbol;
    int m_right_inclusive_symbol;
    unsigned int m_precision;
    CudaAnsCoder(int left_inclusive_symbol, int right_inclusive_symbol, unsigned int precision):
        m_left_inclusive_symbol(left_inclusive_symbol),
        m_right_inclusive_symbol(right_inclusive_symbol),
        m_precision(precision) {}

    py::array_t<unsigned int> encode_reverse_wrapper(py::array_t<int> & message, const py::array_t<float> & mu, const py::array_t<float> & sigma) {

        auto c_msg = from_numpy<int>(message);
        auto c_mu = from_numpy<float>(mu);
        auto c_sigma = from_numpy<float>(sigma);
        
        auto bs = encode(c_mu, c_sigma, c_msg, m_left_inclusive_symbol, m_right_inclusive_symbol, m_precision);
        return to_numpy<unsigned int>(bs);

    }


    py::array_t<int> decode_wrapper(py::array_t<unsigned int> & bitstream, const py::array_t<float> & mu, const py::array_t<float> & sigma) {

        auto bs = torch::from_blob(reinterpret_cast<int *>(bitstream.request().ptr), {bitstream.size()}, torch::kInt32).to(torch::kCUDA);
        auto c_mu = torch::from_blob(reinterpret_cast<float *>(mu.request().ptr), {mu.size()}, torch::kFloat32).to(torch::kCUDA);
        
        auto c_sigma = torch::from_blob(reinterpret_cast<float *>(sigma.request().ptr), {sigma.size()}, torch::kFloat32).to(torch::kCUDA);
        
        auto decoded_msg = decode(c_mu, c_sigma, bs, m_left_inclusive_symbol, m_right_inclusive_symbol, m_precision);
        auto np_result = to_numpy<int>(decoded_msg.to(torch::kCPU));
        return np_result;
    }
    
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CudaAnsCoder>(m, "CudaAnsCoder")
    .def(py::init<int, int, unsigned int>())
    .def("encode_reverse", &CudaAnsCoder::encode_reverse_wrapper, "Encode Msg")
    .def("decode", &CudaAnsCoder::decode_wrapper);
}