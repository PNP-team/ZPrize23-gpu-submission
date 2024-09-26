#include "function.cuh"
#include <cuda_runtime.h>
#include <fstream>

SyncedMemory to_mont(SyncedMemory input) {
   if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::to_mont_cpu(input);
   }
   else{
        return cuda::to_mont_cuda(input);
   }
}

SyncedMemory to_base(SyncedMemory input) {
    if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::to_base_cpu(input);
    }
    else{
        return cuda::to_base_cuda(input);
    }
}

SyncedMemory neg_mod(SyncedMemory input) {
    if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::neg_mod_cpu(input);
    }
    else{
        return cuda::neg_mod_cuda(input);
    }
}

SyncedMemory inv_mod(SyncedMemory input) {
    if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::inv_mod_cpu(input);
    }
    else{
        return cuda::inv_mod_cuda(input);
    }
}

SyncedMemory add_mod(SyncedMemory input1, SyncedMemory input2) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::add_mod_cpu(input1, input2);
    }
    else{
        return cuda::add_mod_cuda(input1, input2);
    }
}

void add_mod_(SyncedMemory input1, SyncedMemory input2) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        cpu::add_mod_cpu_(input1, input2);
    }
    else{
        cuda::add_mod_cuda_(input1, input2);
    }
}

SyncedMemory sub_mod(SyncedMemory input1, SyncedMemory input2) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::sub_mod_cpu(input1, input2);
    }
    else{
        return cuda::sub_mod_cuda(input1, input2);
    }
}

void sub_mod_(SyncedMemory input1, SyncedMemory input2) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        cpu::sub_mod_cpu_(input1, input2);
    }
    else{
        cuda::sub_mod_cuda_(input1, input2);
    }
}

SyncedMemory mul_mod(SyncedMemory input1, SyncedMemory input2) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::mul_mod_cpu(input1, input2);
    }
    else{
        return cuda::mul_mod_cuda(input1, input2);
    }
}

void mul_mod_(SyncedMemory input1, SyncedMemory input2) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        cpu::mul_mod_cpu_(input1, input2);
    }
    else{
        cuda::mul_mod_cuda_(input1, input2);
    }
}

SyncedMemory div_mod(SyncedMemory input1, SyncedMemory input2) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::div_mod_cpu(input1, input2);
    }
    else{
        return cuda::div_mod_cuda(input1, input2);
    }
}

void div_mod_(SyncedMemory input1, SyncedMemory input2) {
    if(input1.head() == SyncedMemory::HEAD_AT_CPU){
        cpu::div_mod_cpu_(input1, input2);
    }
    else{
        cuda::div_mod_cuda_(input1, input2);
    }
}

SyncedMemory exp_mod(SyncedMemory input, int exp) {
    if(input.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::exp_mod_cpu(input, exp);
    }
    else{
        return cuda::exp_mod_cuda(input, exp);
    }
}

SyncedMemory add_mod_scalar(SyncedMemory input1, SyncedMemory input2){
    return cuda::add_mod_scalar_cuda(input1, input2);
}

void add_mod_scalar_(SyncedMemory input1, SyncedMemory input2){
    cuda::add_mod_scalar_cuda_(input1, input2);
}

SyncedMemory sub_mod_scalar(SyncedMemory input1, SyncedMemory input2){
    return cuda::sub_mod_scalar_cuda(input1, input2);
}

void sub_mod_scalar_(SyncedMemory input1, SyncedMemory input2){
    cuda::sub_mod_scalar_cuda_(input1, input2);
}

SyncedMemory mul_mod_scalar(SyncedMemory input1, SyncedMemory input2){
    return cuda::mul_mod_scalar_cuda(input1, input2);
}

void mul_mod_scalar_(SyncedMemory input1, SyncedMemory input2){
    cuda::mul_mod_scalar_cuda_(input1, input2);
}

SyncedMemory div_mod_scalar(SyncedMemory input1, SyncedMemory input2){
    return cuda::div_mod_scalar_cuda(input1, input2);
}

void div_mod_scalar_(SyncedMemory input1, SyncedMemory input2){
    cuda::div_mod_scalar_cuda_(input1, input2);
}

SyncedMemory gen_sequence(uint64_t N, SyncedMemory x){
    return cuda::poly_eval_cuda(x, N);
}

SyncedMemory repeat_to_poly(SyncedMemory x, uint64_t N){
    return cuda::repeat_to_poly_cuda(x, N);
}

SyncedMemory evaluate(SyncedMemory poly, SyncedMemory x){
    if (poly.size() == 0){
        SyncedMemory result(4 * sizeof(uint64_t));
        void* res_gpu = result.mutable_gpu_data();
        cudaMemset(res_gpu, 0, 4 * sizeof(uint64_t));
        return result;
    }
    else{
        SyncedMemory y = cuda::poly_eval_cuda(x, poly.size()/ (cuda::fr_LIMBS*sizeof(uint64_t)));
        return cuda::poly_reduce_cuda(y, poly);
    }
}

SyncedMemory poly_div_poly(SyncedMemory divid, SyncedMemory c){
    return cuda::poly_div_cuda(divid,c);
}

SyncedMemory pad_poly(SyncedMemory x, uint64_t N){
    if(x.head() == SyncedMemory::HEAD_AT_CPU){
        return cpu::pad_poly_cpu(x, N);
    }
    else{
        return cuda::pad_poly_cuda(x, N);
    }
}

SyncedMemory repeat_zero(uint64_t N){
    SyncedMemory out(N * cpu::fr_LIMBS * sizeof(uint64_t));
    void* out_ = out.mutable_cpu_data();
    memset(out_, 0, out.size());
    return out;
}

SyncedMemory cat(SyncedMemory a, SyncedMemory b){
    SyncedMemory res(a.size() + b.size());
    if(a.head() == SyncedMemory::HEAD_AT_CPU){
        void* res_ = res.mutable_cpu_data();
        void* a_ = a.mutable_cpu_data();
        void* b_ = b.mutable_cpu_data();
        memcpy(res_, a_, a.size());
        memcpy(res_ + a.size(), b_, b.size());
    }
    else{
        void* res_ = res.mutable_gpu_data();
        void* a_ = a.mutable_gpu_data();
        void* b_ = b.mutable_gpu_data();
        caffe_gpu_memcpy(a.size(), a_, res_);
        caffe_gpu_memcpy(b.size(), b_, res_ + a.size());
    }
    return res;
}

SyncedMemory slice(SyncedMemory a, uint64_t len, bool forward){
    if(a.head() == SyncedMemory::HEAD_AT_CPU){
        SyncedMemory res(cpu::fr_LIMBS * len * sizeof(uint64_t));
        void* res_ = res.mutable_cpu_data();
        void* a_ = a.mutable_cpu_data();
        if(forward){
            memcpy(res_, a_, res.size());
        }
        else{
            memcpy(res_, a_ + a.size()-res.size(), res.size());
        }
        return res;
    }
    else{
        SyncedMemory res(cuda::fr_LIMBS * len * sizeof(uint64_t));
        void* res_ = res.mutable_gpu_data();
        void* a_ = a.mutable_gpu_data();
        if(forward){
            caffe_gpu_memcpy(res.size(), a_, res_);
        }
        else{
            caffe_gpu_memcpy(res.size(), a_ + a.size()-res.size(), res_);
        }
        return res;
    }
}

SyncedMemory accumulate_mul_poly(SyncedMemory product){
    return cuda::accumulate_mul_poly_cuda(product);
}

SyncedMemory make_tensor(SyncedMemory input, uint64_t pad_len){
    return cuda::make_tensor(input, pad_len);
}

Ntt::Ntt(int domain_size): Params(cuda::params_zkp_cuda(domain_size, false)) {}

SyncedMemory Ntt::forward(SyncedMemory input) {
    return cuda::ntt_zkp_cuda(input, Params, false, false);
}

Intt::Intt(int domain_size): Params(cuda::params_zkp_cuda(domain_size, true)) {}

SyncedMemory Intt::forward(SyncedMemory input) {
    return cuda::ntt_zkp_cuda(input, Params, true, false);
}

Ntt_coset::Ntt_coset(int domain_size, int coset_size)
        :Size(coset_size), Params(cuda::params_zkp_cuda(domain_size, false)) {}

SyncedMemory Ntt_coset::forward(SyncedMemory input) {
    SyncedMemory temp = cuda::pad_poly_cuda(input, Size);
    return cuda::ntt_zkp_cuda(temp, Params, false, true);
}

Intt_coset::Intt_coset(int domain_size): Params(cuda::params_zkp_cuda(domain_size, true)) {}

SyncedMemory Intt_coset::forward(SyncedMemory input) {
    return cuda::ntt_zkp_cuda(input, Params, true, true);
}

SyncedMemory multi_scalar_mult(SyncedMemory points, SyncedMemory scalars){

    int64_t point_num = points.size()/(sizeof(uint64_t) * cpu::fq_LIMBS);
    int64_t scalar_num = scalars.size()/(sizeof(uint64_t) * cpu::fr_LIMBS);
    int64_t msm_size = std::min(point_num, scalar_num);
    int device;
    cudaError_t err = cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, device);
    int smcount = deviceProp.multiProcessorCount;

    SyncedMemory step1_res = cuda::msm_zkp_cuda(points, scalars, smcount);
    SyncedMemory step2_res = cpu::msm_collect_cpu(step1_res, msm_size);
    
    return step2_res;
}

void writeToFile(const std::string& filename, uint64_t* array, size_t size) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // 写入数组到文本文件，每个元素占一行
    for (size_t i = 0; i < size; ++i) {
        outfile << array[i] << std::endl;
    }

    if (!outfile) {
        std::cerr << "Error writing to file: " << filename << std::endl;
    }

    outfile.close();
}