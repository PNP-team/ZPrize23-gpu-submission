#include "function.cuh"

bool gt_zkp(SyncedMemory a, SyncedMemory b){
    int64_t numel = a.size()/sizeof(uint64_t);
    uint64_t* a_ = reinterpret_cast<uint64_t*>(a.mutable_cpu_data());
    uint64_t* b_ = reinterpret_cast<uint64_t*>(b.mutable_cpu_data());
    bool gt = false;
    for(int64_t i = numel-1; i >= 0; i--){
        if(a_[i] > b_[i]){
            gt = true;
            break;
        }
        else if(a_[i] < b_[i]){
            gt = false;
            break;
        }
        else{
            continue;
        }
    }
    return gt;
}

SyncedMemory compress(SyncedMemory t_0, SyncedMemory t_1, SyncedMemory t_2, SyncedMemory t_3, 
                       SyncedMemory challenge){
    return cuda::compress_cuda(t_0, t_1, t_2, t_3, challenge);
}

SyncedMemory compute_query_table(SyncedMemory q_lookup, 
            SyncedMemory w_l_scalar, SyncedMemory w_r_scalar, SyncedMemory w_o_scalar,
            SyncedMemory w_4_scalar, SyncedMemory t_poly, SyncedMemory challenge){
    int64_t n = w_l_scalar.size() / (cuda::fr_LIMBS*sizeof(uint64_t));
    SyncedMemory padded_q_lookup = cuda::pad_poly_cuda(q_lookup, n);
    SyncedMemory concatenated_f_scalars = cuda::compute_query_table_cuda(padded_q_lookup, w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar, t_poly);
    return cuda::compress_cuda_2(concatenated_f_scalars, challenge, n * cuda::fr_LIMBS);  
}

std::vector<SyncedMemory> split_tx_poly(uint64_t n, SyncedMemory t_poly){
    std::vector<SyncedMemory> t_x;
    void* t_gpu = t_poly.mutable_gpu_data();
    for(int i=0; i<8; i++){
        SyncedMemory t_(n*cuda::fr_LIMBS*sizeof(uint64_t));
        void* t_x_gpu = t_.mutable_gpu_data();
        caffe_gpu_memcpy(t_.size(), t_gpu + i*t_.size(), t_x_gpu);
        t_x.push_back(t_);
    }
    return t_x;
}

