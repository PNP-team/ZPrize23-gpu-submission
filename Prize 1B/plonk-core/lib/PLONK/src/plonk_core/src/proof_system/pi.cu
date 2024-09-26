#include "pi.cuh"

SyncedMemory as_evals(SyncedMemory public_inputs, uint64_t pi_pos, uint64_t n) {
    SyncedMemory pi = repeat_zero(n);
    void* pi_gpu_data = pi.mutable_gpu_data();
    void* public_inputs_data = public_inputs.mutable_gpu_data();
    caffe_gpu_memcpy(public_inputs.size(), public_inputs_data, pi_gpu_data + pi_pos*sizeof(uint64_t)*fr::Limbs);
    return pi;
}

SyncedMemory into_dense_poly(SyncedMemory public_inputs, uint64_t pi_pos, uint64_t n, Intt INTT) {
    SyncedMemory field_pi = to_mont(public_inputs);
    SyncedMemory evals_tensor = as_evals(field_pi, pi_pos, n);
    SyncedMemory pi_coeffs = INTT.forward(evals_tensor);
    return pi_coeffs;
}