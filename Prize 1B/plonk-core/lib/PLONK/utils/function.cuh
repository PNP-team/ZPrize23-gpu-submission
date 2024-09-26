#pragma once
#include "../../caffe/interface.hpp"
#include "mont/cuda/curve_def.cuh"
#include "mont/cpu/mont_arithmetic.h"
#include "mont/cuda/mont_arithmetic.cuh"
#include "zkp/cpu/msmcollect.hpp"
#include "zkp/cuda/zksnark.cuh"
#include <iostream>
#include <fstream>

SyncedMemory to_mont(SyncedMemory input);

SyncedMemory to_base(SyncedMemory input);

SyncedMemory neg_mod(SyncedMemory input);

SyncedMemory inv_mod(SyncedMemory input);

SyncedMemory add_mod(SyncedMemory input1, SyncedMemory input2);
void add_mod_(SyncedMemory input1, SyncedMemory input2);

SyncedMemory sub_mod(SyncedMemory input1, SyncedMemory input2);
void sub_mod_(SyncedMemory input1, SyncedMemory input2);

SyncedMemory mul_mod(SyncedMemory input1, SyncedMemory input2);
void mul_mod_(SyncedMemory input1, SyncedMemory input2);

SyncedMemory div_mod(SyncedMemory input1, SyncedMemory input2);
void div_mod_(SyncedMemory input1, SyncedMemory input2);

SyncedMemory exp_mod(SyncedMemory input, int exp);

SyncedMemory add_mod_scalar(SyncedMemory input1, SyncedMemory input2);
void add_mod_scalar_(SyncedMemory input1, SyncedMemory input2);

SyncedMemory sub_mod_scalar(SyncedMemory input1, SyncedMemory input2);
void sub_mod_scalar_(SyncedMemory input1, SyncedMemory input2);

SyncedMemory mul_mod_scalar(SyncedMemory input1, SyncedMemory input2);
void mul_mod_scalar_(SyncedMemory input1, SyncedMemory input2);

SyncedMemory div_mod_scalar(SyncedMemory input1, SyncedMemory input2);
void div_mod_scalar_(SyncedMemory input1, SyncedMemory input2);

SyncedMemory gen_sequence(uint64_t N, SyncedMemory x);

SyncedMemory repeat_to_poly(SyncedMemory x, uint64_t N);

SyncedMemory evaluate(SyncedMemory poly, SyncedMemory x);

SyncedMemory poly_div_poly(SyncedMemory divid, SyncedMemory c);

SyncedMemory pad_poly(SyncedMemory x, uint64_t N);

SyncedMemory repeat_zero(uint64_t N);

SyncedMemory cat(SyncedMemory a, SyncedMemory b);

SyncedMemory slice(SyncedMemory a, uint64_t len, bool forward);

std::vector<SyncedMemory> split_tx_poly(uint64_t n, SyncedMemory t_poly);

SyncedMemory accumulate_mul_poly(SyncedMemory product);

SyncedMemory make_tensor(SyncedMemory input, uint64_t pad_len);

class Ntt{
public:
    SyncedMemory Params;

    Ntt(int domain_size);

    SyncedMemory forward(SyncedMemory input);
};

class Intt {
public:
    SyncedMemory Params;

    Intt(int domain_size);

    SyncedMemory forward(SyncedMemory input);
};

class Ntt_coset {
public:
    SyncedMemory Params;
    int Size;

    Ntt_coset(int domain_size, int coset_size);

    SyncedMemory forward(SyncedMemory input);
};

class Intt_coset {
public:
    SyncedMemory Params;

    Intt_coset(int domain_size);

    SyncedMemory forward(SyncedMemory input);
};

SyncedMemory multi_scalar_mult(SyncedMemory points, SyncedMemory scalars);

bool gt_zkp(SyncedMemory a, SyncedMemory b);

SyncedMemory compress(SyncedMemory t_0, SyncedMemory t_1, SyncedMemory t_2, SyncedMemory t_3, 
                       SyncedMemory challenge);

SyncedMemory compute_query_table(SyncedMemory q_lookup, 
                        SyncedMemory w_l_scalar ,SyncedMemory w_r_scalar , SyncedMemory w_o_scalar,
                        SyncedMemory w_4_scalar, SyncedMemory t_poly, SyncedMemory challenge);

void writeToFile(const std::string& filename, uint64_t* array, size_t size);