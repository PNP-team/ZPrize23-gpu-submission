#pragma once
#include "../../../bls12_381/fr.cuh"
#include "../../../../utils/function.cuh"

SyncedMemory into_dense_poly(SyncedMemory public_inputs, uint64_t pi_pos, uint64_t n, Intt INTT);