#include <iostream>
#include <functional>
#include "../../../structure.cuh"
#include "constants.cuh"
#include "../../../domain.cuh"

SyncedMemory compute_permutation_poly(Radix2EvaluationDomain domain, 
                                       std::vector<SyncedMemory> wires,
                                       SyncedMemory beta, SyncedMemory gamma,
                                       std::vector<SyncedMemory> sigma_polys);

SyncedMemory compute_lookup_permutation_poly(uint64_t n,
                                              SyncedMemory f, 
                                              SyncedMemory t, 
                                              SyncedMemory h_1, 
                                              SyncedMemory h_2, 
                                              SyncedMemory delta, 
                                              SyncedMemory epsilon);