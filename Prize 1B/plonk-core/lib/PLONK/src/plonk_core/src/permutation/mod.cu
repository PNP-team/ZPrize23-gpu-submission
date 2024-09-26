#include "mod.cuh"

SyncedMemory _numerator_irreducible(SyncedMemory root, SyncedMemory w, SyncedMemory k, SyncedMemory beta, SyncedMemory gamma) {
    SyncedMemory mid1 = mul_mod(beta, k); 
    SyncedMemory mid2 = mul_mod(mid1, root); 
    SyncedMemory mid3 = add_mod(w, mid2); 
    SyncedMemory numerator = add_mod(mid3, gamma);
    return numerator; 
}

SyncedMemory _denominator_irreducible(SyncedMemory w, SyncedMemory sigma, SyncedMemory beta, SyncedMemory gamma) {
    SyncedMemory mid1 = mul_mod_scalar(sigma, beta); 
    SyncedMemory mid2 = add_mod(w, mid1);
    SyncedMemory denominator = add_mod(mid2, gamma); 
    return denominator; 
}

SyncedMemory _lookup_ratio(SyncedMemory one, SyncedMemory delta, SyncedMemory epsilon, SyncedMemory f, SyncedMemory t, SyncedMemory t_next,
                  SyncedMemory h_1, SyncedMemory h_1_next, SyncedMemory h_2) {

    SyncedMemory one_plus_delta = add_mod(delta, one); 
    SyncedMemory epsilon_one_plus_delta = mul_mod(epsilon, one_plus_delta); 

    SyncedMemory mid1 = add_mod(epsilon, f); 
    SyncedMemory mid2 = add_mod(epsilon_one_plus_delta, t); 
    SyncedMemory mid3 = mul_mod(delta, t_next); 
    SyncedMemory mid4 = add_mod(mid2, mid3); 
    SyncedMemory mid5 = mul_mod(one_plus_delta, mid1); 
    SyncedMemory result = mul_mod(mid4, mid5); 

    SyncedMemory mid6 = mul_mod(h_2, delta); 
    SyncedMemory mid7 = add_mod(epsilon_one_plus_delta, h_1); 
    SyncedMemory mid8 = add_mod(mid6, mid7); 
    SyncedMemory mid9 = add_mod(epsilon_one_plus_delta, h_2); 
    SyncedMemory mid10 = mul_mod(h_1_next, delta); 
    SyncedMemory mid11 = add_mod(mid9, mid10); 
    SyncedMemory mid12 = mul_mod(mid8, mid11); 
    SyncedMemory mid13 = div_mod(one, mid12); 
    
    mul_mod_(result, mid13); 
    return result; 
}

SyncedMemory compute_permutation_poly(Radix2EvaluationDomain domain, std::vector<SyncedMemory> wires, SyncedMemory beta, SyncedMemory gamma, std::vector<SyncedMemory> sigma_polys) {
    uint64_t n = domain.size;
    SyncedMemory one = fr::one();
    void* one_gpu_data = one.mutable_gpu_data();
    // Constants defining cosets H, k1H, k2H, etc
    std::vector<SyncedMemory> ks;
    SyncedMemory obj1 = fr::one();
    SyncedMemory obj2 = K1();
    SyncedMemory obj3 = K2();
    SyncedMemory obj4 = K3();
    void* obj1_gpu_data = obj1.mutable_gpu_data();
    void* obj2_gpu_data = obj2.mutable_gpu_data();
    void* obj3_gpu_data = obj3.mutable_gpu_data();
    void* obj4_gpu_data = obj4.mutable_gpu_data();

    ks.push_back(obj1);
    ks.push_back(obj2);
    ks.push_back(obj3);
    ks.push_back(obj4);

    Ntt NTT(fr::TWO_ADICITY);
    SyncedMemory sigma_mappings0 = NTT.forward(sigma_polys[0]);
    SyncedMemory sigma_mappings1 = NTT.forward(sigma_polys[1]);
    SyncedMemory sigma_mappings2 = NTT.forward(sigma_polys[2]);
    SyncedMemory sigma_mappings3 = NTT.forward(sigma_polys[3]);
  
    std::vector<SyncedMemory> sigma_mappings;
    sigma_mappings.push_back(sigma_mappings0);
    sigma_mappings.push_back(sigma_mappings1);
    sigma_mappings.push_back(sigma_mappings2);
    sigma_mappings.push_back(sigma_mappings3);
    /*
      Transpose wires and sigma values to get "rows" in the form [wl_i,
      wr_i, wo_i, ... ] where each row contains the wire and sigma
      values for a single gate
     Compute all roots, same as calculating twiddles, but doubled in size
    */
    SyncedMemory group_gen(domain.group_gen.size());
    void* domain_group_gen_cpu = domain.group_gen.mutable_cpu_data();
    caffe_gpu_memcpy(group_gen.size(), domain_group_gen_cpu, group_gen.mutable_gpu_data());
    
    SyncedMemory roots = gen_sequence(n, group_gen);

    SyncedMemory numerator_product = repeat_to_poly(one, n);
    SyncedMemory denominator_product = repeat_to_poly(one, n);

    SyncedMemory extend_beta = repeat_to_poly(beta, n);
    SyncedMemory extend_gamma = repeat_to_poly(gamma, n);
    SyncedMemory extend_one = repeat_to_poly(one, n);

    for (int index = 0; index < ks.size(); index++) {
        SyncedMemory extend_ks = repeat_to_poly(ks[index], n);
        SyncedMemory numerator_temps = _numerator_irreducible(roots, wires[index], extend_ks, extend_beta, extend_gamma);
        mul_mod_(numerator_product, numerator_temps);
        SyncedMemory denominator_temps = _denominator_irreducible(wires[index], sigma_mappings[index], beta, extend_gamma);
        mul_mod_(denominator_product, denominator_temps);
    }

    SyncedMemory denominator_product_under = div_mod(extend_one, denominator_product);
    SyncedMemory gate_coefficient = mul_mod(numerator_product, denominator_product_under);

    SyncedMemory z = accumulate_mul_poly(gate_coefficient);
    Intt INTT(fr::TWO_ADICITY);
    SyncedMemory z_poly = INTT.forward(z);
    return z_poly;
}

SyncedMemory compute_lookup_permutation_poly(uint64_t n, SyncedMemory f, SyncedMemory t, SyncedMemory h_1, SyncedMemory h_2, SyncedMemory delta, SyncedMemory epsilon) {

    assert(f.size()/(sizeof(uint64_t)*fr::Limbs) == n);
    assert(t.size()/(sizeof(uint64_t)*fr::Limbs) == n);
    assert(h_1.size()/(sizeof(uint64_t)*fr::Limbs) == n);
    assert(h_2.size()/(sizeof(uint64_t)*fr::Limbs) == n);

    SyncedMemory t_next = repeat_zero(n);
    void* t_next_gpu = t_next.mutable_gpu_data();
    void* t_gpu = t.mutable_gpu_data();
    caffe_gpu_memcpy(t.size() - sizeof(uint64_t), t_gpu + sizeof(uint64_t), t_next_gpu); //t_next[:n-1]=t[1:]
    caffe_gpu_memcpy(sizeof(uint64_t), t_gpu, t_next_gpu + t.size() - sizeof(uint64_t)); //t_next[-1]=t[0]

    SyncedMemory h_1_next = repeat_zero(n);
    void* h_1_next_gpu_data = h_1_next.mutable_gpu_data();
    void* h_1_gpu_data = h_1.mutable_gpu_data();
    caffe_gpu_memcpy(h_1.size() - sizeof(uint64_t), h_1_gpu_data + sizeof(uint64_t), h_1_next_gpu_data); //h_1_next[:n-1]=h[1:]
    caffe_gpu_memcpy(sizeof(uint64_t), h_1_gpu_data, h_1_next_gpu_data + h_1.size() - sizeof(uint64_t)); //h_1_next[-1]=h_1[0]

    SyncedMemory one = fr::one();
    void* one_gpu_data=one.mutable_gpu_data();

    SyncedMemory extend_one = repeat_to_poly(one, n);
    SyncedMemory extend_delta = repeat_to_poly(delta, n);
    SyncedMemory extend_epsilon = repeat_to_poly(epsilon, n);

    SyncedMemory product_arguments = _lookup_ratio(extend_one, extend_delta, extend_epsilon, f, t, t_next, h_1, h_1_next, h_2);
    
    SyncedMemory p = accumulate_mul_poly(product_arguments);
    Intt INTT{fr::TWO_ADICITY};
    SyncedMemory p_poly = INTT.forward(p);

    return p_poly;
}