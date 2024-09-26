#include "structure.cuh"
#include "bls12_381/fr.cuh"

LookupTable::LookupTable(SyncedMemory ql, SyncedMemory t1, SyncedMemory t2, SyncedMemory t3, SyncedMemory t4)
    : q_lookup(ql), table1(t1), table2(t2), table3(t3), table4(t4) {}

Arithmetic::Arithmetic(SyncedMemory qm, SyncedMemory ql, SyncedMemory qr,
               SyncedMemory qo, SyncedMemory q4, SyncedMemory qc,
               SyncedMemory qhl, SyncedMemory qhr, SyncedMemory qh4,
               SyncedMemory qarith)
        : q_m(qm), q_l(ql), q_r(qr), q_o(qo), q_4(q4),
          q_c(qc), q_hl(qhl), q_hr(qhr), q_h4(qh4), q_arith(qarith) {}

Permutation::Permutation(SyncedMemory ls, SyncedMemory rs, SyncedMemory os, SyncedMemory fs)
        : left_sigma(ls), right_sigma(rs), out_sigma(os), fourth_sigma(fs) {}

Selectors::Selectors(SyncedMemory rs, SyncedMemory ls, SyncedMemory fs, SyncedMemory vs)
        : range_selector(rs), logic_selector(ls),
          fixed_group_add_selector(fs), variable_group_add_selector(vs) {}

ProverKey::ProverKey(
        SyncedMemory q_m_coeffs, SyncedMemory q_m_evals,
        SyncedMemory q_l_coeffs, SyncedMemory q_l_evals,
        SyncedMemory q_r_coeffs, SyncedMemory q_r_evals,
        SyncedMemory q_o_coeffs, SyncedMemory q_o_evals,
        SyncedMemory q_4_coeffs, SyncedMemory q_4_evals,
        SyncedMemory q_c_coeffs, SyncedMemory q_c_evals,
        SyncedMemory q_hl_coeffs, SyncedMemory q_hl_evals,
        SyncedMemory q_hr_coeffs, SyncedMemory q_hr_evals,
        SyncedMemory q_h4_coeffs, SyncedMemory q_h4_evals,
        SyncedMemory q_arith_coeffs, SyncedMemory q_arith_evals,
        SyncedMemory range_selector_coeffs, SyncedMemory range_selector_evals,
        SyncedMemory logic_selector_coeffs, SyncedMemory logic_selector_evals,
        SyncedMemory fixed_group_add_selector_coeffs, SyncedMemory fixed_group_add_selector_evals,
        SyncedMemory variable_group_add_selector_coeffs, SyncedMemory variable_group_add_selector_evals,
        SyncedMemory q_lookup_coeffs, SyncedMemory q_lookup_evals,
        SyncedMemory table1, SyncedMemory table2, SyncedMemory table3, SyncedMemory table4,
        SyncedMemory left_sigma_coeffs, SyncedMemory left_sigma_evals,
        SyncedMemory right_sigma_coeffs, SyncedMemory right_sigma_evals,
        SyncedMemory out_sigma_coeffs, SyncedMemory out_sigma_evals,
        SyncedMemory fourth_sigma_coeffs, SyncedMemory fourth_sigma_evals,
        SyncedMemory linear_evaluations,
        SyncedMemory v_h_coset_8n) : 
        arithmetic_coeffs(Arithmetic(q_m_coeffs, q_l_coeffs, q_r_coeffs, q_o_coeffs, 
                                     q_4_coeffs, q_c_coeffs, q_hl_coeffs, q_hr_coeffs, q_h4_coeffs, q_arith_coeffs)),
        arithmetic_evals(Arithmetic(q_m_evals, q_l_evals, q_r_evals, q_o_evals, 
                                    q_4_evals, q_c_evals, q_hl_evals, q_hr_evals, q_h4_evals, q_arith_evals)),
        selectors_coeffs(Selectors(range_selector_coeffs, logic_selector_coeffs,
                                   fixed_group_add_selector_coeffs, variable_group_add_selector_coeffs)),
        selectors_evals(Selectors(range_selector_evals, logic_selector_evals,
                                   fixed_group_add_selector_evals, variable_group_add_selector_evals)),
        lookup_coeffs(LookupTable(q_lookup_coeffs, table1, table2, table3, table4)), lookup_evals(q_lookup_evals),
        permutation_coeffs(Permutation(left_sigma_coeffs, right_sigma_coeffs, out_sigma_coeffs, fourth_sigma_coeffs)),
        permutation_evals(Permutation(left_sigma_evals, right_sigma_evals, out_sigma_evals, fourth_sigma_evals)),
        linear_evaluations(linear_evaluations),
        v_h_coset_8n(v_h_coset_8n) {}

ProverKey load_pk(ProverKeyC pk, uint64_t n) {
    uint64_t coeff_size = n;
    uint64_t eval_size = 8 * n;

    // SyncedMemory q_m_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    SyncedMemory q_m_coeffs(0);
    void* q_m_coeffs_gpu = q_m_coeffs.mutable_gpu_data();
    // caffe_gpu_memcpy(q_m_coeffs.size(), pk.q_m_coeffs, q_m_coeffs_gpu);

    SyncedMemory q_m_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_m_evals_gpu = q_m_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_m_evals.size(), pk.q_m_evals, q_m_evals_gpu);

    SyncedMemory q_l_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_l_coeffs_gpu = q_l_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_l_coeffs.size(), pk.q_l_coeffs, q_l_coeffs_gpu);

    SyncedMemory q_l_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_l_evals_gpu = q_l_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_l_evals.size(), pk.q_l_evals, q_l_evals_gpu);

    SyncedMemory q_r_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_r_coeffs_gpu = q_r_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_r_coeffs.size(), pk.q_r_coeffs, q_r_coeffs_gpu);

    SyncedMemory q_r_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_r_evals_gpu = q_r_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_r_evals.size(), pk.q_r_evals, q_r_evals_gpu);

    SyncedMemory q_o_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_o_coeffs_gpu = q_o_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_o_coeffs.size(), pk.q_o_coeffs, q_o_coeffs_gpu);

    SyncedMemory q_o_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_o_evals_gpu = q_o_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_o_evals.size(), pk.q_o_evals, q_o_evals_gpu);

    SyncedMemory q_4_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_4_coeffs_gpu = q_4_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_4_coeffs.size(), pk.q_4_coeffs, q_4_coeffs_gpu);

    SyncedMemory q_4_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_4_evals_gpu = q_4_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_4_evals.size(), pk.q_4_evals, q_4_evals_gpu);

    SyncedMemory q_c_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_c_coeffs_gpu = q_c_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_c_coeffs.size(), pk.q_c_coeffs, q_c_coeffs_gpu);

    SyncedMemory q_c_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_c_evals_gpu = q_c_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_c_evals.size(), pk.q_c_evals, q_c_evals_gpu);

    SyncedMemory q_hl_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_hl_coeffs_gpu = q_hl_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_hl_coeffs.size(), pk.q_hl_coeffs, q_hl_coeffs_gpu);

    SyncedMemory q_hl_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_hl_evals_gpu = q_hl_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_hl_evals.size(), pk.q_hl_evals, q_hl_evals_gpu);

    SyncedMemory q_hr_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_hr_coeffs_gpu = q_hr_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_hr_coeffs.size(), pk.q_hr_coeffs, q_hr_coeffs_gpu);

    SyncedMemory q_hr_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_hr_evals_gpu = q_hr_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_hr_evals.size(), pk.q_hr_evals, q_hr_evals_gpu);

    SyncedMemory q_h4_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_h4_coeffs_gpu = q_h4_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_h4_coeffs.size(), pk.q_h4_coeffs, q_h4_coeffs_gpu);

    SyncedMemory q_h4_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_h4_evals_gpu = q_h4_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_h4_evals.size(), pk.q_h4_evals, q_h4_evals_gpu);

    SyncedMemory q_arith_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* q_arith_coeffs_gpu = q_arith_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(q_arith_coeffs.size(), pk.q_arith_coeffs, q_arith_coeffs_gpu);

    SyncedMemory q_arith_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_arith_evals_gpu = q_arith_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_arith_evals.size(), pk.q_arith_evals, q_arith_evals_gpu);

    // SyncedMemory range_selector_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    SyncedMemory range_selector_coeffs(0);
    void* range_selector_coeffs_gpu = range_selector_coeffs.mutable_gpu_data();
    // caffe_gpu_memcpy(range_selector_coeffs.size(), pk.range_selector_coeffs, range_selector_coeffs_gpu);

    SyncedMemory range_selector_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* range_selector_evals_gpu = range_selector_evals.mutable_gpu_data();
    caffe_gpu_memcpy(range_selector_evals.size(), pk.range_selector_evals, range_selector_evals_gpu);

    // SyncedMemory logic_selector_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    SyncedMemory logic_selector_coeffs(0);
    void* logic_selector_coeffs_gpu = logic_selector_coeffs.mutable_gpu_data();
    // caffe_gpu_memcpy(logic_selector_coeffs.size(), pk.logic_selector_coeffs, logic_selector_coeffs_gpu);

    SyncedMemory logic_selector_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* logic_selector_evals_gpu = logic_selector_evals.mutable_gpu_data();
    caffe_gpu_memcpy(logic_selector_evals.size(), pk.logic_selector_evals, logic_selector_evals_gpu);

    // SyncedMemory fixed_group_add_selector_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    SyncedMemory fixed_group_add_selector_coeffs(0);
    void* fixed_group_add_selector_coeffs_gpu = fixed_group_add_selector_coeffs.mutable_gpu_data();
    // caffe_gpu_memcpy(fixed_group_add_selector_coeffs.size(), pk.fixed_group_add_selector_coeffs, fixed_group_add_selector_coeffs_gpu);

    SyncedMemory fixed_group_add_selector_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* fixed_group_add_selector_evals_gpu = fixed_group_add_selector_evals.mutable_gpu_data();
    caffe_gpu_memcpy(fixed_group_add_selector_evals.size(), pk.fixed_group_add_selector_evals, fixed_group_add_selector_evals_gpu);

    // SyncedMemory variable_group_add_selector_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    SyncedMemory variable_group_add_selector_coeffs(0);
    void* variable_group_add_selector_coeffs_gpu = variable_group_add_selector_coeffs.mutable_gpu_data();
    // caffe_gpu_memcpy(variable_group_add_selector_coeffs.size(), pk.variable_group_add_selector_coeffs, variable_group_add_selector_coeffs_gpu);

    SyncedMemory variable_group_add_selector_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* variable_group_add_selector_evals_gpu = variable_group_add_selector_evals.mutable_gpu_data();
    caffe_gpu_memcpy(variable_group_add_selector_evals.size(), pk.variable_group_add_selector_evals, variable_group_add_selector_evals_gpu);

    // SyncedMemory q_lookup_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    SyncedMemory q_lookup_coeffs(0);
    void* q_lookup_coeffs_gpu = q_lookup_coeffs.mutable_gpu_data();
    // caffe_gpu_memcpy(q_lookup_coeffs.size(), pk.q_lookup_coeffs, q_lookup_coeffs_gpu);

    SyncedMemory q_lookup_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* q_lookup_evals_gpu = q_lookup_evals.mutable_gpu_data();
    caffe_gpu_memcpy(q_lookup_evals.size(), pk.q_lookup_evals, q_lookup_evals_gpu);

    SyncedMemory table1(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* table1_gpu = table1.mutable_gpu_data();
    caffe_gpu_memcpy(table1.size(), pk.table1, table1_gpu);

    SyncedMemory table2(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* table2_gpu = table2.mutable_gpu_data();
    caffe_gpu_memcpy(table2.size(), pk.table2, table2_gpu);

    SyncedMemory table3(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* table3_gpu = table3.mutable_gpu_data();
    caffe_gpu_memcpy(table3.size(), pk.table3, table3_gpu);

    SyncedMemory table4(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* table4_gpu = table4.mutable_gpu_data();
    caffe_gpu_memcpy(table4.size(), pk.table4, table4_gpu);

    SyncedMemory left_sigma_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* left_sigma_coeffs_gpu = left_sigma_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(left_sigma_coeffs.size(), pk.left_sigma_coeffs, left_sigma_coeffs_gpu);

    SyncedMemory left_sigma_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* left_sigma_evals_gpu = left_sigma_evals.mutable_gpu_data();
    caffe_gpu_memcpy(left_sigma_evals.size(), pk.left_sigma_evals, left_sigma_evals_gpu);

    SyncedMemory right_sigma_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* right_sigma_coeffs_gpu = right_sigma_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(right_sigma_coeffs.size(), pk.right_sigma_coeffs, right_sigma_coeffs_gpu);

    SyncedMemory right_sigma_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* right_sigma_evals_gpu = right_sigma_evals.mutable_gpu_data();
    caffe_gpu_memcpy(right_sigma_evals.size(), pk.right_sigma_evals, right_sigma_evals_gpu);

    SyncedMemory out_sigma_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* out_sigma_coeffs_gpu = out_sigma_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(out_sigma_coeffs.size(), pk.out_sigma_coeffs, out_sigma_coeffs_gpu);

    SyncedMemory out_sigma_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* out_sigma_evals_gpu = out_sigma_evals.mutable_gpu_data();
    caffe_gpu_memcpy(out_sigma_evals.size(), pk.out_sigma_evals, out_sigma_evals_gpu);

    SyncedMemory fourth_sigma_coeffs(coeff_size*fr::Limbs*sizeof(uint64_t));
    void* fourth_sigma_coeffs_gpu = fourth_sigma_coeffs.mutable_gpu_data();
    caffe_gpu_memcpy(fourth_sigma_coeffs.size(), pk.fourth_sigma_coeffs, fourth_sigma_coeffs_gpu);

    SyncedMemory fourth_sigma_evals(eval_size*fr::Limbs*sizeof(uint64_t));
    void* fourth_sigma_evals_gpu = fourth_sigma_evals.mutable_gpu_data();
    caffe_gpu_memcpy(fourth_sigma_evals.size(), pk.fourth_sigma_evals, fourth_sigma_evals_gpu);

    SyncedMemory linear_evaluations(eval_size*fr::Limbs*sizeof(uint64_t));
    void* linear_evaluations_gpu = linear_evaluations.mutable_gpu_data();
    caffe_gpu_memcpy(linear_evaluations.size(), pk.linear_evaluations, linear_evaluations_gpu);

    SyncedMemory v_h_coset_8n(eval_size*fr::Limbs*sizeof(uint64_t));
    void* v_h_coset_8n_gpu = v_h_coset_8n.mutable_gpu_data();
    caffe_gpu_memcpy(v_h_coset_8n.size(), pk.v_h_coset_8n, v_h_coset_8n_gpu);

    ProverKey proverkey = ProverKey(q_m_coeffs,
         q_m_evals,
         q_l_coeffs,
         q_l_evals,
         q_r_coeffs,
         q_r_evals,
         q_o_coeffs,
         q_o_evals,
         q_4_coeffs,
         q_4_evals,
         q_c_coeffs,
         q_c_evals,
         q_hl_coeffs,
         q_hl_evals,
         q_hr_coeffs,
         q_hr_evals,
         q_h4_coeffs,
         q_h4_evals,
         q_arith_coeffs,
         q_arith_evals,
         range_selector_coeffs,
         range_selector_evals,
         logic_selector_coeffs,
         logic_selector_evals,
         fixed_group_add_selector_coeffs,
         fixed_group_add_selector_evals,
         variable_group_add_selector_coeffs,
         variable_group_add_selector_evals,
         q_lookup_coeffs,
         q_lookup_evals,
         table1,
         table2,
         table3,
         table4,
         left_sigma_coeffs,
         left_sigma_evals,
         right_sigma_coeffs,
         right_sigma_evals,
         out_sigma_coeffs,
         out_sigma_evals,
         fourth_sigma_coeffs,
         fourth_sigma_evals,
         linear_evaluations,
         v_h_coset_8n);
    return proverkey;
}

Circuit::Circuit(
        uint64_t n,
        uint64_t lookup_len,
        uint64_t intended_pi_pos,
        SyncedMemory cs_q_lookup,
        SyncedMemory public_inputs,
        SyncedMemory w_l,
        SyncedMemory w_r,
        SyncedMemory w_o,
        SyncedMemory w_4
    ) : n(n), lookup_len(lookup_len),
        intended_pi_pos(intended_pi_pos),
        cs_q_lookup(cs_q_lookup),
        public_inputs(public_inputs),
        w_l(w_l),
        w_r(w_r),
        w_o(w_o),
        w_4(w_4)
    {}

Circuit load_cs(CircuitC cs){
    uint64_t n = cs.n;
    SyncedMemory q_lookup(n*fr::Limbs*sizeof(uint64_t));
    void* q_lookup_gpu = q_lookup.mutable_gpu_data();
    caffe_gpu_memcpy(q_lookup.size(), cs.cs_q_lookup, q_lookup_gpu);

    SyncedMemory pi(fr::Limbs*sizeof(uint64_t));
    void* pi_cpu = pi.mutable_cpu_data();
    memcpy(pi_cpu, cs.public_inputs, pi.size());

    SyncedMemory w_l(n*fr::Limbs*sizeof(uint64_t));
    void* w_l_gpu = w_l.mutable_gpu_data();
    caffe_gpu_memcpy(w_l.size(), cs.w_l, w_l_gpu);

    SyncedMemory w_r(n*fr::Limbs*sizeof(uint64_t));
    void* w_r_gpu = w_r.mutable_gpu_data();
    caffe_gpu_memcpy(w_r.size(), cs.w_r, w_r_gpu);

    SyncedMemory w_o(n*fr::Limbs*sizeof(uint64_t));
    void* w_o_gpu = w_o.mutable_gpu_data();
    caffe_gpu_memcpy(w_o.size(), cs.w_o, w_o_gpu);

    SyncedMemory w_4(n*fr::Limbs*sizeof(uint64_t));
    void* w_4_gpu = w_4.mutable_gpu_data();
    caffe_gpu_memcpy(w_4.size(), cs.w_4, w_4_gpu);

    return Circuit(cs.n, cs.lookup_len, cs.intended_pi_pos, 
                   q_lookup, pi, w_l, w_r, w_o, w_4);
}

CommitKey::CommitKey(
        SyncedMemory powers_of_g,
        SyncedMemory powers_of_gamma_g
    ) : powers_of_g(powers_of_g),
        powers_of_gamma_g(powers_of_gamma_g)
    {}

CommitKey load_ck(CommitKeyC ck, uint64_t n){
    SyncedMemory powers_of_g(2 * n * fq::Limbs * sizeof(uint64_t));
    void* powers_of_g_gpu = powers_of_g.mutable_gpu_data();
    caffe_gpu_memcpy(powers_of_g.size(), (void*)ck.powers_of_g, powers_of_g_gpu);

    SyncedMemory powers_of_gamma_g(2 * 2 * fq::Limbs * sizeof(uint64_t));
    void* powers_of_gamma_g_gpu = powers_of_gamma_g.mutable_gpu_data();
    caffe_gpu_memcpy(powers_of_gamma_g.size(), (void*)ck.powers_of_gamma_g, powers_of_gamma_g_gpu);

    return CommitKey(powers_of_g, powers_of_gamma_g);
}
