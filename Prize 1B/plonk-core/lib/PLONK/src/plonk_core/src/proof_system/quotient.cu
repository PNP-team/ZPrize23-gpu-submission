#include "mod.cuh"

SyncedMemory compute_first_lagrange_poly_scaled(uint64_t n, SyncedMemory scale) {
    Intt INTT(fr::TWO_ADICITY);
    SyncedMemory x_evals = pad_poly(scale, n);
    SyncedMemory x_coeffs = INTT.forward(x_evals);
    return x_coeffs;
}

SyncedMemory compute_gate_constraint_satisfiability(
    Ntt_coset& coset_NTT,
    SyncedMemory range_challenge,
    SyncedMemory logic_challenge,
    SyncedMemory fixed_base_challenge,
    SyncedMemory var_base_challenge,
    Arithmetic& arithmetics_evals,
    Selectors& selectors_evals,
    SyncedMemory wl_eval_8n,
    SyncedMemory wr_eval_8n,
    SyncedMemory wo_eval_8n,
    SyncedMemory w4_eval_8n,
    SyncedMemory pi_poly) {

    SyncedMemory pi_eval_8n = coset_NTT.forward(pi_poly);
    
    int size = coset_NTT.Size;

    SyncedMemory a_val = slice(wl_eval_8n, size, true);
    SyncedMemory b_val = slice(wr_eval_8n, size, true);
    SyncedMemory c_val = slice(wo_eval_8n, size, true);
    SyncedMemory d_val = slice(w4_eval_8n, size, true);

    WitnessValues wit_vals{a_val,b_val,c_val,d_val};

    SyncedMemory wl_eval_8n_slice_tail = slice(wl_eval_8n, size, false);

    SyncedMemory wr_eval_8n_slice_tail = slice(wr_eval_8n, size, false);
    SyncedMemory wo_eval_8n_slice_tail = slice(wo_eval_8n, size, false);
    SyncedMemory w4_eval_8n_slice_tail = slice(w4_eval_8n, size, false);
    CustomGate custom_vals{wl_eval_8n_slice_tail,
                           wr_eval_8n_slice_tail,
                           w4_eval_8n_slice_tail,
                           arithmetics_evals.q_l,
                           arithmetics_evals.q_r,
                           arithmetics_evals.q_c,
                           arithmetics_evals.q_hl,
                           arithmetics_evals.q_hr,
                           arithmetics_evals.q_h4};
    

    SyncedMemory arithmetic = compute_quotient_i(arithmetics_evals, wit_vals);

    SyncedMemory range_term = range_quotient_term(
        selectors_evals.range_selector,
        range_challenge,
        wit_vals,
        custom_vals
    );

    SyncedMemory logic_term = logic_quotient_term(
        selectors_evals.logic_selector,
        logic_challenge,
        wit_vals,
        custom_vals
    );

    SyncedMemory fixed_base_scalar_mul_term = FBSMGate_quotient_term(
        selectors_evals.fixed_group_add_selector,
        fixed_base_challenge,
        wit_vals,
        FBSMValues::from_evaluations(custom_vals)
    );

    SyncedMemory curve_addition_term = CAGate_quotient_term(
        selectors_evals.variable_group_add_selector,
        var_base_challenge,
        wit_vals,
        CAValues::from_evaluations(custom_vals)
    );

    SyncedMemory gate_contributions = add_mod(arithmetic, pi_eval_8n);

    add_mod_(gate_contributions, range_term);
    add_mod_(gate_contributions, logic_term);
    add_mod_(gate_contributions, fixed_base_scalar_mul_term);
    add_mod_(gate_contributions, curve_addition_term);

    return gate_contributions;
}

SyncedMemory compute_permutation_checks(
    uint64_t n,
    Ntt_coset& coset_ntt,
    SyncedMemory linear_evaluations_evals,
    Permutation permutations_evals,
    SyncedMemory wl_eval_8n,
    SyncedMemory wr_eval_8n,
    SyncedMemory wo_eval_8n,
    SyncedMemory w4_eval_8n,
    SyncedMemory z_eval_8n,
    SyncedMemory alpha,
    SyncedMemory beta,
    SyncedMemory gamma) {

    uint64_t size = 8 * n;

    SyncedMemory alpha2 = mul_mod(alpha, alpha);
    void* alpha2_gpu_data=alpha2.mutable_gpu_data();
    SyncedMemory l1_poly_alpha = compute_first_lagrange_poly_scaled(n, alpha2);
    SyncedMemory l1_alpha_sq_evals = coset_ntt.forward(l1_poly_alpha);
    l1_poly_alpha = SyncedMemory();

    SyncedMemory wl_eval_8n_slice_head = slice(wl_eval_8n, size, true);
    SyncedMemory wr_eval_8n_slice_head = slice(wr_eval_8n, size, true);
    SyncedMemory wo_eval_8n_slice_head = slice(wo_eval_8n, size, true);
    SyncedMemory w4_eval_8n_slice_head = slice(w4_eval_8n, size, true);
    SyncedMemory z_eval_8n_slice_head = slice(z_eval_8n, size, true);
    SyncedMemory z_eval_8n_slice_tail = slice(z_eval_8n, size, false);
    SyncedMemory l1_alpha_sq_evals_slice_head = slice(l1_alpha_sq_evals, size, true);
    
    SyncedMemory quotient = permutation_compute_quotient(
        size,
        linear_evaluations_evals,
        permutations_evals.left_sigma,
        permutations_evals.right_sigma,
        permutations_evals.out_sigma,
        permutations_evals.fourth_sigma,
        wl_eval_8n_slice_head,
        wr_eval_8n_slice_head,
        wo_eval_8n_slice_head,
        w4_eval_8n_slice_head,
        z_eval_8n_slice_head,
        z_eval_8n_slice_tail,
        alpha,
        l1_alpha_sq_evals_slice_head,
        beta,
        gamma
    );

    return quotient;
}

SyncedMemory compute_quotient_poly(
    uint64_t n,
    ProverKey& pk_new,
    SyncedMemory z_poly,
    SyncedMemory z2_poly,
    SyncedMemory w_l_poly,
    SyncedMemory w_r_poly,
    SyncedMemory w_o_poly,
    SyncedMemory w_4_poly,
    SyncedMemory public_inputs_poly,
    SyncedMemory f_poly,
    SyncedMemory table_poly,
    SyncedMemory h1_poly,
    SyncedMemory h2_poly,
    SyncedMemory alpha,
    SyncedMemory beta,
    SyncedMemory gamma,
    SyncedMemory delta,
    SyncedMemory epsilon,
    SyncedMemory zeta,
    SyncedMemory range_challenge,
    SyncedMemory logic_challenge,
    SyncedMemory fixed_base_challenge,
    SyncedMemory var_base_challenge,
    SyncedMemory lookup_challenge) {

    uint64_t coset_size = 8 * n;
    SyncedMemory one = fr::one();
    void* one_gpu_data= one.mutable_gpu_data();
    SyncedMemory l1_poly = compute_first_lagrange_poly_scaled(n, one);

    Ntt_coset NTT_coset(fr::TWO_ADICITY,coset_size);

    SyncedMemory wl_eval_8n_temp = NTT_coset.forward(w_l_poly);
    SyncedMemory wl_eval_8n_head(8 * fr::Limbs * sizeof(uint64_t));
    void* wl_eval_8n_head_gpu = wl_eval_8n_head.mutable_gpu_data();
    void* wl_eval_8n_gpu = wl_eval_8n_temp.mutable_gpu_data();
    caffe_gpu_memcpy(wl_eval_8n_head.size(), wl_eval_8n_gpu, wl_eval_8n_head_gpu);
    SyncedMemory wl_eval_8n = cat(wl_eval_8n_temp, wl_eval_8n_head);
    wl_eval_8n_temp = SyncedMemory();
    wl_eval_8n_head = SyncedMemory();

    SyncedMemory wr_eval_8n_temp = NTT_coset.forward(w_r_poly);
    SyncedMemory b_val(8 * fr::Limbs * sizeof(uint64_t));
    void* b_val_gpu = b_val.mutable_gpu_data();
    void* wr_eval_8n_temp_gpu = wr_eval_8n_temp.mutable_gpu_data();
    caffe_gpu_memcpy(b_val.size(), wr_eval_8n_temp_gpu, b_val_gpu);
    SyncedMemory wr_eval_8n = cat(wr_eval_8n_temp, b_val);
    wr_eval_8n_temp = SyncedMemory();
    b_val = SyncedMemory();

    SyncedMemory wo_eval_8n = NTT_coset.forward(w_o_poly);

    SyncedMemory w4_eval_8n_temp = NTT_coset.forward(w_4_poly);
    SyncedMemory d_val(8 * fr::Limbs * sizeof(uint64_t));
    void* d_val_gpu = d_val.mutable_gpu_data();
    void* w4_eval_8n_temp_gpu = w4_eval_8n_temp.mutable_gpu_data();
    caffe_gpu_memcpy(d_val.size(), w4_eval_8n_temp_gpu, d_val_gpu);
    SyncedMemory w4_eval_8n = cat(w4_eval_8n_temp, d_val);
    w4_eval_8n_temp = SyncedMemory();
    d_val = SyncedMemory();

    SyncedMemory gate_constraints = compute_gate_constraint_satisfiability(
        NTT_coset,
        range_challenge,
        logic_challenge,
        fixed_base_challenge,
        var_base_challenge,
        pk_new.arithmetic_evals,
        pk_new.selectors_evals,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        public_inputs_poly
    );

    SyncedMemory z_eval_8n_temp = NTT_coset.forward(z_poly);
    SyncedMemory z_eval_8n_head = slice(z_eval_8n_temp, 8, true);
    SyncedMemory z_eval_8n = cat(z_eval_8n_temp, z_eval_8n_head);
    z_eval_8n_temp = SyncedMemory();
    z_eval_8n_head = SyncedMemory();
    
    SyncedMemory permutation = compute_permutation_checks(
        n,
        NTT_coset,
        pk_new.linear_evaluations,
        pk_new.permutation_evals,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        z_eval_8n,
        alpha,
        beta,
        gamma
    );

    z_eval_8n = SyncedMemory();

    SyncedMemory lookup = compute_lookup_quotient_term(
        n,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        f_poly,
        table_poly,
        h1_poly,
        h2_poly,
        z2_poly,
        l1_poly,
        delta,
        epsilon,
        zeta,
        lookup_challenge,
        pk_new.lookup_evals
    );
    
    wl_eval_8n = SyncedMemory();
    wr_eval_8n = SyncedMemory();
    wo_eval_8n = SyncedMemory();
    w4_eval_8n = SyncedMemory();

    SyncedMemory numerator = add_mod(gate_constraints, permutation);
    gate_constraints = SyncedMemory();
    permutation = SyncedMemory();

    add_mod_(numerator, lookup);
    lookup = SyncedMemory();

    SyncedMemory denominator = inv_mod(pk_new.v_h_coset_8n);
    SyncedMemory res = mul_mod(numerator, denominator);
    numerator = SyncedMemory();
    denominator = SyncedMemory();

    Intt_coset intt_coset(fr::TWO_ADICITY);
    return intt_coset.forward(res);
    }