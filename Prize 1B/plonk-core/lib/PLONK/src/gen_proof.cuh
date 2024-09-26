#pragma once
#include <functional>
#include "transcript/transcript.cuh"
#include "KZG/kzg10.cuh"
#include "plonk_core/src/permutation/mod.cuh"
#include "plonk_core/src/proof_system/pi.cuh"
#include "plonk_core/src/proof_system/quotient.cuh"
#include "plonk_core/src/proof_system/linearisation.cuh"

ProofC prove(ProverKeyC pkc, CircuitC csc, CommitKeyC ckc){
    uint64_t size = circuit_bound(csc);
    ProverKey pk = load_pk(pkc, size);
    Circuit cs = load_cs(csc);
    CommitKey ck = load_ck(ckc, size);

    Radix2EvaluationDomain domain = newdomain(size);
    uint64_t n = domain.size;

    char transcript_init[] = "Merkle tree";
    Transcript transcript = Transcript(transcript_init);
    char pi[] = "pi";
    transcript.append_pi(pi, cs.public_inputs, cs.intended_pi_pos);
    
    // 1. Compute witness Polynomials
    SyncedMemory w_l_scalar = pad_poly(cs.w_l, n);
    SyncedMemory w_r_scalar = pad_poly(cs.w_r, n);
    SyncedMemory w_o_scalar = pad_poly(cs.w_o, n);
    SyncedMemory w_4_scalar = pad_poly(cs.w_4, n);

    Intt INTT(fr::TWO_ADICITY);

    SyncedMemory w_l_poly = INTT.forward(w_l_scalar);
    SyncedMemory w_r_poly = INTT.forward(w_r_scalar);
    SyncedMemory w_o_poly = INTT.forward(w_o_scalar);
    SyncedMemory w_4_poly = INTT.forward(w_4_scalar);

    std::vector<labeldpoly> w_polys;
    w_polys.push_back(labeldpoly(w_l_poly, NULL));
    w_polys.push_back(labeldpoly(w_r_poly, NULL));
    w_polys.push_back(labeldpoly(w_o_poly, NULL));
    w_polys.push_back(labeldpoly(w_4_poly, NULL));

    std::vector<CommitResult> w_commits = commit_poly(ck, w_polys);  
    transcript.append("w_l", w_commits[0].commitment);
    int tt[200];
    for(int i = 0; i<200; i++){
        tt[i] = (int)transcript.strobe.state[i];
    }
    transcript.append("w_r", w_commits[1].commitment);
    transcript.append("w_o", w_commits[2].commitment);
    transcript.append("w_4", w_commits[3].commitment);

    // 2. Derive lookup polynomials

    // Generate table compression factor
    SyncedMemory zeta = transcript.challenge_scalar("zeta");
    transcript.append("zeta", zeta);
    void* zeta_gpu = zeta.mutable_gpu_data();

    // Compress lookup table into vector of single elements
    SyncedMemory compressed_t_multiset = compress(pk.lookup_coeffs.table1, pk.lookup_coeffs.table2, pk.lookup_coeffs.table3, pk.lookup_coeffs.table4, zeta);

    // Compute table poly
    SyncedMemory table_poly = INTT.forward(compressed_t_multiset);

    // Compute query table f
    SyncedMemory compressed_f_multiset = compute_query_table(
            cs.cs_q_lookup,
            w_l_scalar,
            w_r_scalar,
            w_o_scalar,
            w_4_scalar,
            compressed_t_multiset,
            zeta
        );
    
    // Compute query poly
    SyncedMemory f_poly = INTT.forward(compressed_f_multiset);
    std::vector<labeldpoly> f_polys;
    f_polys.push_back(labeldpoly(f_poly, NULL));

    // Commit to query polynomial
    std::vector<CommitResult> f_poly_commit = commit_poly(ck, f_polys);
    transcript.append("f", f_poly_commit[0].commitment);

    // Compute s, as the sorted and concatenated version of f and t
    SyncedMemory h_1(compressed_t_multiset.size());
    SyncedMemory h_2(compressed_f_multiset.size());

    void* h_1_gpu = h_1.mutable_gpu_data();
    void* h_2_gpu = h_2.mutable_gpu_data();
    caffe_gpu_memset(h_1.size(), 0, h_1_gpu);
    caffe_gpu_memset(h_2.size(), 0, h_2_gpu);

    // Compute h polys
    SyncedMemory h_1_poly = INTT.forward(h_1);
    SyncedMemory h_2_poly = INTT.forward(h_2);

    // Commit to h polys
    std::vector<labeldpoly> h_polys;
    h_polys.push_back(labeldpoly(h_1_poly, NULL));
    h_polys.push_back(labeldpoly(h_2_poly, NULL));
    std::vector<CommitResult> h_commits = commit_poly(ck, h_polys);
    // Add h polynomials to transcript
    transcript.append("h1", h_commits[0].commitment);
    transcript.append("h2", h_commits[1].commitment);

    // 3. Compute permutation polynomial
    // Compute permutation challenge `beta`.
    SyncedMemory beta = transcript.challenge_scalar("beta");
    transcript.append("beta", beta);
    void* beta_gpu = beta.mutable_gpu_data();

    // Compute permutation challenge `gamma`.
    SyncedMemory gamma = transcript.challenge_scalar("gamma");
    transcript.append("gamma", gamma);
    void* gamma_gpu = gamma.mutable_gpu_data();

    // Compute permutation challenge `delta`.
    SyncedMemory delta = transcript.challenge_scalar("delta");
    transcript.append("delta", delta);
    void* delta_gpu = delta.mutable_gpu_data();

    // Compute permutation challenge `epsilon`.
    SyncedMemory epsilon = transcript.challenge_scalar("epsilon");
    transcript.append("epsilon", epsilon);
    void* epsilon_gpu = epsilon.mutable_gpu_data();

    // Challenges must be different
    assert(!fr::is_equal(beta, gamma) && "challenges must be different");
    assert(!fr::is_equal(beta, delta) && "challenges must be different");
    assert(!fr::is_equal(beta, epsilon) && "challenges must be different");
    assert(!fr::is_equal(gamma, delta) && "challenges must be different");
    assert(!fr::is_equal(gamma, epsilon) && "challenges must be different");
    assert(!fr::is_equal(delta, epsilon) && "challenges must be different");

    std::vector<SyncedMemory> w_scalars;
    w_scalars.push_back(w_l_scalar);
    w_scalars.push_back(w_r_scalar);
    w_scalars.push_back(w_o_scalar);
    w_scalars.push_back(w_4_scalar);

    std::vector<SyncedMemory> sigma_polys;
    sigma_polys.push_back(pk.permutation_coeffs.left_sigma);
    sigma_polys.push_back(pk.permutation_coeffs.right_sigma);
    sigma_polys.push_back(pk.permutation_coeffs.out_sigma);
    sigma_polys.push_back(pk.permutation_coeffs.fourth_sigma);
    
    SyncedMemory z_poly = compute_permutation_poly(domain, w_scalars, beta, gamma, sigma_polys);
    
    // Commit to permutation polynomial.
    std::vector<labeldpoly> z_polys;
    z_polys.push_back(labeldpoly(z_poly, NULL));
    std::vector<CommitResult>z_poly_commit = commit_poly(ck, z_polys);

    // Add permutation polynomial commitment to transcript.
    transcript.append("z", z_poly_commit[0].commitment);

    // Compute mega permutation polynomial.
    // Compute lookup permutation poly
    SyncedMemory z_2_poly = compute_lookup_permutation_poly(n, compressed_f_multiset, compressed_t_multiset, 
                                                             h_1, h_2, delta, epsilon);
    
    // Commit to lookup permutation polynomial.
    std::vector<labeldpoly> z_2_polys;
    z_2_polys.push_back(labeldpoly(z_2_poly, NULL));
    std::vector<CommitResult>z_2_poly_commit = commit_poly(ck, z_2_polys);

    // 3. Compute public inputs polynomial
    SyncedMemory pi_poly = into_dense_poly(cs.public_inputs, cs.intended_pi_pos, n, INTT);
    // 4. Compute quotient polynomial
    // Compute quotient challenge `alpha`, and gate-specific separation challenges.
    SyncedMemory alpha = transcript.challenge_scalar("alpha");
    transcript.append("alpha", alpha);

    SyncedMemory range_sep_challenge = transcript.challenge_scalar("range separation challenge");
    transcript.append("range seperation challenge", range_sep_challenge);

    SyncedMemory logic_sep_challenge = transcript.challenge_scalar("logic separation challenge");
    transcript.append("logic seperation challenge", logic_sep_challenge);

    SyncedMemory fixed_base_sep_challenge = transcript.challenge_scalar("fixed base separation challenge");
    transcript.append("fixed base separation challenge", fixed_base_sep_challenge);

    SyncedMemory var_base_sep_challenge = transcript.challenge_scalar("variable base separation challenge");
    transcript.append("variable base separation challenge", var_base_sep_challenge);

    SyncedMemory lookup_sep_challenge = transcript.challenge_scalar("lookup separation challenge");
    transcript.append("lookup separation challenge", lookup_sep_challenge);

    SyncedMemory t_poly = compute_quotient_poly(
        n,
        pk,
        z_poly,
        z_2_poly,
        w_l_poly,
        w_r_poly,
        w_o_poly,
        w_4_poly,
        pi_poly,
        f_poly,
        table_poly,
        h_1_poly,
        h_2_poly,
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        zeta,
        range_sep_challenge,
        logic_sep_challenge,
        fixed_base_sep_challenge,
        var_base_sep_challenge,
        lookup_sep_challenge);

    std::vector<SyncedMemory> t_x = split_tx_poly(n, t_poly);
    std::vector<labeldpoly> t_i_polys;
    for(int i=0;i<t_x.size();i++){
        t_i_polys.push_back(labeldpoly(t_x[i], NULL));
    }
    std::vector<CommitResult> t_commits = commit_poly(ck, t_i_polys);

    // Add quotient polynomial commitments to transcript
    for(int i=0;i<t_commits.size();i++){
        std::string t_i_ = "t_" + std::to_string(i + 1);
        char* t_i = t_i_.data();
        transcript.append(t_i, t_commits[i].commitment);
    }

    // 4. Compute linearisation polynomial
    // Compute evaluation challenge `z`.
    SyncedMemory z_challenge = transcript.challenge_scalar("z");
    transcript.append("z", z_challenge);
    void* z_challenge_gpu = z_challenge.mutable_gpu_data();
    linear Linear = compute_linearisation_poly(
        domain,
        pk,
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        zeta,
        range_sep_challenge,
        logic_sep_challenge,
        fixed_base_sep_challenge,
        var_base_sep_challenge,
        lookup_sep_challenge,
        z_challenge,
        w_l_poly,
        w_r_poly,
        w_o_poly,
        w_4_poly,
        t_x[0],
        t_x[1],
        t_x[2],
        t_x[3],
        t_x[4],
        t_x[5],
        t_x[6],
        t_x[7],
        z_poly,
        z_2_poly,
        f_poly,
        h_1_poly,
        h_2_poly,
        table_poly
    );

    // Add evaluations to transcript.
    // First wire evals
    transcript.append("a_eval", Linear.evaluations.wire_evals.a_eval);
    transcript.append("b_eval", Linear.evaluations.wire_evals.b_eval);
    transcript.append("c_eval", Linear.evaluations.wire_evals.c_eval);
    transcript.append("d_eval", Linear.evaluations.wire_evals.d_eval);

    // Second permutation evals
    transcript.append("left_sig_eval", Linear.evaluations.perm_evals.left_sigma_eval);
    transcript.append("right_sig_eval", Linear.evaluations.perm_evals.right_sigma_eval);
    transcript.append("out_sig_eval", Linear.evaluations.perm_evals.out_sigma_eval);
    transcript.append("perm_eval", Linear.evaluations.perm_evals.permutation_eval);

    // Third lookup evals
    transcript.append("f_eval", Linear.evaluations.lookup_evals.f_eval);
    transcript.append("q_lookup_eval", Linear.evaluations.lookup_evals.q_lookup_eval);
    transcript.append("lookup_perm_eval", Linear.evaluations.lookup_evals.z2_next_eval);
    transcript.append("h_1_eval", Linear.evaluations.lookup_evals.h1_eval);
    transcript.append("h_1_next_eval", Linear.evaluations.lookup_evals.h1_next_eval);
    transcript.append("h_2_eval", Linear.evaluations.lookup_evals.h2_eval);

    // Fourth, all evals needed for custom gates
    transcript.append("q_arith_eval", Linear.evaluations.custom_evals.q_arith_eval);
    transcript.append("q_c_eval", Linear.evaluations.custom_evals.q_c_eval);
    transcript.append("q_l_eval", Linear.evaluations.custom_evals.q_l_eval);
    transcript.append("q_r_eval", Linear.evaluations.custom_evals.q_r_eval);
    transcript.append("q_hl_eval", Linear.evaluations.custom_evals.q_hl_eval);
    transcript.append("q_hr_eval", Linear.evaluations.custom_evals.q_hr_eval);
    transcript.append("q_h4_eval", Linear.evaluations.custom_evals.q_h4_eval);
    transcript.append("a_next_eval", Linear.evaluations.custom_evals.a_next_eval);
    transcript.append("b_next_eval", Linear.evaluations.custom_evals.b_next_eval);
    transcript.append("d_next_eval", Linear.evaluations.custom_evals.d_next_eval);

    // 5. Compute Openings using KZG10
    // We merge the quotient polynomial using the `z_challenge` so the SRS
    // is linear in the circuit size `n`

    // Compute aggregate witness to polynomials evaluated at the evaluation
    SyncedMemory aw_challenge = transcript.challenge_scalar("aggregate_witness");
    
    std::vector<labeldpoly> aw_polys;
    aw_polys.push_back(labeldpoly(Linear.linear_poly, NULL));
    aw_polys.push_back(labeldpoly(pk.permutation_coeffs.left_sigma, NULL));
    aw_polys.push_back(labeldpoly(pk.permutation_coeffs.right_sigma, NULL));
    aw_polys.push_back(labeldpoly(pk.permutation_coeffs.out_sigma, NULL));
    aw_polys.push_back(labeldpoly(f_poly, NULL));
    aw_polys.push_back(labeldpoly(h_2_poly, NULL));
    aw_polys.push_back(labeldpoly(table_poly, NULL));

    std::vector<CommitResult> aw_commits = commit_poly(ck, aw_polys);
    aw_polys.push_back(w_polys[0]);
    aw_polys.push_back(w_polys[1]);
    aw_polys.push_back(w_polys[2]);
    aw_polys.push_back(w_polys[3]);
    std::vector<SyncedMemory> aw_rands;
    for(int i = 0; i<aw_commits.size(); i++){
        aw_rands.push_back(aw_commits[i].randomness);
    }
    for(int i = 0; i<w_commits.size(); i++){
        aw_rands.push_back(w_commits[i].randomness);
    }

    OpenProof aw_opening = open_proof(
                            ck, aw_polys, 
                            z_challenge, aw_challenge,
                            aw_rands);
    
    SyncedMemory saw_challenge = transcript.challenge_scalar("aggregate_witness");
    void* saw_challenge_gpu = saw_challenge.mutable_gpu_data();

    std::vector<labeldpoly> saw_polys;
    saw_polys.push_back(labeldpoly(z_poly, NULL));
    saw_polys.push_back(labeldpoly(w_l_poly, NULL));
    saw_polys.push_back(labeldpoly(w_r_poly, NULL));
    saw_polys.push_back(labeldpoly(w_4_poly, NULL));
    saw_polys.push_back(labeldpoly(h_1_poly, NULL));
    saw_polys.push_back(labeldpoly(z_2_poly, NULL));
    saw_polys.push_back(labeldpoly(table_poly, NULL));
    std::vector<CommitResult> saw_commits = commit_poly(ck, saw_polys);

    std::vector<SyncedMemory> saw_rands;
    for(int i = 0; i<saw_commits.size(); i++){
        saw_rands.push_back(saw_commits[i].randomness);
    }
    SyncedMemory element = from_element(1, domain);
    SyncedMemory open_point = mul_mod(z_challenge, element);
    OpenProof saw_opening = open_proof(
        ck,
        saw_polys,
        open_point,
        saw_challenge,
        saw_rands);

    ProofC proof = ProofC(
            w_commits[0].commitment.CtoR(),
            w_commits[1].commitment.CtoR(),
            w_commits[2].commitment.CtoR(),
            w_commits[3].commitment.CtoR(),
            saw_commits[0].commitment.CtoR(),
            f_poly_commit[0].commitment.CtoR(),
            h_commits[0].commitment.CtoR(),
            h_commits[1].commitment.CtoR(),
            z_2_poly_commit[0].commitment.CtoR(),
            t_commits[0].commitment.CtoR(),
            t_commits[1].commitment.CtoR(),
            t_commits[2].commitment.CtoR(),
            t_commits[3].commitment.CtoR(),
            t_commits[4].commitment.CtoR(),
            t_commits[5].commitment.CtoR(),
            t_commits[6].commitment.CtoR(),
            t_commits[7].commitment.CtoR(),
            aw_opening.w.CtoR(),
            saw_opening.w.CtoR(),
            Linear.evaluations.CtoR()
        );
    return proof;
}