#pragma once
#include "bls12_381/fr.cuh"
#include "bls12_381/fq.cuh"

constexpr int A = 5;

struct CommitmentC
{
    uint64_t x[fq::Limbs];
    uint64_t y[fq::Limbs];
    CommitmentC(uint64_t* x_, uint64_t* y_) {
        for(int i = 0; i<fq::Limbs; i++){
            x[i] = x_[i];
            y[i] = y_[i];
        }
    }
};

class AffinePointG1{
    public:
        SyncedMemory x;
        SyncedMemory y;
        AffinePointG1(SyncedMemory a, SyncedMemory b);
        static bool is_zero(AffinePointG1 self);
        CommitmentC CtoR(){
            void* x_ = x.mutable_cpu_data();
            void* y_ = y.mutable_cpu_data();
            return CommitmentC(reinterpret_cast<uint64_t*>(x_), reinterpret_cast<uint64_t*>(y_));
        }
};

class ProjectivePointG1{
    public:
        SyncedMemory x;
        SyncedMemory y;
        SyncedMemory z;
        ProjectivePointG1(SyncedMemory a, SyncedMemory b, SyncedMemory c);
        static bool is_zero(ProjectivePointG1 self);
};

AffinePointG1 to_affine(ProjectivePointG1 G1);

ProjectivePointG1 add_assign(ProjectivePointG1 self, ProjectivePointG1 other);

ProjectivePointG1 double_ProjectivePointG1(ProjectivePointG1 self);

ProjectivePointG1 add_assign_mixed(ProjectivePointG1 self, AffinePointG1 other);

class BTreeMap{
    public:
    SyncedMemory item;
    uint64_t pos;
    BTreeMap(SyncedMemory item_, uint64_t pos_) : item(item_), pos(pos_) {}

    static BTreeMap new_instance(SyncedMemory item_, uint64_t pos_){
        BTreeMap newmap = BTreeMap(item_,pos_);
        return newmap;
    }
};

struct WireEvaluationsC
{
    uint64_t a_eval[fr::Limbs];
    uint64_t b_eval[fr::Limbs];
    uint64_t c_eval[fr::Limbs];
    uint64_t d_eval[fr::Limbs];
    WireEvaluationsC(uint64_t* a, uint64_t* b, uint64_t* c, uint64_t* d){
        for(int i = 0; i<fr::Limbs; i++){
            a_eval[i] = a[i];
            b_eval[i] = b[i];
            c_eval[i] = c[i];
            d_eval[i] = d[i];
        }
    }
};

struct PermutationEvaluationsC
{
    uint64_t left_sigma_eval[fr::Limbs];
    uint64_t right_sigma_eval[fr::Limbs];
    uint64_t out_sigma_eval[fr::Limbs];
    uint64_t permutation_eval[fr::Limbs];
    PermutationEvaluationsC(uint64_t* left_sigma, uint64_t* right_sigma, uint64_t* out_sigma, uint64_t* permutation){
        for(int i = 0; i<fr::Limbs; i++){
            left_sigma_eval[i] = left_sigma[i];
            right_sigma_eval[i] = right_sigma[i];
            out_sigma_eval[i] = out_sigma[i];
            permutation_eval[i] = permutation[i];
        }
    }
};

struct CustomEvaluationsC
{
    uint64_t q_arith_eval[fr::Limbs];
    uint64_t q_c_eval[fr::Limbs];
    uint64_t q_l_eval[fr::Limbs];
    uint64_t q_r_eval[fr::Limbs];
    uint64_t q_hl_eval[fr::Limbs];
    uint64_t q_hr_eval[fr::Limbs];
    uint64_t q_h4_eval[fr::Limbs];
    uint64_t a_next_eval[fr::Limbs];
    uint64_t b_next_eval[fr::Limbs];
    uint64_t d_next_eval[fr::Limbs];

    CustomEvaluationsC(
        uint64_t* qa_eval,
        uint64_t* qc_eval,
        uint64_t* ql_eval,
        uint64_t* qr_eval,
        uint64_t* qhl_eval,
        uint64_t* qhr_eval,
        uint64_t* qh4_eval,
        uint64_t* anext_eval,
        uint64_t* bnext_eval,
        uint64_t* dnext_eval){
            for(int i = 0; i<fr::Limbs; i++){
                q_arith_eval[i] = qa_eval[i];
                q_c_eval[i] = qc_eval[i];
                q_l_eval[i] = ql_eval[i];
                q_r_eval[i] = qr_eval[i];
                q_hl_eval[i] = qhl_eval[i];
                q_hr_eval[i] = qhr_eval[i];
                q_h4_eval[i] = qh4_eval[i];
                a_next_eval[i] = anext_eval[i];
                b_next_eval[i] = bnext_eval[i];
                d_next_eval[i] = dnext_eval[i];
            }
        }
};

struct LookupEvaluationsC
{
    uint64_t q_lookup_eval[fr::Limbs];
    uint64_t z2_next_eval[fr::Limbs];
    uint64_t h1_eval[fr::Limbs];
    uint64_t h1_next_eval[fr::Limbs];
    uint64_t h2_eval[fr::Limbs];
    uint64_t f_eval[fr::Limbs];
    uint64_t table_eval[fr::Limbs];
    uint64_t table_next_eval[fr::Limbs];
    LookupEvaluationsC(
        uint64_t* q_lookup, 
        uint64_t* z2_next, 
        uint64_t* h1, uint64_t* h1_next, 
        uint64_t* h2, 
        uint64_t* f, 
        uint64_t* table, uint64_t* table_next){
            for(int i = 0; i<fr::Limbs; i++){
                q_lookup_eval[i] = q_lookup[i];
                z2_next_eval[i] = z2_next[i];
                h1_eval[i] = h1[i];
                h1_next_eval[i] = h1_next[i];
                h2_eval[i] = h2[i];
                f_eval[i] = f[i];
                table_eval[i] = table[i];
                table_next_eval[i] = table_next[i];
            }
        }
};

struct ProofEvaluationsC{
    WireEvaluationsC wire_evals;  
    PermutationEvaluationsC perm_evals;
    LookupEvaluationsC lookup_evals;
    CustomEvaluationsC custom_evals;
    ProofEvaluationsC(
        WireEvaluationsC w_eval, 
        PermutationEvaluationsC p_eval, 
        LookupEvaluationsC l_eval,
        CustomEvaluationsC c_eval)
        :wire_evals(w_eval), perm_evals(p_eval), lookup_evals(l_eval), custom_evals(c_eval) {}
} ;

struct ProofC {
    CommitmentC a_comm;
    CommitmentC b_comm;
    CommitmentC c_comm;
    CommitmentC d_comm;
    CommitmentC z_comm;
    CommitmentC f_comm;
    CommitmentC h_1_comm;
    CommitmentC h_2_comm;
    CommitmentC z_2_comm;
    CommitmentC t_1_comm;
    CommitmentC t_2_comm;
    CommitmentC t_3_comm;
    CommitmentC t_4_comm;
    CommitmentC t_5_comm;
    CommitmentC t_6_comm;
    CommitmentC t_7_comm;
    CommitmentC t_8_comm;
    CommitmentC aw_opening;
    CommitmentC saw_opening;
    ProofEvaluationsC evaluations;
    ProofC(
        CommitmentC a,
        CommitmentC b,
        CommitmentC c,
        CommitmentC d,
        CommitmentC z,
        CommitmentC f,
        CommitmentC h_1,
        CommitmentC h_2,
        CommitmentC z_2,
        CommitmentC t_1,
        CommitmentC t_2,
        CommitmentC t_3,
        CommitmentC t_4,
        CommitmentC t_5,
        CommitmentC t_6,
        CommitmentC t_7,
        CommitmentC t_8,
        CommitmentC aw,
        CommitmentC saw,
        ProofEvaluationsC evals)
        :
        a_comm(a),
        b_comm(b),
        c_comm(c),
        d_comm(d),
        z_comm(z),
        f_comm(f),
        h_1_comm(h_1),
        h_2_comm(h_2),
        z_2_comm(z_2),
        t_1_comm(t_1),
        t_2_comm(t_2),
        t_3_comm(t_3),
        t_4_comm(t_4),
        t_5_comm(t_5),
        t_6_comm(t_6),
        t_7_comm(t_7),
        t_8_comm(t_8),
        aw_opening(aw),
        saw_opening(saw),
        evaluations(evals) {}
};

typedef struct {
    uint64_t n;
    uint64_t lookup_len;
    uint64_t intended_pi_pos;
    uint64_t* cs_q_lookup;
    uint64_t* public_inputs;
    uint64_t* w_l;
    uint64_t* w_r;
    uint64_t* w_o;
    uint64_t* w_4;
} CircuitC;

typedef struct {
    const uint64_t* powers_of_g;
    const uint64_t* powers_of_gamma_g;
}CommitKeyC;

uint64_t next_power_of_2(uint64_t x);

uint64_t total_size(CircuitC circuit);

uint64_t circuit_bound(CircuitC circuit);


typedef struct {
    uint64_t* q_m_coeffs;
    uint64_t* q_m_evals;

    uint64_t* q_l_coeffs;
    uint64_t* q_l_evals;

    uint64_t* q_r_coeffs;
    uint64_t* q_r_evals;

    uint64_t* q_o_coeffs;
    uint64_t* q_o_evals;

    uint64_t* q_4_coeffs;
    uint64_t* q_4_evals;

    uint64_t* q_c_coeffs;
    uint64_t* q_c_evals;

    uint64_t* q_hl_coeffs;
    uint64_t* q_hl_evals;

    uint64_t* q_hr_coeffs;
    uint64_t* q_hr_evals;

    uint64_t* q_h4_coeffs;
    uint64_t* q_h4_evals;

    uint64_t* q_arith_coeffs;
    uint64_t* q_arith_evals;

    uint64_t* range_selector_coeffs;
    uint64_t* range_selector_evals;

    uint64_t* logic_selector_coeffs;
    uint64_t* logic_selector_evals;

    uint64_t* fixed_group_add_selector_coeffs;
    uint64_t* fixed_group_add_selector_evals;

    uint64_t* variable_group_add_selector_coeffs;
    uint64_t* variable_group_add_selector_evals;

    uint64_t* q_lookup_coeffs;
    uint64_t* q_lookup_evals;
    uint64_t* table1;
    uint64_t* table2;
    uint64_t* table3;
    uint64_t* table4;

    uint64_t* left_sigma_coeffs;
    uint64_t* left_sigma_evals;

    uint64_t* right_sigma_coeffs;
    uint64_t* right_sigma_evals;

    uint64_t* out_sigma_coeffs;
    uint64_t* out_sigma_evals;

    uint64_t* fourth_sigma_coeffs;
    uint64_t* fourth_sigma_evals;

    uint64_t* linear_evaluations;

    uint64_t* v_h_coset_8n;
} ProverKeyC;


class CommitKey {
    public:
        SyncedMemory powers_of_g;
        SyncedMemory powers_of_gamma_g;
        CommitKey(
            SyncedMemory powers_of_g,
            SyncedMemory powers_of_gamma_g
        );
};

class Circuit {
    public:
        uint64_t n;
        uint64_t lookup_len;
        uint64_t intended_pi_pos;
        SyncedMemory cs_q_lookup;
        SyncedMemory public_inputs;
        SyncedMemory w_l;
        SyncedMemory w_r;
        SyncedMemory w_o;
        SyncedMemory w_4;
        Circuit(
        uint64_t n,
        uint64_t lookup_len,
        uint64_t intended_pi_pos,
        SyncedMemory cs_q_lookup,
        SyncedMemory public_inputs,
        SyncedMemory w_l,
        SyncedMemory w_r,
        SyncedMemory w_o,
        SyncedMemory w_4
    );
};


struct LookupTable {
    SyncedMemory q_lookup;
    SyncedMemory table1;
    SyncedMemory table2;
    SyncedMemory table3;
    SyncedMemory table4;

    LookupTable(SyncedMemory ql, SyncedMemory t1, SyncedMemory t2, SyncedMemory t3, SyncedMemory t4);
};

struct Permutation
{
    SyncedMemory left_sigma;
    SyncedMemory right_sigma;
    SyncedMemory out_sigma;
    SyncedMemory fourth_sigma;

    Permutation(SyncedMemory ls, SyncedMemory rs, SyncedMemory os, SyncedMemory fs);
    ~Permutation(){}
};

struct Arithmetic
{
    SyncedMemory q_m;
    SyncedMemory q_l;
    SyncedMemory q_r;
    SyncedMemory q_o;
    SyncedMemory q_4;
    SyncedMemory q_c;
    SyncedMemory q_hl;
    SyncedMemory q_hr;
    SyncedMemory q_h4;
    SyncedMemory q_arith;

    Arithmetic(SyncedMemory qm, SyncedMemory ql, SyncedMemory qr,
               SyncedMemory qo, SyncedMemory q4, SyncedMemory qc,
               SyncedMemory qhl, SyncedMemory qhr, SyncedMemory qh4,
               SyncedMemory qarith);
    ~Arithmetic(){}
};

struct Selectors
{
    SyncedMemory range_selector;
    SyncedMemory logic_selector;
    SyncedMemory fixed_group_add_selector;
    SyncedMemory variable_group_add_selector;

    Selectors(SyncedMemory rs,
              SyncedMemory ls,
              SyncedMemory fs,
              SyncedMemory vs);
    ~Selectors(){}
};


class ProverKey{
    public:
        Arithmetic arithmetic_coeffs;
        Arithmetic arithmetic_evals;

        Selectors selectors_coeffs;
        Selectors selectors_evals;
        
        LookupTable lookup_coeffs;
        SyncedMemory lookup_evals;

        Permutation permutation_coeffs;
        Permutation permutation_evals;

        SyncedMemory linear_evaluations;

        SyncedMemory v_h_coset_8n;
        ProverKey(
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
        SyncedMemory v_h_coset_8n);
};

class labeldpoly {
    public:
        SyncedMemory poly;
        int hiding_bound;
        labeldpoly(SyncedMemory poly_, int hiding_bound_):poly(poly_), hiding_bound(hiding_bound_){}
};

class witness_poly {
    public:
        SyncedMemory witness;
        SyncedMemory random_witness;
        witness_poly(SyncedMemory witness_, SyncedMemory random_witness_): witness(witness_), random_witness(random_witness_){}
};

ProverKey load_pk(ProverKeyC pk, uint64_t n);

Circuit load_cs(CircuitC cs);

CommitKey load_ck(CommitKeyC ck, uint64_t n);