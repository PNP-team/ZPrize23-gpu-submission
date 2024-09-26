// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE
// or https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Permutations over Lagrange-bases for Oecumenical Noninteractive
//! arguments of Knowledge (PLONK) is a zero knowledge proof system.
//!
//! This protocol was created by:
//! - Ariel Gabizon (Protocol Labs),
//! - Zachary J. Williamson (Aztec Protocol)
//! - Oana Ciobotaru
//!
//! This crate contains a pure Rust implementation of this algorithm using
//! code done by the creators of the protocol as a reference implementation:
//!
//! <https://github.com/AztecProtocol/barretenberg/blob/master/barretenberg/src/aztec/plonk/>

// Bitshift/Bitwise ops are allowed to gain performance.
#![allow(clippy::suspicious_arithmetic_impl)]
// Some structs do not have AddAssign or MulAssign impl.
#![allow(clippy::suspicious_op_assign_impl)]
// Variables have always the same names in respect to wires.
#![allow(clippy::many_single_char_names)]
// Bool expr are usually easier to read with match statements.
#![allow(clippy::match_bool)]
// We have quite some functions that require quite some args by it's nature.
// It can be refactored but for now, we avoid these warns.
#![allow(clippy::too_many_arguments)]
#![deny(rustdoc::broken_intra_doc_links)]
#![feature(trait_alias)]
// #![deny(missing_docs)]
#![feature(more_qualified_paths)]
extern crate alloc;

mod permutation;
mod transcript;
pub mod util;

pub mod circuit;
pub mod commitment;
pub mod constraint_system;
pub mod error;
pub mod lookup;
pub mod prelude;
pub mod proof_system;
#[cfg(test)]
mod test;

extern "C" {
    pub fn transfer_pp(powers_of_g: *const u64, powers_of_gamma_g: *const u64);
}

#[repr(C)]
pub struct WireEvaluationsC {
    pub a_eval: [u64;4],
    pub b_eval: [u64;4],
    pub c_eval: [u64;4],
    pub d_eval: [u64;4]
}

#[repr(C)]
pub struct PermutationEvaluationsC {
    pub left_sigma_eval: [u64;4],
    pub right_sigma_eval: [u64;4],
    pub out_sigma_eval: [u64;4],
    pub permutation_eval: [u64;4]
}

#[repr(C)]
pub struct CustomEvaluationsC {
    pub q_arith_eval: [u64;4],
    pub q_c_eval: [u64;4],
    pub q_l_eval: [u64;4],
    pub q_r_eval: [u64;4],
    pub q_hl_eval: [u64;4],
    pub q_hr_eval: [u64;4],
    pub q_h4_eval: [u64;4],
    pub a_next_eval: [u64;4],
    pub b_next_eval: [u64;4],
    pub d_next_eval: [u64;4]
}

impl CustomEvaluationsC {
    pub fn iter(&self) -> impl Iterator<Item = &[u64; 4]> {
        vec![
            &self.q_arith_eval,
            &self.q_c_eval,
            &self.q_l_eval,
            &self.q_r_eval,
            &self.q_hl_eval,
            &self.q_hr_eval,
            &self.q_h4_eval,
            &self.a_next_eval,
            &self.b_next_eval,
            &self.d_next_eval,
        ]
        .into_iter()
    }
}

#[repr(C)]
pub struct LookupEvaluationsC {
    pub q_lookup_eval: [u64;4],
    pub z2_next_eval: [u64;4],
    pub h1_eval: [u64;4],
    pub h1_next_eval: [u64;4],
    pub h2_eval: [u64;4],
    pub f_eval: [u64;4],
    pub table_eval: [u64;4],
    pub table_next_eval: [u64;4]
}
#[repr(C)]
pub struct ProofEvaluationsC {
    pub wire_evals: WireEvaluationsC,
    pub perm_evals: PermutationEvaluationsC,
    pub lookup_evals: LookupEvaluationsC,
    pub custom_evals: CustomEvaluationsC,
}

#[repr(C)]
pub struct ProofC {
    pub a_comm: CommitmentC,
    pub b_comm: CommitmentC,
    pub c_comm: CommitmentC,
    pub d_comm: CommitmentC,
    pub z_comm: CommitmentC,
    pub f_comm: CommitmentC,
    pub h_1_comm: CommitmentC,
    pub h_2_comm: CommitmentC,
    pub z_2_comm: CommitmentC,
    pub t_1_comm: CommitmentC,
    pub t_2_comm: CommitmentC,
    pub t_3_comm: CommitmentC,
    pub t_4_comm: CommitmentC,
    pub t_5_comm: CommitmentC,
    pub t_6_comm: CommitmentC,
    pub t_7_comm: CommitmentC,
    pub t_8_comm: CommitmentC,
    pub aw_opening: CommitmentC,
    pub saw_opening: CommitmentC,
    pub evaluations:  ProofEvaluationsC,
}

#[repr(C)]
pub struct CircuitC {
    pub n: u64,
    pub lookup_len: u64,
    pub intended_pi_pos:  u64,
    pub q_lookup: *mut u64,
    pub pi: *mut u64,
    pub w_l: *mut u64,
    pub w_r: *mut u64,
    pub w_o: *mut u64,
    pub w_4: *mut u64
}

#[repr(C)]
pub struct ProverKeyC {
    pub q_m_coeffs: *mut u64,
    pub q_m_evals: *mut u64,

    pub q_l_coeffs: *mut u64,
    pub q_l_evals: *mut u64,

    pub q_r_coeffs: *mut u64,
    pub q_r_evals: *mut u64,

    pub q_o_coeffs: *mut u64,
    pub q_o_evals: *mut u64,

    pub q_4_coeffs: *mut u64,
    pub q_4_evals: *mut u64,

    pub q_c_coeffs: *mut u64,
    pub q_c_evals: *mut u64,

    pub q_hl_coeffs: *mut u64,
    pub q_hl_evals: *mut u64,

    pub q_hr_coeffs: *mut u64,
    pub q_hr_evals: *mut u64,

    pub q_h4_coeffs: *mut u64,
    pub q_h4_evals: *mut u64,

    pub q_arith_coeffs: *mut u64,
    pub q_arith_evals: *mut u64,

    pub range_selector_coeffs: *mut u64,
    pub range_selector_evals: *mut u64,

    pub logic_selector_coeffs: *mut u64,
    pub logic_selector_evals: *mut u64,

    pub fixed_group_add_selector_coeffs: *mut u64,
    pub fixed_group_add_selector_evals: *mut u64,

    pub variable_group_add_selector_coeffs: *mut u64,
    pub variable_group_add_selector_evals: *mut u64,

    pub q_lookup_coeffs: *mut u64,
    pub q_lookup_evals: *mut u64,
    pub table1: *mut u64,
    pub table2: *mut u64,
    pub table3: *mut u64,
    pub table4: *mut u64,

    pub left_sigma_coeffs: *mut u64,
    pub left_sigma_evals: *mut u64,

    pub right_sigma_coeffs: *mut u64,
    pub right_sigma_evals: *mut u64,

    pub out_sigma_coeffs: *mut u64,
    pub out_sigma_evals: *mut u64,

    pub fourth_sigma_coeffs: *mut u64,
    pub fourth_sigma_evals: *mut u64,

    pub linear_evaluations: *mut u64,

    pub v_h_coset_8n: *mut u64
}

#[repr(C)]
pub struct CommitKeyC {
    pub powers_of_g: *const u64,
    pub powers_of_gamma_g: *const u64
}

#[repr(C)]
pub struct CommitmentC {
    pub x: [u64; 6],
    pub y: [u64; 6]
}

extern  "C"{
    pub fn gen_proof(circuit: CircuitC, pk: ProverKeyC, ck: CommitKeyC) -> ProofC;
}