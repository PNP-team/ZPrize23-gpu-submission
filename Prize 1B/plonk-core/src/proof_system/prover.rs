// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Prover-side of the PLONK Proving System

use crate::lookup::MultiSet;
use crate::{
    commitment::HomomorphicCommitment,
    constraint_system::{StandardComposer, Variable},
    error::{to_pc_error, Error},
    label_polynomial,
    proof_system::{
        linearisation_poly, proof::Proof, quotient_poly, ProverKey,
    },
    transcript::TranscriptProtocol
};
use crate::{gen_proof, CircuitC, ProofC, ProverKeyC, CommitKeyC};
use ark_ec::{ModelParameters, TEModelParameters};
use ark_ff::{BigInteger256, Fp256, Fp256Parameters, PrimeField, ToConstraintField};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain,
    UVPolynomial,
};
use ark_poly_commit;
use core::marker::PhantomData;
use itertools::izip;
use merlin::Transcript;
use std::time::Instant;
use hashbrown::HashMap;
use rayon::prelude::*;
use rayon::scope;
use std::sync::{Arc, Mutex};
/// Abstraction structure designed to construct a circuit and generate
/// [`Proof`]s for it.
pub struct Prover<F, P, PC>
where
    F: PrimeField,
    P: ModelParameters<BaseField = F>,
    PC: HomomorphicCommitment<F>
{
    /// Proving Key which is used to create proofs about a specific PLONK
    /// circuit.
    pub prover_key: Option<ProverKey<F>>,

    /// Circuit Description
    pub(crate) cs: StandardComposer<F, P>,

    /// Store the messages exchanged during the preprocessing stage.
    ///
    /// This is copied each time, we make a proof.
    pub preprocessed_transcript: Transcript,

    _phantom: PhantomData<PC>,
}
impl<F, P, PC> Prover<F, P, PC>
where
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
    PC: HomomorphicCommitment<F>,
{
    /// Creates a new `Prover` instance.
    pub fn new(label: &'static [u8]) -> Self {
        Self {
            prover_key: None,
            cs: StandardComposer::new(),
            preprocessed_transcript: Transcript::new(label),
            _phantom: PhantomData::<PC>,
        }
    }

    /// Creates a new `Prover` object with some expected size.
    pub fn with_expected_size(label: &'static [u8], size: usize) -> Self {
        Self {
            prover_key: None,
            cs: StandardComposer::with_expected_size(size),
            preprocessed_transcript: Transcript::new(label),
            _phantom: PhantomData::<PC>,
        }
    }

    /// Returns a mutable copy of the underlying [`StandardComposer`].
    pub fn mut_cs(&mut self) -> &mut StandardComposer<F, P> {
        &mut self.cs
    }

    /// Returns the smallest power of two needed for the curcuit.
    pub fn circuit_bound(&self) -> usize {
        self.cs.circuit_bound()
    }

    /// Preprocesses the underlying constraint system.
    pub fn preprocess(
        &mut self,
        commit_key: &PC::CommitterKey,
    ) -> Result<(), Error> {
        if self.prover_key.is_some() {
            return Err(Error::CircuitAlreadyPreprocessed);
        }
        let pk = self.cs.preprocess_prover(
            commit_key,
            &mut self.preprocessed_transcript,
            PhantomData::<PC>,
        )?;
        self.prover_key = Some(pk);
        Ok(())
    }

    /// Split `t(X)` poly into 8 n-sized polynomials.
    #[allow(clippy::type_complexity)] // NOTE: This is an ok type for internal use.
    fn split_tx_poly(
        &self,
        n: usize,
        t_x: &DensePolynomial<F>,
    ) -> [DensePolynomial<F>; 8] {
        let mut buf = t_x.coeffs.to_vec();
        buf.resize(n << 3, F::zero());

        [
            DensePolynomial::from_coefficients_vec(buf[0..n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[n..2 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[2 * n..3 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[3 * n..4 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[4 * n..5 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[5 * n..6 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[6 * n..7 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[7 * n..].to_vec()),
        ]
    }

    /// Convert variables to their actual witness values.
    fn to_scalars(&self, vars: &[Variable]) -> Vec<F> {
        vars.iter().map(|var| self.cs.variables[var]).collect()
    }

    /// Resets the witnesses in the prover object.
    ///
    /// This function is used when the user wants to make multiple proofs with
    /// the same circuit.
    pub fn clear_witness(&mut self) {
        self.cs = StandardComposer::new();
    }

    /// Clears all data in the [`Prover`] instance.
    ///
    /// This function is used when the user wants to use the same `Prover` to
    /// make a [`Proof`] regarding a different circuit.
    pub fn clear(&mut self) {
        self.clear_witness();
        self.prover_key = None;
        self.preprocessed_transcript = Transcript::new(b"plonk");
    }

    /// Keys the [`Transcript`] with additional seed information
    /// Wrapper around [`Transcript::append_message`].
    ///
    /// [`Transcript`]: merlin::Transcript
    /// [`Transcript::append_message`]: merlin::Transcript::append_message
    pub fn key_transcript(&mut self, label: &'static [u8], message: &[u8]) {
        self.preprocessed_transcript.append_message(label, message);
    }

    /// Creates a [`Proof]` that demonstrates that a circuit is satisfied.
    /// # Note
    /// If you intend to construct multiple [`Proof`]s with different witnesses,
    /// after calling this method, the user should then call
    /// [`Prover::clear_witness`].
    /// This is automatically done when [`Prover::prove`] is called.
    pub fn prove_with_preprocessed(
        &self,
        commit_key: &PC::CommitterKey,
        prover_key: &ProverKey<F>,
        _data: PhantomData<PC>,
    ) -> Result< Proof<F, PC>, Error> {
        let domain =
            GeneralEvaluationDomain::new(self.cs.circuit_bound()).ok_or(Error::InvalidEvalDomainSize {
                log_size_of_group: self.cs.circuit_bound().trailing_zeros(),
                adicity: <<F as ark_ff::FftField>::FftParams as ark_ff::FftParameters>::TWO_ADICITY,
            })?;
        let n = domain.size();

        // Since the caller is passing a pre-processed circuit
        // We assume that the Transcript has been seeded with the preprocessed
        // Commitments
        let mut transcript = self.preprocessed_transcript.clone();

        // Append Public Inputs to the transcript
        transcript.append(b"pi", self.cs.get_pi());

        // 1. Compute witness Polynomials
        //
        // Convert Variables to scalars padding them to the
        // correct domain size.
        let pad = vec![F::zero(); n - self.cs.w_l.len()];
        let w_l_scalar = &[&self.to_scalars(&self.cs.w_l)[..], &pad].concat();
        let w_r_scalar = &[&self.to_scalars(&self.cs.w_r)[..], &pad].concat();
        let w_o_scalar = &[&self.to_scalars(&self.cs.w_o)[..], &pad].concat();
        let w_4_scalar = &[&self.to_scalars(&self.cs.w_4)[..], &pad].concat();

        // Witnesses are now in evaluation form, convert them to coefficients
        // so that we may commit to them.
        let w_l_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(w_l_scalar));
        let w_r_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(w_r_scalar));
        let w_o_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(w_o_scalar));
        let w_4_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(w_4_scalar));

        let w_polys = [
            label_polynomial!(w_l_poly),
            label_polynomial!(w_r_poly),
            label_polynomial!(w_o_poly),
            label_polynomial!(w_4_poly),
        ];

        // Commit to witness polynomials.
        let (w_commits, w_rands) = PC::commit(commit_key, w_polys.iter(), None)
            .map_err(to_pc_error::<F, PC>)?;

        // Add witness polynomial commitments to transcript.
        transcript.append(b"w_l", w_commits[0].commitment());
        transcript.append(b"w_r", w_commits[1].commitment());
        transcript.append(b"w_o", w_commits[2].commitment());
        transcript.append(b"w_4", w_commits[3].commitment());

        // 2. Derive lookup polynomials

        // Generate table compression factor
        let zeta = transcript.challenge_scalar(b"zeta");
        transcript.append(b"zeta", &zeta);

        // Compress lookup table into vector of single elements
        let compressed_t_multiset = MultiSet::compress(
            &[
                prover_key.lookup.table_1.clone(),
                prover_key.lookup.table_2.clone(),
                prover_key.lookup.table_3.clone(),
                prover_key.lookup.table_4.clone(),
            ],
            zeta,
        );

        // Compute table poly
        let table_poly = DensePolynomial::from_coefficients_vec(
            domain.ifft(&compressed_t_multiset.0),
        );

        // Compute query table f
        // When q_lookup[i] is zero the wire value is replaced with a dummy
        //   value currently set as the first row of the public table
        // If q_lookup[i] is one the wire values are preserved
        // This ensures the ith element of the compressed query table
        //   is an element of the compressed lookup table even when
        //   q_lookup[i] is 0 so the lookup check will pass

        let q_lookup_pad = vec![F::zero(); n - self.cs.q_lookup.len()];
        let padded_q_lookup =
            &[self.cs.q_lookup.as_slice(), q_lookup_pad.as_slice()].concat();

        let mut f_scalars: Vec<MultiSet<F>> =
            vec![MultiSet::with_capacity(w_l_scalar.len()); 4];

        for (q_lookup, w_l, w_r, w_o, w_4) in izip!(
            padded_q_lookup,
            w_l_scalar,
            w_r_scalar,
            w_o_scalar,
            w_4_scalar,
        ) {
            if q_lookup.is_zero() {
                f_scalars[0].push(compressed_t_multiset.0[0]);
                f_scalars.iter_mut().skip(1).for_each(|f| f.push(F::zero()));
            } else {
                f_scalars[0].push(*w_l);
                f_scalars[1].push(*w_r);
                f_scalars[2].push(*w_o);
                f_scalars[3].push(*w_4);
            }
        }

        // Compress all wires into a single vector
        let compressed_f_multiset = MultiSet::compress(&f_scalars, zeta);

        // Compute query poly
        let f_poly = DensePolynomial::from_coefficients_vec(
            domain.ifft(&compressed_f_multiset.0),
        );

        // Add blinders to query polynomials
        // let f_poly = Self::add_blinder(&f_poly, n, 1);

        // Commit to query polynomial
        let (f_poly_commit, _) =
            PC::commit(commit_key, &[label_polynomial!(f_poly)], None)
                .map_err(to_pc_error::<F, PC>)?;

        // Add f_poly commitment to transcript
        transcript.append(b"f", f_poly_commit[0].commitment());

        // Compute s, as the sorted and concatenated version of f and t
        let (h_1, h_2) = compressed_t_multiset
            .combine_split(&compressed_f_multiset)
            .unwrap();

        // Compute h polys
        let h_1_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(&h_1.0));
        let h_2_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(&h_2.0));

        // Add blinders to h polynomials
        // let h_1_poly = Self::add_blinder(&h_1_poly, n, 1);
        // let h_2_poly = Self::add_blinder(&h_2_poly, n, 1);

        // Commit to h polys
        let (h_1_poly_commit, _) =
            PC::commit(commit_key, &[label_polynomial!(h_1_poly)], None)
                .map_err(to_pc_error::<F, PC>)?;
        let (h_2_poly_commit, _) =
            PC::commit(commit_key, &[label_polynomial!(h_2_poly)], None)
                .map_err(to_pc_error::<F, PC>)?;

        // Add h polynomials to transcript
        transcript.append(b"h1", h_1_poly_commit[0].commitment());
        transcript.append(b"h2", h_2_poly_commit[0].commitment());

        // 3. Compute permutation polynomial
        //
        // Compute permutation challenge `beta`.
        let beta = transcript.challenge_scalar(b"beta");
        transcript.append(b"beta", &beta);
        // Compute permutation challenge `gamma`.
        let gamma = transcript.challenge_scalar(b"gamma");
        transcript.append(b"gamma", &gamma);
        // Compute permutation challenge `delta`.
        let delta = transcript.challenge_scalar(b"delta");
        transcript.append(b"delta", &delta);

        // Compute permutation challenge `epsilon`.
        let epsilon = transcript.challenge_scalar(b"epsilon");
        transcript.append(b"epsilon", &epsilon);

        // Challenges must be different
        assert!(beta != gamma, "challenges must be different");
        assert!(beta != delta, "challenges must be different");
        assert!(beta != epsilon, "challenges must be different");
        assert!(gamma != delta, "challenges must be different");
        assert!(gamma != epsilon, "challenges must be different");
        assert!(delta != epsilon, "challenges must be different");

        let z_poly = self.cs.perm.compute_permutation_poly(
            &domain,
            (w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar),
            beta,
            gamma,
            (
                &prover_key.permutation.left_sigma.0,
                &prover_key.permutation.right_sigma.0,
                &prover_key.permutation.out_sigma.0,
                &prover_key.permutation.fourth_sigma.0,
            ),
        );

        // Commit to permutation polynomial.
        let (z_poly_commit, _) =
            PC::commit(commit_key, &[label_polynomial!(z_poly)], None)
                .map_err(to_pc_error::<F, PC>)?;

        // Add permutation polynomial commitment to transcript.
        transcript.append(b"z", z_poly_commit[0].commitment());

        // Compute mega permutation polynomial.
        // Compute lookup permutation poly
        let z_2_poly = DensePolynomial::from_coefficients_slice(
            &self.cs.perm.compute_lookup_permutation_poly(
                &domain,
                &compressed_f_multiset.0,
                &compressed_t_multiset.0,
                &h_1.0,
                &h_2.0,
                delta,
                epsilon,
            ),
        );

        // TODO: Find strategy for blinding lookups
        // Add blinder for lookup permutation poly
        // z_2_poly = Self::add_blinder(&z_2_poly, n, 2);

        // Commit to lookup permutation polynomial.
        let (z_2_poly_commit, _) =
            PC::commit(commit_key, &[label_polynomial!(z_2_poly)], None)
                .map_err(to_pc_error::<F, PC>)?;

        // 3. Compute public inputs polynomial.
        let pi_poly = self.cs.get_pi().into_dense_poly(n);

        // 4. Compute quotient polynomial
        //
        // Compute quotient challenge; `alpha`, and gate-specific separation
        // challenges.
        let alpha = transcript.challenge_scalar(b"alpha");
        transcript.append(b"alpha", &alpha);

        let range_sep_challenge =
            transcript.challenge_scalar(b"range separation challenge");
        transcript.append(b"range seperation challenge", &range_sep_challenge);

        let logic_sep_challenge =
            transcript.challenge_scalar(b"logic separation challenge");
        transcript.append(b"logic seperation challenge", &logic_sep_challenge);

        let fixed_base_sep_challenge =
            transcript.challenge_scalar(b"fixed base separation challenge");
        transcript.append(
            b"fixed base separation challenge",
            &fixed_base_sep_challenge,
        );

        let var_base_sep_challenge =
            transcript.challenge_scalar(b"variable base separation challenge");
        transcript.append(
            b"variable base separation challenge",
            &var_base_sep_challenge,
        );

        let lookup_sep_challenge =
            transcript.challenge_scalar(b"lookup separation challenge");
        transcript
            .append(b"lookup separation challenge", &lookup_sep_challenge);

        let t_poly = quotient_poly::compute::<F, P>(
            &domain,
            prover_key,
            &z_poly,
            &z_2_poly,
            &w_l_poly,
            &w_r_poly,
            &w_o_poly,
            &w_4_poly,
            &pi_poly,
            &f_poly,
            &table_poly,
            &h_1_poly,
            &h_2_poly,
            &alpha,
            &beta,
            &gamma,
            &delta,
            &epsilon,
            &zeta,
            &range_sep_challenge,
            &logic_sep_challenge,
            &fixed_base_sep_challenge,
            &var_base_sep_challenge,
            &lookup_sep_challenge,
        )?;

        let t_i_polys = self.split_tx_poly(n, &t_poly);
        // Commit to splitted quotient polynomial
        let (t_commits, _) = PC::commit(
            commit_key,
            &[
                label_polynomial!(t_i_polys[0]),
                label_polynomial!(t_i_polys[1]),
                label_polynomial!(t_i_polys[2]),
                label_polynomial!(t_i_polys[3]),
                label_polynomial!(t_i_polys[4]),
                label_polynomial!(t_i_polys[5]),
                label_polynomial!(t_i_polys[6]),
                label_polynomial!(t_i_polys[7]),
            ],
            None,
        )
        .map_err(to_pc_error::<F, PC>)?;

        // Add quotient polynomial commitments to transcript
        transcript.append(b"t_1", t_commits[0].commitment());
        transcript.append(b"t_2", t_commits[1].commitment());
        transcript.append(b"t_3", t_commits[2].commitment());
        transcript.append(b"t_4", t_commits[3].commitment());
        transcript.append(b"t_5", t_commits[4].commitment());
        transcript.append(b"t_6", t_commits[5].commitment());
        transcript.append(b"t_7", t_commits[6].commitment());
        transcript.append(b"t_8", t_commits[7].commitment());

        // 4. Compute linearisation polynomial
        //
        // Compute evaluation challenge; `z`.
        let z_challenge = transcript.challenge_scalar(b"z");
        transcript.append(b"z", &z_challenge);

        let (lin_poly, evaluations) = linearisation_poly::compute::<F, P>(
            &domain,
            prover_key,
            &alpha,
            &beta,
            &gamma,
            &delta,
            &epsilon,
            &zeta,
            &range_sep_challenge,
            &logic_sep_challenge,
            &fixed_base_sep_challenge,
            &var_base_sep_challenge,
            &lookup_sep_challenge,
            &z_challenge,
            &w_l_poly,
            &w_r_poly,
            &w_o_poly,
            &w_4_poly,
            &t_i_polys[0],
            &t_i_polys[1],
            &t_i_polys[2],
            &t_i_polys[3],
            &t_i_polys[4],
            &t_i_polys[5],
            &t_i_polys[6],
            &t_i_polys[7],
            &z_poly,
            &z_2_poly,
            &f_poly,
            &h_1_poly,
            &h_2_poly,
            &table_poly,
        )?;

        // Add evaluations to transcript.
        // First wire evals
        transcript.append(b"a_eval", &evaluations.wire_evals.a_eval);
        transcript.append(b"b_eval", &evaluations.wire_evals.b_eval);
        transcript.append(b"c_eval", &evaluations.wire_evals.c_eval);
        transcript.append(b"d_eval", &evaluations.wire_evals.d_eval);

        // Second permutation evals
        transcript
            .append(b"left_sig_eval", &evaluations.perm_evals.left_sigma_eval);
        transcript.append(
            b"right_sig_eval",
            &evaluations.perm_evals.right_sigma_eval,
        );
        transcript
            .append(b"out_sig_eval", &evaluations.perm_evals.out_sigma_eval);
        transcript
            .append(b"perm_eval", &evaluations.perm_evals.permutation_eval);

        // Third lookup evals
        transcript.append(b"f_eval", &evaluations.lookup_evals.f_eval);
        transcript
            .append(b"q_lookup_eval", &evaluations.lookup_evals.q_lookup_eval);
        transcript.append(
            b"lookup_perm_eval",
            &evaluations.lookup_evals.z2_next_eval,
        );
        transcript.append(b"h_1_eval", &evaluations.lookup_evals.h1_eval);
        transcript
            .append(b"h_1_next_eval", &evaluations.lookup_evals.h1_next_eval);
        transcript.append(b"h_2_eval", &evaluations.lookup_evals.h2_eval);

        // Third, all evals needed for custom gates
        evaluations
            .custom_evals
            .vals
            .iter()
            .for_each(|(label, eval)| {
                let static_label = Box::leak(label.to_owned().into_boxed_str());
                transcript.append(static_label.as_bytes(), eval);
            });

        // 5. Compute Openings using KZG10
        //
        // We merge the quotient polynomial using the `z_challenge` so the SRS
        // is linear in the circuit size `n`

        // Compute aggregate witness to polynomials evaluated at the evaluation
        // challenge `z`
        let aw_challenge: F = transcript.challenge_scalar(b"aggregate_witness");

        // XXX: The quotient polynomials is used here and then in the
        // opening poly. It is being left in for now but it may not
        // be necessary. Warrants further investigation.
        // Ditto with the out_sigma poly.
        let aw_polys = [
            label_polynomial!(lin_poly),
            label_polynomial!(prover_key.permutation.left_sigma.0.clone()),
            label_polynomial!(prover_key.permutation.right_sigma.0.clone()),
            label_polynomial!(prover_key.permutation.out_sigma.0.clone()),
            label_polynomial!(f_poly),
            label_polynomial!(h_2_poly),
            label_polynomial!(table_poly),
        ];

        let (aw_commits, aw_rands) = PC::commit(commit_key, &aw_polys, None)
            .map_err(to_pc_error::<F, PC>)?;

        let aw_opening = PC::open(
            commit_key,
            aw_polys.iter().chain(w_polys.iter()),
            aw_commits.iter().chain(w_commits.iter()),
            &z_challenge,
            aw_challenge,
            aw_rands.iter().chain(w_rands.iter()),
            None,
        )
        .map_err(to_pc_error::<F, PC>)?;

        let saw_challenge: F =
            transcript.challenge_scalar(b"aggregate_witness");

        let saw_polys = [
            label_polynomial!(z_poly),
            label_polynomial!(w_l_poly),
            label_polynomial!(w_r_poly),
            label_polynomial!(w_4_poly),
            label_polynomial!(h_1_poly),
            label_polynomial!(z_2_poly),
            label_polynomial!(table_poly),
        ];

        let (saw_commits, saw_rands) = PC::commit(commit_key, &saw_polys, None)
            .map_err(to_pc_error::<F, PC>)?;

        let saw_opening = PC::open(
            commit_key,
            &saw_polys,
            &saw_commits,
            &(z_challenge * domain.element(1)),
            saw_challenge,
            &saw_rands,
            None,
        )
        .map_err(to_pc_error::<F, PC>)?;

        Ok(Proof {
            a_comm: w_commits[0].commitment().clone(),
            b_comm: w_commits[1].commitment().clone(),
            c_comm: w_commits[2].commitment().clone(),
            d_comm: w_commits[3].commitment().clone(),
            z_comm: saw_commits[0].commitment().clone(),
            f_comm: f_poly_commit[0].commitment().clone(),
            h_1_comm: h_1_poly_commit[0].commitment().clone(),
            h_2_comm: h_2_poly_commit[0].commitment().clone(),
            z_2_comm: z_2_poly_commit[0].commitment().clone(),
            t_1_comm: t_commits[0].commitment().clone(),
            t_2_comm: t_commits[1].commitment().clone(),
            t_3_comm: t_commits[2].commitment().clone(),
            t_4_comm: t_commits[3].commitment().clone(),
            t_5_comm: t_commits[4].commitment().clone(),
            t_6_comm: t_commits[5].commitment().clone(),
            t_7_comm: t_commits[6].commitment().clone(),
            t_8_comm: t_commits[7].commitment().clone(),
            aw_opening,
            saw_opening,
            evaluations,
        })
    }

    /// Proves a circuit is satisfied, then clears the witness variables
    /// If the circuit is not pre-processed, then the preprocessed circuit will
    /// also be computed.
    pub fn prove(
        &mut self,
        commit_key: &PC::CommitterKey,
    ) -> Result<Proof<F, PC>, Error> {
        if self.prover_key.is_none() {
            // Preprocess circuit and store preprocessed circuit and transcript
            // in the Prover.
            self.prover_key = Some(self.cs.preprocess_prover(
                commit_key,
                &mut self.preprocessed_transcript,
                PhantomData::<PC>,
            )?);
        }

        let prover_key = self.prover_key.as_ref().unwrap();
        let proof = self.prove_with_preprocessed(
            commit_key,
            prover_key,
            PhantomData::<PC>,
        )?;

        // Clear witness and reset composer variables
        self.clear_witness();

        Ok(proof)
    }

    /// proving with pnp optimizations
    pub fn prove_pnp(
        &mut self,
        commit_key: &ark_poly_commit::kzg10::UniversalParams<ark_ec::bls12::Bls12<ark_bls12_381::Parameters>>,
    ) -> ProofC {

        unsafe{
            let now = Instant::now();
            let powers_of_g_: Vec<_> = commit_key.powers_of_g
                                                        .iter()
                                                        .map(|fixed_row| (fixed_row.x.0, fixed_row.y.0))
                                                        .collect();
            let powers_of_gamma_g_vec: Vec<_> = (0..2).map(|i| commit_key.powers_of_gamma_g[&i]).collect();
            let powers_of_gamma_g_: Vec<_> = powers_of_gamma_g_vec
                                                        .iter()
                                                        .map(|fixed_row| (fixed_row.x.0, fixed_row.y.0))
                                                        .collect();
            
            let powers_of_g = &*(&powers_of_g_ as *const _ as *const Vec<u64>);
            let powers_of_gamma_g = &*(&powers_of_gamma_g_ as *const _ as *const Vec<u64>);

            let mut n:u64 = self.cs.n.try_into().unwrap();

            let mut lookup_size:u64 = self.cs.lookup_table.size().try_into().unwrap();

            let mut intended_pi_pos:u64 = self.cs.intended_pi_pos[0].try_into().unwrap();

            let cs_q_lookup = &mut *(&mut self.cs.q_lookup  as *mut _ as *mut Vec<u64>);

            let mut key_value_pairs:&mut Vec<F> = &mut self.cs.public_inputs.values.iter()
                .map(|(&_, &value)| value)
                .collect();
            let mut cs_p = key_value_pairs[0].into_repr().into();
            let mut cs_pi = &mut *(&mut cs_p as *mut _ as *mut Vec<u64>);

            let mut w_l_scalars = &mut self.cs.w_l; 
            let mut w_r_scalars = &mut self.cs.w_r;
            let mut w_o_scalars = &mut self.cs.w_o;
            let mut w_4_scalars = &mut self.cs.w_4;

            let cs_variables = &mut self.cs.variables;

            let mut w_l1 = (&mut Prover::<F, P, PC>::to_scalars_mut(w_l_scalars, cs_variables) );
            let mut w_r1 = (&mut Prover::<F, P, PC>::to_scalars_mut(w_r_scalars, cs_variables) );
            let mut w_o1 = (&mut Prover::<F, P, PC>::to_scalars_mut(w_o_scalars, cs_variables) );
            let mut w_41 = (&mut Prover::<F, P, PC>::to_scalars_mut(w_4_scalars, cs_variables) );

            // let mut w_l1 = Vec::new();
            // let mut w_r1 = Vec::new();
            // let mut w_o1 = Vec::new();
            // let mut w_41 = Vec::new();

            // scope(|s| {
            //     s.spawn(|_| {
            //         w_l1 = Prover::<F, P, PC>::to_scalars_mut(w_l_scalars, cs_variables);
            //     });
            //     s.spawn(|_| {
            //         w_r1 = Prover::<F, P, PC>::to_scalars_mut(w_r_scalars, cs_variables);
            //     });
            //     s.spawn(|_| {
            //         w_o1 = Prover::<F, P, PC>::to_scalars_mut(w_o_scalars, cs_variables);
            //     });
            //     s.spawn(|_| {
            //         w_41 = Prover::<F, P, PC>::to_scalars_mut(w_4_scalars, cs_variables);
            //     });
            // });

            let (h,mut w_l,t) = w_l1.align_to_mut::<u64>();
            let (h,mut w_r,t) = w_r1.align_to_mut::<u64>();
            let (h,mut w_o,t) = w_o1.align_to_mut::<u64>();
            let (h,mut w_4,t) = w_41.align_to_mut::<u64>();

            let prover_key: &mut ProverKey<F> = self.prover_key.as_mut().unwrap();
            let q_m_coeffs = &mut *(&mut prover_key.arithmetic.q_m.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_m_evals = &mut *(&mut prover_key.arithmetic.q_m.1.evals  as *mut _ as *mut Vec<u64>);

            let q_l_coeffs = &mut *(&mut prover_key.arithmetic.q_l.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_l_evals = &mut *(&mut prover_key.arithmetic.q_l.1.evals  as *mut _ as *mut Vec<u64>);

            let q_r_coeffs = &mut *(&mut prover_key.arithmetic.q_r.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_r_evals = &mut *(&mut prover_key.arithmetic.q_r.1.evals  as *mut _ as *mut Vec<u64>);

            let q_o_coeffs = &mut *(&mut prover_key.arithmetic.q_o.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_o_evals = &mut *(&mut prover_key.arithmetic.q_o.1.evals  as *mut _ as *mut Vec<u64>);

            let q_4_coeffs = &mut *(&mut prover_key.arithmetic.q_4.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_4_evals = &mut *(&mut prover_key.arithmetic.q_4.1.evals  as *mut _ as *mut Vec<u64>);

            let q_c_coeffs = &mut *(&mut prover_key.arithmetic.q_c.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_c_evals = &mut *(&mut prover_key.arithmetic.q_c.1.evals  as *mut _ as *mut Vec<u64>);

            let q_hl_coeffs = &mut *(&mut prover_key.arithmetic.q_hl.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_hl_evals = &mut *(&mut prover_key.arithmetic.q_hl.1.evals  as *mut _ as *mut Vec<u64>);

            let q_hr_coeffs = &mut *(&mut prover_key.arithmetic.q_hr.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_hr_evals = &mut *(&mut prover_key.arithmetic.q_hr.1.evals  as *mut _ as *mut Vec<u64>);

            let q_h4_coeffs = &mut *(&mut prover_key.arithmetic.q_h4.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_h4_evals = &mut *(&mut prover_key.arithmetic.q_h4.1.evals  as *mut _ as *mut Vec<u64>);

            let q_arith_coeffs = &mut *(&mut prover_key.arithmetic.q_arith.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_arith_evals = &mut *(&mut prover_key.arithmetic.q_arith.1.evals  as *mut _ as *mut Vec<u64>);

            let range_selector_coeffs = &mut *(&mut prover_key.range_selector.0.coeffs  as *mut _ as *mut Vec<u64>);
            let range_selector_evals = &mut *(&mut prover_key.range_selector.1.evals  as *mut _ as *mut Vec<u64>);

            let logic_selector_coeffs = &mut *(&mut prover_key.logic_selector.0.coeffs  as *mut _ as *mut Vec<u64>);
            let logic_selector_evals = &mut *(&mut prover_key.logic_selector.1.evals  as *mut _ as *mut Vec<u64>);

            let fixed_group_add_selector_coeffs = &mut *(&mut prover_key.fixed_group_add_selector.0.coeffs  as *mut _ as *mut Vec<u64>);
            let fixed_group_add_selector_evals = &mut *(&mut prover_key.fixed_group_add_selector.1.evals  as *mut _ as *mut Vec<u64>);

            let variable_group_add_selector_coeffs = &mut *(&mut prover_key.variable_group_add_selector.0.coeffs  as *mut _ as *mut Vec<u64>);
            let variable_group_add_selector_evals = &mut *(&mut prover_key.variable_group_add_selector.1.evals  as *mut _ as *mut Vec<u64>);

            let q_lookup_coeffs = &mut *(&mut prover_key.lookup.q_lookup.0.coeffs  as *mut _ as *mut Vec<u64>);
            let q_lookup_evals = &mut *(&mut prover_key.lookup.q_lookup.1.evals  as *mut _ as *mut Vec<u64>);
            let table1 = &mut *(&mut prover_key.lookup.table_1.0  as *mut _ as *mut Vec<u64>);
            let table2 = &mut *(&mut prover_key.lookup.table_2.0  as *mut _ as *mut Vec<u64>);
            let table3 = &mut *(&mut prover_key.lookup.table_3.0  as *mut _ as *mut Vec<u64>);
            let table4 = &mut *(&mut prover_key.lookup.table_4.0  as *mut _ as *mut Vec<u64>);

            let left_sigma_coeffs = &mut *(&mut prover_key.permutation.left_sigma.0.coeffs  as *mut _ as *mut Vec<u64>);
            let left_sigma_evals = &mut *(&mut prover_key.permutation.left_sigma.1.evals  as *mut _ as *mut Vec<u64>);

            let right_sigma_coeffs = &mut *(&mut prover_key.permutation.right_sigma.0.coeffs  as *mut _ as *mut Vec<u64>);
            let right_sigma_evals = &mut *(&mut prover_key.permutation.right_sigma.1.evals  as *mut _ as *mut Vec<u64>);

            let out_sigma_coeffs = &mut *(&mut prover_key.permutation.out_sigma.0.coeffs  as *mut _ as *mut Vec<u64>);
            let out_sigma_evals = &mut *(&mut prover_key.permutation.out_sigma.1.evals  as *mut _ as *mut Vec<u64>);

            let fourth_sigma_coeffs = &mut *(&mut prover_key.permutation.fourth_sigma.0.coeffs  as *mut _ as *mut Vec<u64>);
            let fourth_sigma_evals = &mut *(&mut prover_key.permutation.fourth_sigma.1.evals  as *mut _ as *mut Vec<u64>);

            let linear_evaluations = &mut *(&mut prover_key.permutation.linear_evaluations.evals  as *mut _ as *mut Vec<u64>);

            let v_h_coset_8n = &mut *(&mut prover_key.v_h_coset_8n.evals  as *mut _ as *mut Vec<u64>);
            
            let mut circuit_c = CircuitC {
                n: n,
                lookup_len: lookup_size,
                intended_pi_pos: intended_pi_pos,
                q_lookup: cs_q_lookup.as_mut_ptr(),
                pi: cs_pi.as_mut_ptr(),
                w_l: w_l.as_mut_ptr(),
                w_r: w_r.as_mut_ptr(),
                w_o: w_o.as_mut_ptr(),
                w_4: w_4.as_mut_ptr()
            };

            let mut prover_key_c = ProverKeyC {
                q_m_coeffs: q_m_coeffs.as_mut_ptr(),
                q_m_evals: q_m_evals.as_mut_ptr(),

                q_l_coeffs: q_l_coeffs.as_mut_ptr(),
                q_l_evals: q_l_evals.as_mut_ptr(),

                q_r_coeffs: q_r_coeffs.as_mut_ptr(),
                q_r_evals: q_r_evals.as_mut_ptr(),

                q_o_coeffs: q_o_coeffs.as_mut_ptr(),
                q_o_evals: q_o_evals.as_mut_ptr(),

                q_4_coeffs: q_4_coeffs.as_mut_ptr(),
                q_4_evals: q_4_evals.as_mut_ptr(),

                q_c_coeffs: q_c_coeffs.as_mut_ptr(),
                q_c_evals: q_c_evals.as_mut_ptr(),

                q_hl_coeffs: q_hl_coeffs.as_mut_ptr(),
                q_hl_evals: q_hl_evals.as_mut_ptr(),

                q_hr_coeffs: q_hr_coeffs.as_mut_ptr(),
                q_hr_evals: q_hr_evals.as_mut_ptr(),

                q_h4_coeffs: q_h4_coeffs.as_mut_ptr(),
                q_h4_evals: q_h4_evals.as_mut_ptr(),

                q_arith_coeffs: q_arith_coeffs.as_mut_ptr(),
                q_arith_evals: q_arith_evals.as_mut_ptr(),

                range_selector_coeffs: range_selector_coeffs.as_mut_ptr(),
                range_selector_evals: range_selector_evals.as_mut_ptr(),

                logic_selector_coeffs: logic_selector_coeffs.as_mut_ptr(),
                logic_selector_evals: logic_selector_evals.as_mut_ptr(),

                fixed_group_add_selector_coeffs: fixed_group_add_selector_coeffs.as_mut_ptr(),
                fixed_group_add_selector_evals: fixed_group_add_selector_evals.as_mut_ptr(),

                variable_group_add_selector_coeffs: variable_group_add_selector_coeffs.as_mut_ptr(),
                variable_group_add_selector_evals: variable_group_add_selector_evals.as_mut_ptr(),

                q_lookup_coeffs: q_lookup_coeffs.as_mut_ptr(),
                q_lookup_evals: q_lookup_evals.as_mut_ptr(),
                table1: table1.as_mut_ptr(),
                table2: table2.as_mut_ptr(),
                table3: table3.as_mut_ptr(),
                table4: table4.as_mut_ptr(),
                left_sigma_coeffs: left_sigma_coeffs.as_mut_ptr(),
                left_sigma_evals: left_sigma_evals.as_mut_ptr(),
                right_sigma_coeffs: right_sigma_coeffs.as_mut_ptr(),
                right_sigma_evals: right_sigma_evals.as_mut_ptr(),
                out_sigma_coeffs: out_sigma_coeffs.as_mut_ptr(),
                out_sigma_evals: out_sigma_evals.as_mut_ptr(),
                fourth_sigma_coeffs: fourth_sigma_coeffs.as_mut_ptr(),
                fourth_sigma_evals: fourth_sigma_evals.as_mut_ptr(),
                linear_evaluations: linear_evaluations.as_mut_ptr(),
                v_h_coset_8n: v_h_coset_8n.as_mut_ptr(),
            };
            let proofc = gen_proof(circuit_c, prover_key_c, CommitKeyC { powers_of_g: (powers_of_g.as_ptr()), powers_of_gamma_g: (powers_of_gamma_g.as_ptr()) });
            println!("prove time is {:?}", now.elapsed());
            proofc

        }
    }

    fn to_scalars_mut(vars: &Vec<Variable>, cs_variables: &HashMap<Variable, F>) -> Vec<F> {
        vars.par_iter().map(|var| cs_variables[var]).collect()
    }
}



impl<F, P, PC> Default for Prover<F, P, PC>
where
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
    PC: HomomorphicCommitment<F>,
{
    #[inline]
    fn default() -> Self {
        Prover::new(b"plonk")
    }
}
