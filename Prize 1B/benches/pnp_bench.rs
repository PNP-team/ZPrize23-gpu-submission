use std::clone;

use ark_bls12_381::{Bls12_381, Fr};
use ark_ed_on_bls12_381::EdwardsParameters;
use ark_ff::Fp256;
use ark_bls12_381::FrParameters;
use ark_poly_commit::PolynomialCommitment;
use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use merkle_tree::HEIGHT;
use merkle_tree::{MerkleTree, MerkleTreeCircuit};
use plonk_core::commitment::KZG10;
use plonk_core::prelude::{verify_proof, Circuit, VerifierData};
use plonk_core::proof_system::{Prover,Proof};
use plonk_core::util;
use plonk_hashing::poseidon::constants::PoseidonConstants;
use plonk_hashing::poseidon::poseidon_ref::NativeSpecRef;
// repeating the prover and verifier for REPEAT times
const REPEAT: usize = 4;

fn build_tree<R: RngCore>(
    rng: &mut R,
    param: &PoseidonConstants<Fr>,
) -> MerkleTree<NativeSpecRef<Fr>> {
    let leaf_nodes = (0..1 << (HEIGHT - 1))
        .map(|_| Fr::rand(rng))
        .collect::<Vec<_>>();
    MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(param, &leaf_nodes)
}

fn main() {
    let mut rng = test_rng();

    let param = PoseidonConstants::<Fr>::generate::<3>();

    // public parameters
    let size = 1 << (HEIGHT + 9);
    let pp = KZG10::<Bls12_381>::setup(size, None, &mut rng).unwrap();

    let dummy_tree = build_tree(&mut rng, &param);
    let mut dummy_circuit = MerkleTreeCircuit {
        param: param.clone(),
        merkle_tree: dummy_tree,
    };

    // pre-processing
    let (mut pk, (vk, _pi_pos)) =
        dummy_circuit.compile::<KZG10<Bls12_381>>(&pp).unwrap();

    // trace generation
    let mut real_circuits = Vec::with_capacity(REPEAT);
    for _ in 0..REPEAT {
        let tree = build_tree(&mut rng, &param);
        let real_circuit = MerkleTreeCircuit {
            param: param.clone(),
            merkle_tree: tree,
        };
        real_circuits.push(real_circuit);
    }
    
    // proof generation
    let mut waste_time = std::time::Duration::new(0, 0);
    let now = std::time::Instant::now();
    println!("==============================");
    println!("Start generating {} proofs", REPEAT);
    let mut proof_and_pi_s = vec![];
    for i in 0..REPEAT {
        //time cost to clone pk should not count to proof generation
        let start = std::time::Instant::now();
        let pk_cloned = pk.clone();
        let clone_time = start.elapsed();
        waste_time += clone_time;
        println!("clone time: {:?}", clone_time);

        let (proofc, pi) = {

            let mut prover =
                Prover::<Fr, EdwardsParameters, KZG10<Bls12_381>>::new(
                    b"Merkle tree",
                );

            real_circuits[i].gadget(prover.mut_cs()).unwrap();
            real_circuits[i]
                .gen_proof_pnp::<KZG10<Bls12_381>>(&pp, pk_cloned, b"Merkle tree")
        };

        let proof: Proof<Fp256<FrParameters>, KZG10<Bls12_381>> = Proof {
            a_comm: util::to_commitment(proofc.a_comm, false),
            b_comm: util::to_commitment(proofc.b_comm, false),
            c_comm: util::to_commitment(proofc.c_comm, false),
            d_comm: util::to_commitment(proofc.d_comm, false),
            z_comm: util::to_commitment(proofc.z_comm, false),
            f_comm: util::to_commitment(proofc.f_comm, true),
            h_1_comm: util::to_commitment(proofc.h_1_comm, true),
            h_2_comm: util::to_commitment(proofc.h_2_comm, true),
            z_2_comm: util::to_commitment(proofc.z_2_comm, false),
            t_1_comm: util::to_commitment(proofc.t_1_comm, false),
            t_2_comm: util::to_commitment(proofc.t_2_comm, false),
            t_3_comm: util::to_commitment(proofc.t_3_comm, false),
            t_4_comm: util::to_commitment(proofc.t_4_comm, false),
            t_5_comm: util::to_commitment(proofc.t_5_comm, false),
            t_6_comm: util::to_commitment(proofc.t_6_comm, false),
            t_7_comm: util::to_commitment(proofc.t_7_comm, true),
            t_8_comm: util::to_commitment(proofc.t_8_comm, true),
            aw_opening: util::to_openproof(proofc.aw_opening, false),
            saw_opening: util::to_openproof(proofc.saw_opening, false),
            evaluations: util::to_proof_evaluations(proofc.evaluations)
        };

        proof_and_pi_s.push((proof, pi));
        println!("Proof {} is generated", i);
        println!("Time elapse: {:?}", now.elapsed()-clone_time);
    }
    println!("The total prove generation time is {:?}", now.elapsed()-waste_time);
    println!(
        "Aromatized cost for each proof is {:?}",
        (now.elapsed()-waste_time) / REPEAT as u32
    );
    println!("==============================");

    // proof verification
    let now = std::time::Instant::now();
    println!("Start verifying {} proofs", REPEAT);
    for i in 0..REPEAT {
        let (proof, pi) = &proof_and_pi_s[i];
        let verifier_data = VerifierData::new(vk.clone(), pi.clone());
        let res = verify_proof::<Fr, EdwardsParameters, KZG10<Bls12_381>>(
            &pp,
            verifier_data.key.clone(),
            &proof,
            &verifier_data.pi,
            b"Merkle tree",
        );
        println!("Proof {} is verified: {}", i, res.is_ok());
        println!("Time elapsed: {:?}", now.elapsed());
    }
    println!("The prove verification time is {:?}", now.elapsed());
    println!(
        "Aromatized cost for each proof is {:?}",
        now.elapsed() / REPEAT as u32
    );
    println!("==============================");
}
