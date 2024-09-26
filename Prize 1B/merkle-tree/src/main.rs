use ark_bls12_381::{Bls12_381, Fr};
use ark_crypto_primitives::commitment::blake2s::Commitment;
use ark_ed_on_bls12_381::EdwardsParameters;
use ark_poly_commit::PolynomialCommitment;
use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use merkle_tree::HEIGHT;
use merkle_tree::{MerkleTree, MerkleTreeCircuit};
use plonk_core::commitment::KZG10;
use plonk_core::prelude::{
    verify_proof, Circuit, StandardComposer, VerifierData,
};
use plonk_core::proof_system::{Prover,Proof};
use plonk_hashing::poseidon::constants::PoseidonConstants;
use plonk_hashing::poseidon::poseidon_ref::NativeSpecRef;
use std::time::Instant;
use plonk_core::ProofC;
use plonk_core::util;
use ark_ff::Fp256;
use ark_bls12_381::FrParameters;

fn main() {
    let mut rng = test_rng();

    let param = PoseidonConstants::<Fr>::generate::<3>();

    // ==============================
    // first we build a merkle tree
    // ==============================

    let leaf_nodes = (0..1 << (HEIGHT - 1))
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();

    let tree = MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(
        &param,
        &leaf_nodes,
    );

    let index = rng.next_u32() % (1 << (HEIGHT - 1));
    let proof = tree.gen_proof(index as usize);
    let res = proof.verify(&param, &tree.root());

    // omitted: parameters too large
    // println!("generating merkle tree with parameter {:?}", param);

    //println!("merkle tree with height: {}:\n{}\n", HEIGHT, tree);
    //println!(
    //    "merkle proof for {}-th leaf: {}\n{}\n",
    //    index, leaf_nodes[index as usize], proof
    //);
    //println!("proof is valid: {}", res);

    // ==============================
    // next we generate the constraints for the tree
    // ==============================

    let mut composer = StandardComposer::<Fr, EdwardsParameters>::new();
    tree.gen_constraints(&mut composer, &param);

    composer.check_circuit_satisfied();

    // ==============================
    // last we generate the plonk proof
    // ==============================
    {
        // public parameters
        let size = 1 << (HEIGHT + 9);
        // let mut pp: ark_poly_commit::kzg10::UniversalParams<ark_ec::bls12::Bls12<ark_bls12_381::Parameters>> = KZG10::<Bls12_381>::setup(size, None, &mut rng).unwrap();
        let pp = KZG10::<Bls12_381>::setup(size, None, &mut rng).unwrap();
        let mut dummy_circuit = MerkleTreeCircuit {
            param: param.clone(),
            merkle_tree: tree,
        };

        // proprocessing
        let (pk, (vk, _pi_pos)) =
            dummy_circuit.compile::<KZG10<Bls12_381>>(&pp).unwrap();

        // proof generation
        let leaf_nodes = (0..1 << (HEIGHT - 1))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let tree = MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(
            &param,
            &leaf_nodes,
        );
        let mut real_circuit = MerkleTreeCircuit {
            param: param.clone(),
            merkle_tree: tree,
        };
        
        unsafe{         
            let (proofc, pi) = {
                
                let mut prover =
                    Prover::<Fr, EdwardsParameters, KZG10<Bls12_381>>::new(
                        b"Merkle tree",
                    );
                let start = Instant::now();
                real_circuit.gadget(prover.mut_cs()).unwrap();
                println!("gadget1 time is {:?}", start.elapsed());
                real_circuit
                    .gen_proof::<KZG10<Bls12_381>>(&pp, pk, b"Merkle tree")
                    .unwrap()
            };
            let now = std::time::Instant::now();      
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

            println!("The prove generation time is {:?}", now.elapsed());

            let verifier_data = VerifierData::new(vk, pi.clone());
            let res = verify_proof::<Fr, EdwardsParameters, KZG10<Bls12_381>>(
                &pp,
                verifier_data.key.clone(),
                &proof,
                &verifier_data.pi,
                b"Merkle tree",
            );

            println!("proof is verified: {}", res.is_ok());
    }
    }
}
