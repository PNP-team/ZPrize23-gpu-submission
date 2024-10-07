# ZPrize23-Prize1b
## TL;DR
The following is PNP's GPU submission to [ZPrize2023-Prize1b](https://github.com/cysic-labs/ZPrize-23-Prize1/tree/main/Prize%201B) —— EtoE Proof Acceleration of a 15-level Poseidon Merkle-Tree.  

PNP is a team under CATS LAB, School of Cyber Science and Technology, Shandong University. We are committed to the implementation and acceleration of advanced cryptographic algorithms and protocols. Our research interests include zero-knowledge proofs and fully homomorphic encryption.

## Performance
Our performance test is conducted on officially provided device(AMD EPYC 75F3 + NVIDIA RTX 6000 Ada) and follows the [official benchmark](https://github.com/cysic-labs/ZPrize-23-Prize1/blob/main/Prize%201B/benches/zprize_bench.rs). The benchmark test time mainly consists of three parts, clone pk(we moved this part outside the timing scope because this is not required in a real proof generation, just for the security check of rust programming in benchmark), circuit gadget(the process of synthesizing circuit and obtaining witness, twice per proof) and proof generation. **What we accelerate is the process of proof generation.**  

The inputs we use are randomly generated finite field elements. We observed that the average time taken by the benchmark was basically between **31** and **32** seconds, with the gadget taking between **9** and **10** seconds, and the actual proof generation taking between **9.5** and **10.5** seconds.  

**HEIGHT=15 BENCHMARK**  

| full run     |clone pk | gadget(once)     | gen_proof     |
| ------- | ------- | ------- |------- |
| 31.933755031s | 11.356669964s | 9.514367318s   | 10.07100116s   |
| 31.584186511s | 11.114444828s | 9.406599309s   | 9.808210833s   |
| 31.420484675s  | 10.926322927s | 9.296232692s   | 9.965366536s   |
| 31.61689248s  | 10.941579919s | 9.357072084s   | 10.08814598s   |
## Platform requirements
All our development is based on x86_64 and linux operating system. GPU operators(e.g. NTT, MSM) in our library support all Nvidia's Volta<sup>+</sup> architecture. To be on the safe side, the end-to-end proof of tree height required by the competition needs $sm \geq 80$(Ampere<sup>+</sup>) and no less than 40GB video memory.  
## Building and running instructions
First, install [CUDA12](https://developer.nvidia.com/cuda-toolkit-archive) and [Rust](https://www.rust-lang.org/tools/install).  
To run the benchmark: 

```cargo bench --bench submission_bench```   

To test performance please run the source code direcly(removed clone pk). We recommend running the program in release mode instead of debug mode to achieve better performance.  

```cargo run --package merkle-tree --bin merkle-tree --release``` 

## Repository structure
Our implementation's structure is generally consistent with the [official harness repo](https://github.com/cysic-labs/ZPrize-23-Prize1/tree/main/Prize%201B). All new additions are in one directory `/plonk-core/lib`. In other files, we only added some data type conversions and modified the access permissions of some class members. 
* benches - benchmark codes include the zk-Garage's PLONK protocol, the official baseline offered by cysic and our submission.
* examples - some exmaple circuits of the plonk arithmetization.
* merkle-tree - the front end, source codes for generating a Poseidon Merkle Tree and the corresponding contraints.
* plonk-book - a tutorial of PLONK.
* plonk-hashing - implementation of [Poseidon](https://eprint.iacr.org/2019/458.pdf).
* plonk-core
  * src - ZK-Garage's implementation of PLONK, we write FFIs in `lib.rs`.
  * lib - PNP's acceleration codes
    * blst -  CPU-side high-performance big-integer library.
    * caffe - a communication control logic between CPU and GPU side following [caffe](http://caffe.berkeleyvision.org/).
    * PLONK - our implementation of PLONK protocol, contains the complete proof generation process, the underlying operators and corresponding CUDA kernels.
   
## Optimizations 
In this part we'll give a high level overview of the optimizations in our solution. We designed an end-to-end ZKP system tailored for GPU for this competition, implementing various optimizations in both computation and storage.   

On the computation side, [ZPrize2022](https://www.zprize.io/blog/zprize-retrospective) focused on the key operators of zkSNARK: NTT and MSM. We integrated the GPU winner's implementation -- [sppark](https://github.com/supranational/sppark), as the foundation of our system. Besides, we developed several other operators required by PLONK, such as a compression kernel designed for lookup tables and a polynomial division kernel using a double-buffer method. Moreover, we re-implemented highly parallelized upper-level computations including the process of computing permutation polynomial.  

For an end-to-end program, the system's memory pressure far exceeds that of the calculation. In fact, each polynomial will occupy 128MB of memory under the given tree height. Our tests showed that the peak memory usage of the proof system reached 62.2GB. Throughout the protocol, the highest memory pressure occurs during the fourth step, computing the quotient polynomial. In our submission, we optimized the execution order for each module in this step to minimize the data that needs to remain on GPU. On the other hand, the high-performance operators we employed significantly reduced the system's computational complexity. However, the frequent kernel calls led to substantial interaction between the host and device, becoming a new bottleneck. Due to the fixed PCIe bandwidth of the device, we had to minimize memory transfer instances. In our submission, the twiddle factor used in NTT, the elliptic curve points for polynomial commitments, and all witness polynomials will remain on GPU during the entire process. Additionally, to manage memory more flexibly, we implemented a mini Caffe that can control dataflow, allowing us to dynamically load each part of the public key as needed and release them on-the-fly when no longer needed. Our optimization reduced the peak memory usage by at least 20GB.  

BTW, in our submission, the protocol remains fully serial. The repository is still under development, we will later add support for multiple CUDA streams to enable asynchronous execution of the protocol. For technical questions about our submission, please contact [@zlyber](https://github.com/zlyber).

