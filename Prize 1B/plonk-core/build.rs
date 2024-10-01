use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn assembly(
    file_vec: &mut Vec<PathBuf>,
    base_dir: &Path,
    _arch: &str,
    _is_msvc: bool,
) {
    file_vec.push(base_dir.join("assembly.S"));
    return;
}

fn main() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let plonk_utils_mont_cpp = manifest_dir.join("lib/PLONK/utils/mont");
    let plonk_utils_zkp_cpp = manifest_dir.join("lib/PLONK/utils/zkp");
    let caffe_cpp = manifest_dir.join("lib/caffe");
    let caffe_utils_cpp = manifest_dir.join("lib/caffe/utils");
    let plonk_src_cpp = manifest_dir.join("lib/PLONK/src");
    let plonk_src_cuda = manifest_dir.join("lib/PLONK/src");
    let plonk_utils_cuda = manifest_dir.join("lib/PLONK/utils");
    
    let manifest_dir_2 = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let blst_base_dir = manifest_dir_2.join("lib/blst");

    println!("Using blst source directory {}", blst_base_dir.display());
    let mut cc = cc::Build::new();

    let c_src_dir = blst_base_dir.join("src");
    println!("cargo:rerun-if-changed={}", c_src_dir.display());
    let mut file_vec = vec![c_src_dir.join("server.c")];

    assembly(
        &mut file_vec,
        &c_src_dir,
        &target_arch,
        cc.get_compiler().is_like_msvc(),
    );
    
    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-command-line-argument");
    cc.opt_level(2);
    cc.include("lib/blst/include");
    cc.files(&file_vec);
    cc.compile("blstaaaa");


    // 获取输出目录
    let out_dir = env::var("OUT_DIR").unwrap();

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=blstaaaa");
    println!("cargo:rerun-if-changed=lib/blst/include");

    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };
    if nvcc.is_ok() {
        let mut nvcc = cc::Build::new();
        nvcc.cuda(true)
            .debug(false)
            .no_default_flags(true)
            .flag("-Xcompiler").flag("-gdwarf-4")
            .opt_level(3)
            .flag("-std=c++17")
            .flag("-arch=sm_80");
            //.flag("-gencode").flag("arch=compute_70,code=sm_70")
            //.flag("--maxrregcount=128");
            //.flag("-t0");

        let cpp_files = vec![
            plonk_utils_mont_cpp,
            plonk_utils_zkp_cpp,
            caffe_cpp,
            caffe_utils_cpp,
            plonk_src_cpp,
        ].into_iter().flat_map(|dir| glob::glob(&format!("{}/**/*.cpp", dir.display())).unwrap())
        .filter_map(Result::ok)
        .collect::<Vec<PathBuf>>();

        let cuda_files = vec![
            plonk_src_cuda,
            plonk_utils_cuda,
        ].into_iter().flat_map(|dir| glob::glob(&format!("{}/**/*.cu", dir.display())).unwrap())
        .filter_map(Result::ok)
        .collect::<Vec<PathBuf>>();

        nvcc.include("lib/blst/include");
        nvcc.files(cpp_files);
        nvcc.files(cuda_files);
        nvcc.file("lib/hello.cu");

        nvcc.compile("libzprize");
        println!("cargo:rustc-link-lib=pthread");
    }
}

