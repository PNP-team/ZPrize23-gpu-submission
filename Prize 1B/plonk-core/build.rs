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
        // let sfx = match _arch {
        //     "x86_64" => "x86_64",
        //     "aarch64" => "armv8",
        //     _ => "unknown",
        // };
        // let files =
        //     glob::glob(&format!("{}/elf/*-{}.s", base_dir.display(), sfx))
        //         .expect("unable to collect assembly files");
        // for file in files {
        //     file_vec.push(file.unwrap());
        // }


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

    let mut blst_base_dir = manifest_dir_2.join("lib/blst");

    println!("Using blst source directory {}", blst_base_dir.display());
    let mut cc = cc::Build::new();

    let c_src_dir = blst_base_dir.join("src");
    println!("cargo:rerun-if-changed={}", c_src_dir.display());
    let mut file_vec = vec![c_src_dir.join("server.c")];
    // let asm_files = vec![
    //     c_src_dir
    // ].into_iter().flat_map(|dir| glob::glob(&format!("{}/**/*.S", dir.display())).unwrap())
    //   .filter_map(Result::ok)
    //   .collect::<Vec<PathBuf>>();

    assembly(
        &mut file_vec,
        &c_src_dir,
        &target_arch,
        cc.get_compiler().is_like_msvc(),
    );
    // let mut file_vec = vec!["lib/blst/src/elf/sha256-x86_64.s", "lib/blst/src/elf/ctx_inverse_mod_384-x86_64.s",
    //                                     "lib/blst/src/elf/add_mod_384-x86_64.s", "lib/blst/src/elf/add_mod_384x384-x86_64.s",
    //                                     "lib/blst/src/elf/mulx_mont_384-x86_64.s", "lib/blst/src/elf/mulx_mont_256-x86_64.s",
    //                                     "lib/blst/src/elf/add_mod_256-x86_64.s", "lib/blst/src/elf/ct_inverse_mod_256-x86_64.s",
    //                                     "lib/blst/src/elf/div3w-x86_64.s", "lib/blst/src/elf/ct_is_square_mod_384-x86_64.s"];
    // println!("Enabling ADX support via `force-adx` feature");
    // cc.define("__ADX__", None);
    
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

    let mut nvcc = cc::Build::new();
    nvcc.cuda(true)
        .debug(false)
        //.opt_level(3)
        .no_default_flags(true)
        .flag("-Xcompiler").flag("-gdwarf-4")
        .opt_level(3)
        .flag("-std=c++17")
        .flag("-arch=sm_80");
        // .flag("-gencode").flag("arch=compute_70,code=sm_70")
        //.flag("--maxrregcount=128");
        //.flag("-t0");

    // if cfg!(feature = "quiet") {
    //     nvcc.flag("-diag-suppress=177"); // bug in the warning system.
    // }
    // nvcc.flag("-Xcompiler").flag("-Wno-unused-function");

    // Collect all .cpp and .cu files
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


    // if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
    //     nvcc.include(include);
    // }
    nvcc.include("lib/blst/include");
    nvcc.files(cpp_files);
    nvcc.files(cuda_files);
    nvcc.file("lib/hello.cu");
    
    // 编译库
    nvcc.compile("libzprize");
    println!("cargo:rustc-link-lib=pthread");

}

