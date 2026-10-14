//! Developer tasks for the tpt-simd workspace, run as `cargo xtask <task>`.
//!
//! The tasks mirror the CI jobs in `.github/workflows/ci.yml`, so a passing
//! `cargo xtask ci` on your machine is a good predictor of a green build.
//! This crate has no dependencies and runs the same on Windows, macOS and Linux.

use std::env;
use std::process::{Command, ExitCode};

/// Minimum supported Rust version (`rust-version` in the workspace manifest).
const MSRV: &str = "1.95";

/// Bare-metal and wasm targets used by the `no-std` CI job.
const NO_STD_TARGETS: &[&str] = &[
    "thumbv7em-none-eabihf",
    "wasm32-unknown-unknown",
    "riscv32imc-unknown-none-elf",
    "aarch64-unknown-none-softfloat",
];

const USAGE: &str = "\
cargo xtask <task>

Tasks:
  ci       fmt, clippy, docs and tests (std and scalar-only), as in the CI lint and test jobs
  no-std   no_std builds for the bare-metal and wasm targets installed with rustup
  msrv     cargo check on the minimum supported toolchain (1.95)
  cpu      print the SIMD features this build and CPU support (examples/cpu_report)
  asm      check that the hot kernels still contain their SIMD instructions (needs cargo-show-asm)
  help     show this text
";

/// Hot kernels checked by `cargo xtask asm`. Each symbol is one that `cargo asm` can emit from
/// the library (not inlined away), and each minimum is a floor below the count measured when the
/// check was added. Dropping a SIMD path makes the count fall below the floor.
struct AsmKernel {
    krate: &'static str,
    symbol: &'static str,
    /// Minimum count of instructions that use an xmm/ymm/zmm register.
    min_vector: usize,
    /// Minimum count of fused multiply-add instructions (0 when the kernel has none).
    min_fma: usize,
}

const ASM_KERNELS: &[AsmKernel] = &[
    AsmKernel {
        krate: "tpt-simd-blas",
        symbol: "tpt_simd_blas::single::gemm_with_workspace",
        min_vector: 140,
        min_fma: 18,
    },
    AsmKernel {
        krate: "tpt-simd-convolve",
        symbol: "convolve_1d_f32",
        min_vector: 49,
        min_fma: 10,
    },
    // The explicit Q15 FIR path is AVX-512 only; under the AVX2 target this relies on
    // autovectorisation, so the floor is low. The AVX-512 path is checked by the CI avx512 job.
    AsmKernel {
        krate: "tpt-simd-convolve",
        symbol: "fir_filter_i16",
        min_vector: 2,
        min_fma: 0,
    },
    AsmKernel {
        krate: "tpt-simd-dot",
        symbol: "x86::dot_i16",
        min_vector: 40,
        min_fma: 0,
    },
];

fn main() -> ExitCode {
    let task = env::args().nth(1).unwrap_or_else(|| "help".to_owned());
    let result = match task.as_str() {
        "ci" => ci(),
        "no-std" => no_std(),
        "msrv" => msrv(),
        "cpu" => cpu(),
        "asm" => asm(),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown task `{other}`\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("xtask: {message}");
            ExitCode::FAILURE
        }
    }
}

fn ci() -> Result<(), String> {
    run(cargo().args(["fmt", "--all", "--check"]))?;
    run(cargo().args([
        "clippy",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ]))?;
    run(cargo()
        .args(["doc", "--workspace", "--no-deps"])
        .env("RUSTDOCFLAGS", "-D warnings"))?;
    run(cargo().args(["test", "--workspace", "--features", "std"]))?;
    run(cargo().args(["test", "--workspace", "--features", "std,scalar-only"]))?;
    println!("xtask ci: all checks passed");
    Ok(())
}

fn no_std() -> Result<(), String> {
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map_err(|e| format!("could not run rustup: {e}"))?;
    let installed = String::from_utf8_lossy(&installed.stdout);

    for target in NO_STD_TARGETS {
        if !installed.lines().any(|line| line.trim() == *target) {
            println!("xtask no-std: skipping {target} (install with `rustup target add {target}`)");
            continue;
        }
        // Same arguments as the `no-std` job in ci.yml: default features off, and
        // `bench-vs` excluded because its dev-dependencies turn `std` on for the crates it uses.
        run(cargo().args([
            "build",
            "--workspace",
            "--no-default-features",
            "--exclude",
            "tpt-simd-testutil",
            "--exclude",
            "xtask",
            "--exclude",
            "bench-vs",
            "--target",
            target,
        ]))?;
    }
    Ok(())
}

fn msrv() -> Result<(), String> {
    // `cargo +<toolchain>` only works when rustup is the entry point, so go through `rustup run`.
    run(Command::new("rustup").args([
        "run",
        MSRV,
        "cargo",
        "check",
        "--workspace",
        "--all-targets",
    ]))
}

fn cpu() -> Result<(), String> {
    run(cargo().args([
        "run",
        "--quiet",
        "-p",
        "tpt-simd",
        "--features",
        "std",
        "--example",
        "cpu_report",
    ]))
}

/// Instruction counts for one function, read from `cargo asm` output.
#[derive(Debug, Default, PartialEq)]
struct AsmCounts {
    instructions: usize,
    vector: usize,
    fma: usize,
}

/// Count instructions in `cargo asm` output. Directives (`.section`, `.cfi_*`), labels and
/// blank lines are skipped; a line is an instruction when it starts with whitespace and a mnemonic.
fn count_asm(text: &str) -> AsmCounts {
    let mut counts = AsmCounts::default();
    for line in text.lines() {
        if !line.starts_with(|c: char| c.is_whitespace()) {
            continue;
        }
        let mut words = line.split_whitespace();
        let Some(mnemonic) = words.next() else {
            continue;
        };
        if mnemonic.starts_with('.') || mnemonic.starts_with('#') || mnemonic.ends_with(':') {
            continue;
        }
        counts.instructions += 1;
        if line.contains("xmm") || line.contains("ymm") || line.contains("zmm") {
            counts.vector += 1;
        }
        if ["vfmadd", "vfmsub", "vfnmadd", "vfnmsub"]
            .iter()
            .any(|prefix| mnemonic.starts_with(prefix))
        {
            counts.fma += 1;
        }
    }
    counts
}

/// Check the hot kernels' SIMD instruction counts with `cargo asm` (needs cargo-show-asm).
///
/// Built with `-C target-cpu=x86-64-v3`, so the AVX2+FMA paths are selected. On other hosts the
/// check is skipped, because the symbols and the codegen are different.
fn asm() -> Result<(), String> {
    if !cfg!(target_arch = "x86_64") {
        println!("xtask asm: skipped (the checks are for x86_64 codegen)");
        return Ok(());
    }
    let installed = Command::new("cargo")
        .args(["asm", "--help"])
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false);
    if !installed {
        return Err(
            "cargo-show-asm is not installed (`cargo install cargo-show-asm --locked`)".into(),
        );
    }

    let mut failures = Vec::new();
    for kernel in ASM_KERNELS {
        let output = Command::new("cargo")
            .args([
                "asm",
                "-p",
                kernel.krate,
                "--lib",
                "--release",
                kernel.symbol,
            ])
            .env("RUSTFLAGS", "-C target-cpu=x86-64-v3")
            .output()
            .map_err(|e| format!("could not run cargo asm: {e}"))?;
        let text = String::from_utf8_lossy(&output.stdout);
        let counts = count_asm(&text);
        let ok = counts.vector >= kernel.min_vector && counts.fma >= kernel.min_fma;
        println!(
            "{:<40} instructions {:>5}  vector {:>4} (min {:>3})  fma {:>3} (min {:>2})  {}",
            kernel.symbol,
            counts.instructions,
            counts.vector,
            kernel.min_vector,
            counts.fma,
            kernel.min_fma,
            if ok { "ok" } else { "FAIL" },
        );
        if !ok {
            failures.push(kernel.symbol);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "SIMD instruction count below the floor for: {}",
            failures.join(", ")
        ))
    }
}

/// A `cargo` command, using the same toolchain that launched xtask.
fn cargo() -> Command {
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    Command::new(cargo)
}

/// Runs a command with inherited stdio and turns a failure into an error that names it.
fn run(command: &mut Command) -> Result<(), String> {
    let description = format!("{command:?}");
    let status = command
        .status()
        .map_err(|e| format!("failed to start {description}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{description} exited with {status}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{NO_STD_TARGETS, count_asm};

    /// The `no-std` CI job in `.github/workflows/ci.yml` must build the same targets as `cargo xtask no-std`.
    #[test]
    fn count_asm_counts_vector_and_fma_instructions() {
        let text = "tpt_simd::f:\n\
            \t.cfi_startproc\n\
            \tvmovups ymm0, ymmword ptr [rdi]\n\
            \tvfmadd231ps ymm1, ymm0, ymm2\n\
            \tvfnmsub132ps xmm3, xmm1, xmm2\n\
            \tadd rax, 4\n\
            \tcall foo\n";
        let counts = count_asm(text);
        assert_eq!(counts.instructions, 5);
        assert_eq!(counts.vector, 3);
        assert_eq!(counts.fma, 2);
    }

    #[test]
    fn no_std_targets_match_ci_workflow() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../.github/workflows/ci.yml");
        let ci = std::fs::read_to_string(path).expect("read ci.yml");
        let job = ci
            .split("\n  no-std:\n")
            .nth(1)
            .and_then(|rest| rest.split("\n  aarch64:").next())
            .expect("no-std job in ci.yml");

        let mut ci_targets: Vec<&str> = job
            .lines()
            .filter_map(|line| line.trim().strip_prefix("- run: "))
            .filter_map(|cmd| {
                let mut words = cmd.split_whitespace();
                words.find(|w| *w == "--target")?;
                words.next()
            })
            .collect();
        let mut xtask_targets: Vec<&str> = NO_STD_TARGETS.to_vec();
        ci_targets.sort_unstable();
        xtask_targets.sort_unstable();
        assert_eq!(ci_targets, xtask_targets);
    }
}
