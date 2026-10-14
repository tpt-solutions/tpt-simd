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
  help     show this text
";

fn main() -> ExitCode {
    let task = env::args().nth(1).unwrap_or_else(|| "help".to_owned());
    let result = match task.as_str() {
        "ci" => ci(),
        "no-std" => no_std(),
        "msrv" => msrv(),
        "cpu" => cpu(),
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
    use super::NO_STD_TARGETS;

    /// The `no-std` CI job in `.github/workflows/ci.yml` must build the same targets as `cargo xtask no-std`.
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
