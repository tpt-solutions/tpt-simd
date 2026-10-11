//! Reports which SIMD features this binary was compiled for and which the
//! running CPU supports. A mismatch means the portable or scalar path is in
//! use. Run with:
//!
//! ```text
//! cargo run -p tpt-simd --features std --example cpu_report
//! RUSTFLAGS="-C target-cpu=native" cargo run -p tpt-simd --features std --example cpu_report
//! ```

use tpt_simd::detect::Features;
use tpt_simd::width::{NATIVE_F32_LANES, NATIVE_VECTOR_BITS};

fn row(name: &str, compiled: bool, runtime: bool) {
    let state = match (compiled, runtime) {
        (true, true) => "compiled + available",
        (false, true) => "available, not compiled (portable path)",
        (true, false) => "compiled but NOT available on this CPU",
        (false, false) => "-",
    };
    println!("  {name:<8} {state}");
}

fn main() {
    let compiled = Features::compile_time();
    let runtime = Features::runtime();

    println!("tpt-simd CPU report");
    println!("  native vector width: {NATIVE_VECTOR_BITS} bits ({NATIVE_F32_LANES} x f32 lanes)");
    println!("  x86 / x86_64:");
    row("sse2", compiled.sse2, runtime.sse2);
    row("sse4.1", compiled.sse41, runtime.sse41);
    row("avx", compiled.avx, runtime.avx);
    row("avx2", compiled.avx2, runtime.avx2);
    row("fma", compiled.fma, runtime.fma);
    row("avx512f", compiled.avx512f, runtime.avx512f);
    println!("  aarch64:");
    row("neon", compiled.neon, runtime.neon);
    row("sve", compiled.sve, runtime.sve);
    println!("  riscv:");
    row("rvv", compiled.rvv, runtime.rvv);

    if compiled.avx2 && !runtime.avx2 {
        println!("\nwarning: built for AVX2 but this CPU does not report it; the binary may fault");
    }
}
