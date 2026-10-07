//! Macro that stamps out the whole kernel set for one element type.

macro_rules! blas_impl {
    (
        $modname:ident, $t:ident, w = $w:expr, mr = $mr:expr, nr = $nr:expr,
        mc = $mc:expr, kc = $kc:expr, nc = $nc:expr, sqrt = $sqrt:path,
        kernel = $kernel:path
    ) => {
        pub(crate) mod $modname {
            //! Per-type implementation (see the crate docs).
            use crate::check::{check_mat, check_ws};

            const W: usize = $w;
            /// Microkernel rows.
            pub(crate) const MR: usize = $mr;
            /// Microkernel columns.
            pub(crate) const NR: usize = $nr;
            const MC: usize = $mc;
            const KC: usize = $kc;
            const NC: usize = $nc;

            #[inline(always)]
            fn reduce(mut acc: [$t; W]) -> $t {
                let mut w = W;
                while w > 1 {
                    w /= 2;
                    for i in 0..w {
                        acc[i] += acc[i + w];
                    }
                }
                acc[0]
            }

            pub(crate) fn axpy(alpha: $t, x: &[$t], y: &mut [$t]) {
                assert_eq!(x.len(), y.len(), "axpy: length mismatch ({} vs {})", x.len(), y.len());
                if alpha == 0.0 {
                    return;
                }
                for (yi, xi) in y.iter_mut().zip(x) {
                    *yi += alpha * *xi;
                }
            }

            pub(crate) fn scal(alpha: $t, x: &mut [$t]) {
                if alpha == 1.0 {
                    return;
                }
                if alpha == 0.0 {
                    x.fill(0.0);
                    return;
                }
                for xi in x.iter_mut() {
                    *xi *= alpha;
                }
            }

            pub(crate) fn dot(x: &[$t], y: &[$t]) -> $t {
                assert_eq!(x.len(), y.len(), "dot: length mismatch ({} vs {})", x.len(), y.len());
                let mut acc = [0.0 as $t; W];
                let xc = x.chunks_exact(W);
                let yc = y.chunks_exact(W);
                let (xt, yt) = (xc.remainder(), yc.remainder());
                for (a, b) in xc.zip(yc) {
                    for j in 0..W {
                        acc[j] += a[j] * b[j];
                    }
                }
                let mut tail = 0.0 as $t;
                for (a, b) in xt.iter().zip(yt) {
                    tail += *a * *b;
                }
                reduce(acc) + tail
            }

            pub(crate) fn nrm2(x: &[$t]) -> $t {
                let mut acc = [0.0 as $t; W];
                let xc = x.chunks_exact(W);
                let xt = xc.remainder();
                for a in xc {
                    for j in 0..W {
                        acc[j] += a[j] * a[j];
                    }
                }
                let mut tail = 0.0 as $t;
                for a in xt {
                    tail += *a * *a;
                }
                $sqrt(reduce(acc) + tail)
            }

            pub(crate) fn asum(x: &[$t]) -> $t {
                let mut acc = [0.0 as $t; W];
                let xc = x.chunks_exact(W);
                let xt = xc.remainder();
                for a in xc {
                    for j in 0..W {
                        acc[j] += a[j].abs();
                    }
                }
                let mut tail = 0.0 as $t;
                for a in xt {
                    tail += a.abs();
                }
                reduce(acc) + tail
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemv(
                m: usize, n: usize, alpha: $t, a: &[$t], lda: usize,
                x: &[$t], beta: $t, y: &mut [$t],
            ) {
                check_mat("gemv", "a", m, n, lda, a.len());
                assert_eq!(x.len(), n, "gemv: x has {} elements, expected n = {}", x.len(), n);
                assert_eq!(y.len(), m, "gemv: y has {} elements, expected m = {}", y.len(), m);
                scal(beta, y);
                if alpha == 0.0 || m == 0 || n == 0 {
                    return;
                }
                let mut j = 0;
                while j + 4 <= n {
                    let t0 = alpha * x[j];
                    let t1 = alpha * x[j + 1];
                    let t2 = alpha * x[j + 2];
                    let t3 = alpha * x[j + 3];
                    let a0 = &a[j * lda..][..m];
                    let a1 = &a[(j + 1) * lda..][..m];
                    let a2 = &a[(j + 2) * lda..][..m];
                    let a3 = &a[(j + 3) * lda..][..m];
                    for i in 0..m {
                        y[i] += ((a0[i] * t0 + a1[i] * t1) + a2[i] * t2) + a3[i] * t3;
                    }
                    j += 4;
                }
                while j < n {
                    let t = alpha * x[j];
                    let col = &a[j * lda..][..m];
                    for (yi, ai) in y.iter_mut().zip(col) {
                        *yi += *ai * t;
                    }
                    j += 1;
                }
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemv_t(
                m: usize, n: usize, alpha: $t, a: &[$t], lda: usize,
                x: &[$t], beta: $t, y: &mut [$t],
            ) {
                check_mat("gemv_t", "a", m, n, lda, a.len());
                assert_eq!(x.len(), m, "gemv_t: x has {} elements, expected m = {}", x.len(), m);
                assert_eq!(y.len(), n, "gemv_t: y has {} elements, expected n = {}", y.len(), n);
                if alpha == 0.0 || m == 0 {
                    scal(beta, y);
                    return;
                }
                for (j, yj) in y.iter_mut().enumerate() {
                    let d = alpha * dot(&a[j * lda..][..m], x);
                    *yj = if beta == 0.0 { d } else { d + beta * *yj };
                }
            }

            fn round_up(v: usize, to: usize) -> usize {
                v.div_ceil(to) * to
            }

            /// Workspace length (elements) needed by `gemm` for these sizes.
            pub(crate) fn gemm_workspace_len(m: usize, n: usize, k: usize) -> usize {
                if m == 0 || n == 0 || k == 0 {
                    return 0;
                }
                let kc = KC.min(k);
                let mc = MC.min(round_up(m, MR));
                let nc = NC.min(round_up(n, NR));
                mc * kc + kc * nc
            }

            fn pack_a(src: &[$t], ld: usize, mc: usize, kc: usize, dst: &mut [$t]) {
                for ir in (0..mc).step_by(MR) {
                    let mr = MR.min(mc - ir);
                    let panel = &mut dst[(ir / MR) * kc * MR..][..kc * MR];
                    for p in 0..kc {
                        let d = &mut panel[p * MR..][..MR];
                        let s = &src[ir + p * ld..][..mr];
                        d[..mr].copy_from_slice(s);
                        d[mr..].fill(0.0);
                    }
                }
            }

            fn pack_b(src: &[$t], ld: usize, kc: usize, nc: usize, dst: &mut [$t]) {
                for jr in (0..nc).step_by(NR) {
                    let nr = NR.min(nc - jr);
                    let panel = &mut dst[(jr / NR) * kc * NR..][..kc * NR];
                    for jj in 0..NR {
                        if jj < nr {
                            let col = &src[(jr + jj) * ld..][..kc];
                            for p in 0..kc {
                                panel[p * NR + jj] = col[p];
                            }
                        } else {
                            for p in 0..kc {
                                panel[p * NR + jj] = 0.0;
                            }
                        }
                    }
                }
            }

            /// `out[j*MR + i] = sum_p a[p*MR + i] * b[p*NR + j]`.
            #[inline(always)]
            fn micro(kc: usize, a: &[$t], b: &[$t], out: &mut [$t; MR * NR]) {
                assert!(a.len() >= kc * MR && b.len() >= kc * NR);
                #[cfg(all(
                    not(feature = "scalar-only"),
                    target_arch = "x86_64",
                    any(
                        all(target_feature = "avx2", target_feature = "fma"),
                        feature = "runtime-dispatch"
                    )
                ))]
                {
                    if crate::x86::available() {
                        // SAFETY: lengths asserted above; `out` has MR*NR
                        // elements; AVX2+FMA are enabled at compile time or
                        // were detected at runtime by `available()`.
                        unsafe { $kernel(kc, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };
                        return;
                    }
                }
                {
                    let mut acc = [[0.0 as $t; MR]; NR];
                    for p in 0..kc {
                        let ap = &a[p * MR..][..MR];
                        let bp = &b[p * NR..][..NR];
                        for j in 0..NR {
                            for i in 0..MR {
                                acc[j][i] += ap[i] * bp[j];
                            }
                        }
                    }
                    for j in 0..NR {
                        out[j * MR..][..MR].copy_from_slice(&acc[j]);
                    }
                }
            }

            #[allow(clippy::too_many_arguments)]
            fn macro_kernel(
                mc: usize, nc: usize, kc: usize, alpha: $t,
                ap: &[$t], bp: &[$t], c: &mut [$t], ldc: usize,
            ) {
                let mut tmp = [0.0 as $t; MR * NR];
                for jr in (0..nc).step_by(NR) {
                    let nr = NR.min(nc - jr);
                    let bpanel = &bp[(jr / NR) * kc * NR..][..kc * NR];
                    for ir in (0..mc).step_by(MR) {
                        let mr = MR.min(mc - ir);
                        let apanel = &ap[(ir / MR) * kc * MR..][..kc * MR];
                        micro(kc, apanel, bpanel, &mut tmp);
                        for jj in 0..nr {
                            let col = &mut c[(jr + jj) * ldc + ir..][..mr];
                            let t = &tmp[jj * MR..][..mr];
                            for i in 0..mr {
                                col[i] += alpha * t[i];
                            }
                        }
                    }
                }
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemm_with_workspace(
                m: usize, n: usize, k: usize, alpha: $t, a: &[$t], lda: usize,
                b: &[$t], ldb: usize, beta: $t, c: &mut [$t], ldc: usize, ws: &mut [$t],
            ) {
                check_mat("gemm", "a", m, k, lda, a.len());
                check_mat("gemm", "b", k, n, ldb, b.len());
                check_mat("gemm", "c", m, n, ldc, c.len());
                if m == 0 || n == 0 {
                    return;
                }
                if beta != 1.0 {
                    for j in 0..n {
                        scal(beta, &mut c[j * ldc..][..m]);
                    }
                }
                if alpha == 0.0 || k == 0 {
                    return;
                }
                check_ws("gemm", ws.len(), gemm_workspace_len(m, n, k));
                let kcap = KC.min(k);
                let mcap = MC.min(round_up(m, MR));
                let (ap, bp) = ws.split_at_mut(mcap * kcap);
                let mut jc = 0;
                while jc < n {
                    let nc = NC.min(n - jc);
                    let mut pc = 0;
                    while pc < k {
                        let kc = KC.min(k - pc);
                        pack_b(&b[pc + jc * ldb..], ldb, kc, nc, bp);
                        let mut ic = 0;
                        while ic < m {
                            let mc = MC.min(m - ic);
                            pack_a(&a[ic + pc * lda..], lda, mc, kc, ap);
                            macro_kernel(mc, nc, kc, alpha, ap, bp, &mut c[ic + jc * ldc..], ldc);
                            ic += mc;
                        }
                        pc += kc;
                    }
                    jc += nc;
                }
            }

            #[cfg(feature = "alloc")]
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemm(
                m: usize, n: usize, k: usize, alpha: $t, a: &[$t], lda: usize,
                b: &[$t], ldb: usize, beta: $t, c: &mut [$t], ldc: usize,
            ) {
                let mut ws = alloc::vec![0.0 as $t; gemm_workspace_len(m, n, k)];
                gemm_with_workspace(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, &mut ws);
            }

            /// Naive scalar reference implementations (left-to-right sums).
            pub(crate) mod reference {
                use crate::check::check_mat;

                pub(crate) fn axpy(alpha: $t, x: &[$t], y: &mut [$t]) {
                    assert_eq!(x.len(), y.len());
                    for i in 0..x.len() {
                        y[i] += alpha * x[i];
                    }
                }
                pub(crate) fn dot(x: &[$t], y: &[$t]) -> $t {
                    assert_eq!(x.len(), y.len());
                    let mut s = 0.0 as $t;
                    for i in 0..x.len() {
                        s += x[i] * y[i];
                    }
                    s
                }
                pub(crate) fn nrm2(x: &[$t]) -> $t {
                    $sqrt(dot(x, x))
                }
                pub(crate) fn asum(x: &[$t]) -> $t {
                    let mut s = 0.0 as $t;
                    for v in x {
                        s += v.abs();
                    }
                    s
                }
                #[allow(clippy::too_many_arguments)]
                pub(crate) fn gemv(
                    m: usize, n: usize, alpha: $t, a: &[$t], lda: usize,
                    x: &[$t], beta: $t, y: &mut [$t],
                ) {
                    check_mat("reference gemv", "a", m, n, lda, a.len());
                    for i in 0..m {
                        let mut s = 0.0 as $t;
                        for j in 0..n {
                            s += a[i + j * lda] * x[j];
                        }
                        y[i] = alpha * s + if beta == 0.0 { 0.0 } else { beta * y[i] };
                    }
                }
                #[allow(clippy::too_many_arguments)]
                pub(crate) fn gemv_t(
                    m: usize, n: usize, alpha: $t, a: &[$t], lda: usize,
                    x: &[$t], beta: $t, y: &mut [$t],
                ) {
                    check_mat("reference gemv_t", "a", m, n, lda, a.len());
                    for j in 0..n {
                        let mut s = 0.0 as $t;
                        for i in 0..m {
                            s += a[i + j * lda] * x[i];
                        }
                        y[j] = alpha * s + if beta == 0.0 { 0.0 } else { beta * y[j] };
                    }
                }
                #[allow(clippy::too_many_arguments)]
                pub(crate) fn gemm(
                    m: usize, n: usize, k: usize, alpha: $t, a: &[$t], lda: usize,
                    b: &[$t], ldb: usize, beta: $t, c: &mut [$t], ldc: usize,
                ) {
                    check_mat("reference gemm", "a", m, k, lda, a.len());
                    check_mat("reference gemm", "b", k, n, ldb, b.len());
                    check_mat("reference gemm", "c", m, n, ldc, c.len());
                    for j in 0..n {
                        for i in 0..m {
                            let mut s = 0.0 as $t;
                            for p in 0..k {
                                s += a[i + p * lda] * b[p + j * ldb];
                            }
                            c[i + j * ldc] =
                                alpha * s + if beta == 0.0 { 0.0 } else { beta * c[i + j * ldc] };
                        }
                    }
                }
            }
        }
    };
}
