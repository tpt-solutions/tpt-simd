//! Macro that stamps out the complex kernel set for one precision.
//!
//! Complex data is held either as **split planes** (separate `re` and `im`
//! slices) or as **interleaved** `[T; 2]` pairs. Everything is built from the
//! real kernels of the matching real module (`$real`), so complex `gemm`
//! reuses the packed (and AVX2+FMA) real `gemm`.

macro_rules! complex_impl {
    ($modname:ident, $t:ident, real = $real:ident, w = $w:expr, sqrt = $sqrt:path) => {
        pub(crate) mod $modname {
            //! Per-precision complex implementation (see the crate docs).
            use crate::check::{check_mat, check_ws};

            const W: usize = $w;
            type C = [$t; 2];

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

            #[inline(always)]
            fn cmul(a: C, b: C) -> C {
                [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
            }

            const ZERO: C = [0.0, 0.0];
            const ONE: C = [1.0, 0.0];

            fn len_check(func: &str, a: usize, b: usize) {
                assert_eq!(a, b, "{func}: length mismatch ({a} vs {b})");
            }

            // ------------------------------------------------------------
            // Split-plane level 1
            // ------------------------------------------------------------

            pub(crate) fn axpy(alpha: C, xr: &[$t], xi: &[$t], yr: &mut [$t], yi: &mut [$t]) {
                len_check("axpy", xr.len(), xi.len());
                len_check("axpy", yr.len(), yi.len());
                len_check("axpy", xr.len(), yr.len());
                if alpha == ZERO {
                    return;
                }
                let (ar, ai) = (alpha[0], alpha[1]);
                for i in 0..xr.len() {
                    yr[i] += ar * xr[i] - ai * xi[i];
                    yi[i] += ar * xi[i] + ai * xr[i];
                }
            }

            pub(crate) fn scal(alpha: C, xr: &mut [$t], xi: &mut [$t]) {
                len_check("scal", xr.len(), xi.len());
                if alpha == ONE {
                    return;
                }
                if alpha == ZERO {
                    xr.fill(0.0);
                    xi.fill(0.0);
                    return;
                }
                let (ar, ai) = (alpha[0], alpha[1]);
                for i in 0..xr.len() {
                    let (r, m) = (xr[i], xi[i]);
                    xr[i] = ar * r - ai * m;
                    xi[i] = ar * m + ai * r;
                }
            }

            pub(crate) fn dot(conj: bool, xr: &[$t], xi: &[$t], yr: &[$t], yi: &[$t]) -> C {
                let f = if conj { "dotc" } else { "dotu" };
                len_check(f, xr.len(), xi.len());
                len_check(f, yr.len(), yi.len());
                len_check(f, xr.len(), yr.len());
                let (mut rr, mut ii, mut ri, mut ir) = ([0.0 as $t; W], [0.0 as $t; W], [0.0 as $t; W], [0.0 as $t; W]);
                let a = xr.chunks_exact(W);
                let b = xi.chunks_exact(W);
                let c = yr.chunks_exact(W);
                let d = yi.chunks_exact(W);
                let (ta, tb, tc, td) = (a.remainder(), b.remainder(), c.remainder(), d.remainder());
                for (((a, b), c), d) in a.zip(b).zip(c).zip(d) {
                    for j in 0..W {
                        rr[j] += a[j] * c[j];
                        ii[j] += b[j] * d[j];
                        ri[j] += a[j] * d[j];
                        ir[j] += b[j] * c[j];
                    }
                }
                let (mut trr, mut tii, mut tri, mut tir) = (0.0 as $t, 0.0 as $t, 0.0 as $t, 0.0 as $t);
                for j in 0..ta.len() {
                    trr += ta[j] * tc[j];
                    tii += tb[j] * td[j];
                    tri += ta[j] * td[j];
                    tir += tb[j] * tc[j];
                }
                let (rr, ii, ri, ir) = (reduce(rr) + trr, reduce(ii) + tii, reduce(ri) + tri, reduce(ir) + tir);
                if conj { [rr + ii, ri - ir] } else { [rr - ii, ri + ir] }
            }

            pub(crate) fn nrm2(xr: &[$t], xi: &[$t]) -> $t {
                len_check("nrm2", xr.len(), xi.len());
                let mut acc = [0.0 as $t; W];
                let a = xr.chunks_exact(W);
                let b = xi.chunks_exact(W);
                let (ta, tb) = (a.remainder(), b.remainder());
                for (a, b) in a.zip(b) {
                    for j in 0..W {
                        acc[j] += a[j] * a[j] + b[j] * b[j];
                    }
                }
                let mut tail = 0.0 as $t;
                for j in 0..ta.len() {
                    tail += ta[j] * ta[j] + tb[j] * tb[j];
                }
                $sqrt(reduce(acc) + tail)
            }

            pub(crate) fn asum(xr: &[$t], xi: &[$t]) -> $t {
                len_check("asum", xr.len(), xi.len());
                let mut acc = [0.0 as $t; W];
                let a = xr.chunks_exact(W);
                let b = xi.chunks_exact(W);
                let (ta, tb) = (a.remainder(), b.remainder());
                for (a, b) in a.zip(b) {
                    for j in 0..W {
                        acc[j] += a[j].abs() + b[j].abs();
                    }
                }
                let mut tail = 0.0 as $t;
                for j in 0..ta.len() {
                    tail += ta[j].abs() + tb[j].abs();
                }
                reduce(acc) + tail
            }

            /// `y = beta * y` with BLAS semantics (`beta == 0` overwrites).
            fn cscale(beta: C, yr: &mut [$t], yi: &mut [$t]) {
                scal(beta, yr, yi);
            }

            // ------------------------------------------------------------
            // Split-plane gemv
            // ------------------------------------------------------------

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemv(
                m: usize, n: usize, alpha: C, ar: &[$t], ai: &[$t], lda: usize,
                xr: &[$t], xi: &[$t], beta: C, yr: &mut [$t], yi: &mut [$t],
            ) {
                check_mat("gemv", "a(re)", m, n, lda, ar.len());
                check_mat("gemv", "a(im)", m, n, lda, ai.len());
                len_check("gemv x", xr.len(), n);
                len_check("gemv x", xi.len(), n);
                len_check("gemv y", yr.len(), m);
                len_check("gemv y", yi.len(), m);
                cscale(beta, yr, yi);
                if alpha == ZERO || m == 0 || n == 0 {
                    return;
                }
                let mut j = 0;
                while j + 2 <= n {
                    let t0 = cmul(alpha, [xr[j], xi[j]]);
                    let t1 = cmul(alpha, [xr[j + 1], xi[j + 1]]);
                    let (r0, i0) = (&ar[j * lda..][..m], &ai[j * lda..][..m]);
                    let (r1, i1) = (&ar[(j + 1) * lda..][..m], &ai[(j + 1) * lda..][..m]);
                    for i in 0..m {
                        yr[i] += (r0[i] * t0[0] - i0[i] * t0[1]) + (r1[i] * t1[0] - i1[i] * t1[1]);
                        yi[i] += (r0[i] * t0[1] + i0[i] * t0[0]) + (r1[i] * t1[1] + i1[i] * t1[0]);
                    }
                    j += 2;
                }
                if j < n {
                    let t = cmul(alpha, [xr[j], xi[j]]);
                    let (r0, i0) = (&ar[j * lda..][..m], &ai[j * lda..][..m]);
                    for i in 0..m {
                        yr[i] += r0[i] * t[0] - i0[i] * t[1];
                        yi[i] += r0[i] * t[1] + i0[i] * t[0];
                    }
                }
            }

            /// `y = alpha * op(A) x + beta * y`, `op` = transpose or conjugate transpose.
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemv_t(
                conj: bool, m: usize, n: usize, alpha: C, ar: &[$t], ai: &[$t], lda: usize,
                xr: &[$t], xi: &[$t], beta: C, yr: &mut [$t], yi: &mut [$t],
            ) {
                check_mat("gemv_t", "a(re)", m, n, lda, ar.len());
                check_mat("gemv_t", "a(im)", m, n, lda, ai.len());
                len_check("gemv_t x", xr.len(), m);
                len_check("gemv_t x", xi.len(), m);
                len_check("gemv_t y", yr.len(), n);
                len_check("gemv_t y", yi.len(), n);
                if alpha == ZERO || m == 0 {
                    cscale(beta, yr, yi);
                    return;
                }
                for j in 0..n {
                    let d = dot(conj, &ar[j * lda..][..m], &ai[j * lda..][..m], xr, xi);
                    let mut v = cmul(alpha, d);
                    if beta != ZERO {
                        let b = cmul(beta, [yr[j], yi[j]]);
                        v = [v[0] + b[0], v[1] + b[1]];
                    }
                    yr[j] = v[0];
                    yi[j] = v[1];
                }
            }

            // ------------------------------------------------------------
            // Split-plane gemm via 4 real gemm calls
            // ------------------------------------------------------------

            pub(crate) fn gemm_workspace_len(m: usize, n: usize, k: usize) -> usize {
                if m == 0 || n == 0 || k == 0 {
                    return 0;
                }
                crate::$real::gemm_workspace_len(m, n, k) + 2 * k * n
            }

            #[allow(clippy::too_many_arguments)]
            fn gemm_need(m: usize, n: usize, k: usize, alpha: C) -> usize {
                if m == 0 || n == 0 || k == 0 {
                    return 0;
                }
                crate::$real::gemm_workspace_len(m, n, k) + if alpha == ONE { 0 } else { 2 * k * n }
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemm_with_workspace(
                m: usize, n: usize, k: usize, alpha: C, ar: &[$t], ai: &[$t], lda: usize,
                br: &[$t], bi: &[$t], ldb: usize, beta: C, cr: &mut [$t], ci: &mut [$t],
                ldc: usize, ws: &mut [$t],
            ) {
                check_mat("gemm", "a(re)", m, k, lda, ar.len());
                check_mat("gemm", "a(im)", m, k, lda, ai.len());
                check_mat("gemm", "b(re)", k, n, ldb, br.len());
                check_mat("gemm", "b(im)", k, n, ldb, bi.len());
                check_mat("gemm", "c(re)", m, n, ldc, cr.len());
                check_mat("gemm", "c(im)", m, n, ldc, ci.len());
                if m == 0 || n == 0 {
                    return;
                }
                if beta != ONE {
                    for j in 0..n {
                        cscale(beta, &mut cr[j * ldc..][..m], &mut ci[j * ldc..][..m]);
                    }
                }
                if alpha == ZERO || k == 0 {
                    return;
                }
                let need = gemm_need(m, n, k, alpha);
                check_ws("gemm", ws.len(), need);
                let real_len = crate::$real::gemm_workspace_len(m, n, k);
                let (rws, rest) = ws.split_at_mut(real_len);
                if alpha == ONE {
                    four(m, n, k, ar, ai, lda, br, bi, ldb, cr, ci, ldc, rws);
                } else {
                    let (bsr, bsi) = rest[..2 * k * n].split_at_mut(k * n);
                    for j in 0..n {
                        for p in 0..k {
                            let v = cmul(alpha, [br[p + j * ldb], bi[p + j * ldb]]);
                            bsr[p + j * k] = v[0];
                            bsi[p + j * k] = v[1];
                        }
                    }
                    four(m, n, k, ar, ai, lda, bsr, bsi, k, cr, ci, ldc, rws);
                }
            }

            /// `C += A B` as four real gemms (`beta = 1`, so `C` is only accumulated).
            #[allow(clippy::too_many_arguments)]
            fn four(
                m: usize, n: usize, k: usize, ar: &[$t], ai: &[$t], lda: usize,
                br: &[$t], bi: &[$t], ldb: usize, cr: &mut [$t], ci: &mut [$t],
                ldc: usize, ws: &mut [$t],
            ) {
                crate::$real::gemm_with_workspace(m, n, k, 1.0, ar, lda, br, ldb, 1.0, cr, ldc, ws);
                crate::$real::gemm_with_workspace(m, n, k, -1.0, ai, lda, bi, ldb, 1.0, cr, ldc, ws);
                crate::$real::gemm_with_workspace(m, n, k, 1.0, ar, lda, bi, ldb, 1.0, ci, ldc, ws);
                crate::$real::gemm_with_workspace(m, n, k, 1.0, ai, lda, br, ldb, 1.0, ci, ldc, ws);
            }

            #[cfg(feature = "alloc")]
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemm(
                m: usize, n: usize, k: usize, alpha: C, ar: &[$t], ai: &[$t], lda: usize,
                br: &[$t], bi: &[$t], ldb: usize, beta: C, cr: &mut [$t], ci: &mut [$t],
                ldc: usize,
            ) {
                let mut ws = alloc::vec![0.0 as $t; gemm_need(m, n, k, alpha)];
                gemm_with_workspace(m, n, k, alpha, ar, ai, lda, br, bi, ldb, beta, cr, ci, ldc, &mut ws);
            }

            // ------------------------------------------------------------
            // Layout adapters and interleaved entry points
            // ------------------------------------------------------------

            pub(crate) fn deinterleave(src: &[C], re: &mut [$t], im: &mut [$t]) {
                len_check("deinterleave re", re.len(), src.len());
                len_check("deinterleave im", im.len(), src.len());
                for (i, z) in src.iter().enumerate() {
                    re[i] = z[0];
                    im[i] = z[1];
                }
            }

            pub(crate) fn interleave(re: &[$t], im: &[$t], dst: &mut [C]) {
                len_check("interleave re", re.len(), dst.len());
                len_check("interleave im", im.len(), dst.len());
                for (i, z) in dst.iter_mut().enumerate() {
                    *z = [re[i], im[i]];
                }
            }

            pub(crate) fn axpy_il(alpha: C, x: &[C], y: &mut [C]) {
                len_check("axpy_il", x.len(), y.len());
                if alpha == ZERO {
                    return;
                }
                for (yi, xi) in y.iter_mut().zip(x) {
                    let t = cmul(alpha, *xi);
                    yi[0] += t[0];
                    yi[1] += t[1];
                }
            }

            pub(crate) fn scal_il(alpha: C, x: &mut [C]) {
                if alpha == ONE {
                    return;
                }
                if alpha == ZERO {
                    x.fill(ZERO);
                    return;
                }
                for xi in x.iter_mut() {
                    *xi = cmul(alpha, *xi);
                }
            }

            pub(crate) fn dot_il(conj: bool, x: &[C], y: &[C]) -> C {
                len_check(if conj { "dotc_il" } else { "dotu_il" }, x.len(), y.len());
                let (mut rr, mut ii, mut ri, mut ir) = ([0.0 as $t; W], [0.0 as $t; W], [0.0 as $t; W], [0.0 as $t; W]);
                let xc = x.chunks_exact(W);
                let yc = y.chunks_exact(W);
                let (xt, yt) = (xc.remainder(), yc.remainder());
                for (a, b) in xc.zip(yc) {
                    for j in 0..W {
                        rr[j] += a[j][0] * b[j][0];
                        ii[j] += a[j][1] * b[j][1];
                        ri[j] += a[j][0] * b[j][1];
                        ir[j] += a[j][1] * b[j][0];
                    }
                }
                let (mut trr, mut tii, mut tri, mut tir) = (0.0 as $t, 0.0 as $t, 0.0 as $t, 0.0 as $t);
                for (a, b) in xt.iter().zip(yt) {
                    trr += a[0] * b[0];
                    tii += a[1] * b[1];
                    tri += a[0] * b[1];
                    tir += a[1] * b[0];
                }
                let (rr, ii, ri, ir) = (reduce(rr) + trr, reduce(ii) + tii, reduce(ri) + tri, reduce(ir) + tir);
                if conj { [rr + ii, ri - ir] } else { [rr - ii, ri + ir] }
            }

            pub(crate) fn nrm2_il(x: &[C]) -> $t {
                let mut acc = [0.0 as $t; W];
                let xc = x.chunks_exact(W);
                let xt = xc.remainder();
                for a in xc {
                    for j in 0..W {
                        acc[j] += a[j][0] * a[j][0] + a[j][1] * a[j][1];
                    }
                }
                let mut tail = 0.0 as $t;
                for a in xt {
                    tail += a[0] * a[0] + a[1] * a[1];
                }
                $sqrt(reduce(acc) + tail)
            }

            pub(crate) fn asum_il(x: &[C]) -> $t {
                let mut acc = [0.0 as $t; W];
                let xc = x.chunks_exact(W);
                let xt = xc.remainder();
                for a in xc {
                    for j in 0..W {
                        acc[j] += a[j][0].abs() + a[j][1].abs();
                    }
                }
                let mut tail = 0.0 as $t;
                for a in xt {
                    tail += a[0].abs() + a[1].abs();
                }
                reduce(acc) + tail
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemv_il(
                m: usize, n: usize, alpha: C, a: &[C], lda: usize, x: &[C], beta: C, y: &mut [C],
            ) {
                check_mat("gemv_il", "a", m, n, lda, a.len());
                len_check("gemv_il x", x.len(), n);
                len_check("gemv_il y", y.len(), m);
                scal_il(beta, y);
                if alpha == ZERO || m == 0 || n == 0 {
                    return;
                }
                for j in 0..n {
                    let t = cmul(alpha, x[j]);
                    let col = &a[j * lda..][..m];
                    for (yi, ai) in y.iter_mut().zip(col) {
                        yi[0] += ai[0] * t[0] - ai[1] * t[1];
                        yi[1] += ai[0] * t[1] + ai[1] * t[0];
                    }
                }
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemv_t_il(
                conj: bool, m: usize, n: usize, alpha: C, a: &[C], lda: usize, x: &[C],
                beta: C, y: &mut [C],
            ) {
                check_mat("gemv_t_il", "a", m, n, lda, a.len());
                len_check("gemv_t_il x", x.len(), m);
                len_check("gemv_t_il y", y.len(), n);
                if alpha == ZERO || m == 0 {
                    scal_il(beta, y);
                    return;
                }
                for j in 0..n {
                    let d = dot_il(conj, &a[j * lda..][..m], x);
                    let mut v = cmul(alpha, d);
                    if beta != ZERO {
                        let b = cmul(beta, y[j]);
                        v = [v[0] + b[0], v[1] + b[1]];
                    }
                    y[j] = v;
                }
            }

            /// Interleaved gemm: de-interleaves `A`, `B` (and `C` unless
            /// `beta == 0`) into split planes, runs the split gemm and
            /// re-interleaves `C`.
            #[cfg(feature = "alloc")]
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn gemm_il(
                m: usize, n: usize, k: usize, alpha: C, a: &[C], lda: usize, b: &[C],
                ldb: usize, beta: C, c: &mut [C], ldc: usize,
            ) {
                check_mat("gemm_il", "a", m, k, lda, a.len());
                check_mat("gemm_il", "b", k, n, ldb, b.len());
                check_mat("gemm_il", "c", m, n, ldc, c.len());
                if m == 0 || n == 0 {
                    return;
                }
                if (alpha == ZERO || k == 0) && beta == ONE {
                    return;
                }
                let split = |src: &[C], rows: usize, cols: usize, ld: usize| {
                    let mut re = alloc::vec![0.0 as $t; rows * cols];
                    let mut im = alloc::vec![0.0 as $t; rows * cols];
                    for j in 0..cols {
                        deinterleave(&src[j * ld..][..rows], &mut re[j * rows..][..rows], &mut im[j * rows..][..rows]);
                    }
                    (re, im)
                };
                let (ar, ai) = if alpha == ZERO || k == 0 {
                    (alloc::vec::Vec::new(), alloc::vec::Vec::new())
                } else {
                    split(a, m, k, lda)
                };
                let (br, bi) = if alpha == ZERO || k == 0 {
                    (alloc::vec::Vec::new(), alloc::vec::Vec::new())
                } else {
                    split(b, k, n, ldb)
                };
                let (mut cr, mut ci) = if beta == ZERO {
                    (alloc::vec![0.0 as $t; m * n], alloc::vec![0.0 as $t; m * n])
                } else {
                    split(c, m, n, ldc)
                };
                if alpha == ZERO || k == 0 {
                    // gemm with an empty product only scales C.
                    gemm(m, n, 0, alpha, &[], &[], m, &[], &[], 1, beta, &mut cr, &mut ci, m);
                } else {
                    gemm(m, n, k, alpha, &ar, &ai, m, &br, &bi, k, beta, &mut cr, &mut ci, m);
                }
                for j in 0..n {
                    interleave(&cr[j * m..][..m], &ci[j * m..][..m], &mut c[j * ldc..][..m]);
                }
            }

            /// Naive scalar reference implementations (split planes, left-to-right sums).
            pub(crate) mod reference {
                use super::{C, cmul};
                use crate::check::check_mat;

                pub(crate) fn axpy(alpha: C, xr: &[$t], xi: &[$t], yr: &mut [$t], yi: &mut [$t]) {
                    for i in 0..xr.len() {
                        let t = cmul(alpha, [xr[i], xi[i]]);
                        yr[i] += t[0];
                        yi[i] += t[1];
                    }
                }
                pub(crate) fn scal(alpha: C, xr: &mut [$t], xi: &mut [$t]) {
                    for i in 0..xr.len() {
                        let t = cmul(alpha, [xr[i], xi[i]]);
                        xr[i] = t[0];
                        xi[i] = t[1];
                    }
                }
                pub(crate) fn dot(conj: bool, xr: &[$t], xi: &[$t], yr: &[$t], yi: &[$t]) -> C {
                    let mut s: C = [0.0, 0.0];
                    for i in 0..xr.len() {
                        let x = [xr[i], if conj { -xi[i] } else { xi[i] }];
                        let t = cmul(x, [yr[i], yi[i]]);
                        s = [s[0] + t[0], s[1] + t[1]];
                    }
                    s
                }
                pub(crate) fn nrm2(xr: &[$t], xi: &[$t]) -> $t {
                    let mut s = 0.0 as $t;
                    for i in 0..xr.len() {
                        s += xr[i] * xr[i] + xi[i] * xi[i];
                    }
                    $sqrt(s)
                }
                pub(crate) fn asum(xr: &[$t], xi: &[$t]) -> $t {
                    let mut s = 0.0 as $t;
                    for i in 0..xr.len() {
                        s += xr[i].abs() + xi[i].abs();
                    }
                    s
                }
                /// `op` = 0 none, 1 transpose, 2 conjugate transpose.
                #[allow(clippy::too_many_arguments)]
                pub(crate) fn gemv(
                    op: u8, m: usize, n: usize, alpha: C, ar: &[$t], ai: &[$t], lda: usize,
                    xr: &[$t], xi: &[$t], beta: C, yr: &mut [$t], yi: &mut [$t],
                ) {
                    check_mat("reference gemv", "a", m, n, lda, ar.len());
                    let (rows, cols) = if op == 0 { (m, n) } else { (n, m) };
                    for r in 0..rows {
                        let mut s: C = [0.0, 0.0];
                        for c in 0..cols {
                            let idx = if op == 0 { r + c * lda } else { c + r * lda };
                            let a = [ar[idx], if op == 2 { -ai[idx] } else { ai[idx] }];
                            let t = cmul(a, [xr[c], xi[c]]);
                            s = [s[0] + t[0], s[1] + t[1]];
                        }
                        let mut v = cmul(alpha, s);
                        if beta != [0.0, 0.0] {
                            let b = cmul(beta, [yr[r], yi[r]]);
                            v = [v[0] + b[0], v[1] + b[1]];
                        }
                        yr[r] = v[0];
                        yi[r] = v[1];
                    }
                }
                #[allow(clippy::too_many_arguments)]
                pub(crate) fn gemm(
                    m: usize, n: usize, k: usize, alpha: C, ar: &[$t], ai: &[$t], lda: usize,
                    br: &[$t], bi: &[$t], ldb: usize, beta: C, cr: &mut [$t], ci: &mut [$t],
                    ldc: usize,
                ) {
                    check_mat("reference gemm", "a", m, k, lda, ar.len());
                    check_mat("reference gemm", "b", k, n, ldb, br.len());
                    check_mat("reference gemm", "c", m, n, ldc, cr.len());
                    for j in 0..n {
                        for i in 0..m {
                            let mut s: C = [0.0, 0.0];
                            for p in 0..k {
                                let t = cmul([ar[i + p * lda], ai[i + p * lda]], [br[p + j * ldb], bi[p + j * ldb]]);
                                s = [s[0] + t[0], s[1] + t[1]];
                            }
                            let mut v = cmul(alpha, s);
                            if beta != [0.0, 0.0] {
                                let b = cmul(beta, [cr[i + j * ldc], ci[i + j * ldc]]);
                                v = [v[0] + b[0], v[1] + b[1]];
                            }
                            cr[i + j * ldc] = v[0];
                            ci[i + j * ldc] = v[1];
                        }
                    }
                }
            }
        }
    };
}
