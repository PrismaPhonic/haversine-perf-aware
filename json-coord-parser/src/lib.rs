#![feature(portable_simd, duration_millis_float)]

use std::mem::{MaybeUninit, transmute};
#[cfg(target_feature = "avx512f")]
use std::path::Path;
use std::simd::simd_swizzle;
use std::simd::{Simd, num::SimdUint};

use serde::{Deserialize, Serialize};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub const EARTH_RADIUS: f64 = 6372.8;

type U8x16 = Simd<u8, 16>;
type U32x8 = Simd<u32, 8>;
type U32x16 = Simd<u32, 16>;
type U64x8 = Simd<u64, 8>;
type F64x8 = Simd<f64, 8>;

const POW10_16: U32x16 = Simd::from_array([
    10_000_000, 1_000_000, 100_000, 10_000, 1_000, 100, 10, 1, 10_000_000, 1_000_000, 100_000,
    10_000, 1_000, 100, 10, 1,
]);

#[inline(always)]
fn int_from_16_ascii_bytes(digits_ascii: &[u8; 16]) -> u64 {
    // Load and ASCII->digit in SIMD (u8 - b'0')
    let d_u8: U8x16 = U8x16::from_slice(digits_ascii);
    let d_u8 = d_u8 - U8x16::splat(b'0');

    // Widen once to u32 and dot with constant weights
    let d_u32: U32x16 = d_u8.cast::<u32>();
    let prod = d_u32 * POW10_16;

    // Split into hi/lo halves, reduce each, then combine with *1e8
    let hi: U32x8 = simd_swizzle!(prod, [0, 1, 2, 3, 4, 5, 6, 7]);
    let lo: U32x8 = simd_swizzle!(prod, [8, 9, 10, 11, 12, 13, 14, 15]);

    let hi_sum = hi.reduce_sum() as u64;
    let lo_sum = lo.reduce_sum() as u64;

    hi_sum * 100_000_000u64 + lo_sum
}

#[inline(always)]
fn parse_one_fast_u8(bytes: &[u8; 17]) -> ([u8; 16], u32, bool) {
    let neg = bytes[0] == b'-';
    let mut di = neg as usize; // If negative will be 1.

    // Uninitialized [u8;16] for digits only (no '.')
    let mut buf: MaybeUninit<[u8; 16]> = MaybeUninit::uninit();
    let base = buf.as_mut_ptr() as *mut u8;

    let mut int_digits = 0u32;
    let mut i = 0usize;

    loop {
        let ch = bytes[di];
        if ch == b'.' {
            int_digits = i as u32;
            i -= 1;
        } else {
            unsafe {
                base.add(i).write(ch);
            }
        }

        if di == 16 {
            break;
        }
        di += 1;
        i += 1;
    }

    if neg {
        // Pad tail for negatives (so both signs share frac placement math)
        unsafe {
            base.add(15).write(b'0');
        }
    }

    let digits = unsafe { buf.assume_init() };
    // frac_digits = total(16) - int_digits (shared for both signs thanks to tail pad)
    let frac_digits = 16 - int_digits;
    (digits, frac_digits, neg)
}

#[cfg(target_feature = "avx512f")]
#[inline(always)]
unsafe fn apply_scale_sign_avx512(int_f: F64x8, k_u64: U64x8, neg: [bool; 8]) -> F64x8 {
    // Reinterpret to AVX-512 intrinsic vectors
    let v_pd: __m512d = unsafe { transmute::<F64x8, __m512d>(int_f) };
    let k_vi: __m512i = unsafe { transmute::<U64x8, __m512i>(k_u64) };

    // Build masks: k == 13 and k == 14 (else => 15)
    let m13: __mmask8 = unsafe { _mm512_cmpeq_epu64_mask(k_vi, _mm512_set1_epi64(13)) };
    let m14: __mmask8 = unsafe { _mm512_cmpeq_epu64_mask(k_vi, _mm512_set1_epi64(14)) };

    // Three candidates
    let c13 = unsafe { _mm512_mul_pd(v_pd, _mm512_set1_pd(1e-13)) };
    let c14 = unsafe { _mm512_mul_pd(v_pd, _mm512_set1_pd(1e-14)) };
    let c15 = unsafe { _mm512_mul_pd(v_pd, _mm512_set1_pd(1e-15)) };

    // 14 vs 15
    let t = unsafe { _mm512_mask_blend_pd(m14, c15, c14) };
    // 13 vs (14/15)
    let mut out = unsafe { _mm512_mask_blend_pd(m13, t, c13) };

    // Sign: flip sign bit where neg[i] is true (branchless)
    let mut neg_mask: u8 = 0;
    // Pack 8 bools into __mmask8
    for (i, neg) in neg.into_iter().enumerate() {
        neg_mask |= (neg as u8) << i;
    }
    let km_neg: __mmask8 = neg_mask as __mmask8;

    let signbit = unsafe { _mm512_set1_pd(f64::from_bits(0x8000_0000_0000_0000)) };
    out = unsafe { _mm512_mask_xor_pd(out, km_neg, out, signbit) };

    unsafe { transmute::<__m512d, F64x8>(out) }
}

#[cfg(target_feature = "avx512f")]
#[inline(always)]
pub unsafe fn parse8_simd(s: [&[u8; 17]; 8]) -> [f64; 8] {
    // 1) Parse each string into 16 ASCII digits (no subtraction yet), frac count, sign
    let (d0, k0, n0) = parse_one_fast_u8(s[0]);
    let (d1, k1, n1) = parse_one_fast_u8(s[1]);
    let (d2, k2, n2) = parse_one_fast_u8(s[2]);
    let (d3, k3, n3) = parse_one_fast_u8(s[3]);
    let (d4, k4, n4) = parse_one_fast_u8(s[4]);
    let (d5, k5, n5) = parse_one_fast_u8(s[5]);
    let (d6, k6, n6) = parse_one_fast_u8(s[6]);
    let (d7, k7, n7) = parse_one_fast_u8(s[7]);

    // 2) SIMD dot16 per number → scalar u64, via SIMD ASCII->digit & weights
    let ints = [
        int_from_16_ascii_bytes(&d0),
        int_from_16_ascii_bytes(&d1),
        int_from_16_ascii_bytes(&d2),
        int_from_16_ascii_bytes(&d3),
        int_from_16_ascii_bytes(&d4),
        int_from_16_ascii_bytes(&d5),
        int_from_16_ascii_bytes(&d6),
        int_from_16_ascii_bytes(&d7),
    ];

    let int_u64 = U64x8::from_array(ints);

    // 3) f64 conversion
    let int_f = int_u64.cast::<f64>();

    // 4) SIMD scale & sign (A): k∈{13,14,15} → masked blends; sign via mask xor
    let ks = U64x8::from_array([
        k0 as u64, k1 as u64, k2 as u64, k3 as u64, k4 as u64, k5 as u64, k6 as u64, k7 as u64,
    ]);
    let nega = [n0, n1, n2, n3, n4, n5, n6, n7];

    // SAFETY: we require AVX-512F here; caller should run on appropriate CPU.
    let val: F64x8 = unsafe { apply_scale_sign_avx512(int_f, ks, nega) };

    val.to_array()
}

const HEADER_LEN: usize = 8; // {pairs:[
const TRAILER_LEN: usize = 2; // ]}
const LABEL_X0_LEN: usize = 6; // {"x0":
const TOKEN_LEN: usize = 17;
const GAP_INTRA: usize = 23;
const GAP_INTER: usize = 25;
const OBJ_PLUS_COMMA: usize = 94;
const WINDOW: usize = OBJ_PLUS_COMMA * 2;
const OFFS: [usize; 8] = [
    0,                         // obj0 x0
    GAP_INTRA,                 // obj0 y0
    2 * GAP_INTRA,             // obj0 x1
    3 * GAP_INTRA,             // obj0 y1
    3 * GAP_INTRA + GAP_INTER, // obj1 x0
    4 * GAP_INTRA + GAP_INTER, // obj1 y0
    5 * GAP_INTRA + GAP_INTER, // obj1 x1
    6 * GAP_INTRA + GAP_INTER, // obj1 y1
];

#[derive(Deserialize)]
pub struct CoordinatePairs {
    pub pairs: Vec<CoordinatePair>,
}

#[derive(Deserialize)]
pub struct CoordinatePair {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

#[inline(always)]
fn object_count_bytes(input: &[u8]) -> usize {
    let total = input.len();
    debug_assert!(total >= HEADER_LEN + TRAILER_LEN);
    let body_len = total - HEADER_LEN - TRAILER_LEN;
    (body_len + 1) / OBJ_PLUS_COMMA
}

#[inline(always)]
fn byte_slice_to_array(bytes: &[u8]) -> &[u8; 17] {
    unsafe { &*(bytes.as_ptr() as *const [u8; 17]) }
}

#[cfg(target_feature = "avx512f")]
pub fn parse_pairs_json_fixed_bytes(input: &[u8]) -> CoordinatePairs {
    debug_assert!(input.len() >= HEADER_LEN + TRAILER_LEN);
    debug_assert!(&input[0..HEADER_LEN] == b"{pairs:[");
    debug_assert!(&input[input.len() - TRAILER_LEN..] == b"]}");

    let m = object_count_bytes(input);
    debug_assert!(m > 0);
    debug_assert!(m % 2 == 0); // we batch in twos

    let mut pairs = Vec::with_capacity(m);

    // Pointer to first number (right after {"x0":)
    let mut p = HEADER_LEN + LABEL_X0_LEN;

    // We consume 2 objects (8 numbers) per batch
    let batches = m / 2;
    for _ in 0..batches {
        let base = p;

        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        unsafe {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let next = base + WINDOW;
            // one or two prefetches are usually enough
            if next + OFFS[0] + TOKEN_LEN <= input.len() {
                _mm_prefetch(input.as_ptr().add(next + OFFS[0]) as *const i8, _MM_HINT_T0);
                _mm_prefetch(input.as_ptr().add(next + OFFS[4]) as *const i8, _MM_HINT_T0);
            }
        }

        // compute all 8 indices from 'base' independently (no mutation chain)
        let idx0 = base + OFFS[0];
        let idx1 = base + OFFS[1];
        let idx2 = base + OFFS[2];
        let idx3 = base + OFFS[3];
        let idx4 = base + OFFS[4];
        let idx5 = base + OFFS[5];
        let idx6 = base + OFFS[6];
        let idx7 = base + OFFS[7];

        // grab 8 tokens
        let toks: [&[u8; 17]; 8] = unsafe {
            [
                byte_slice_to_array(&input.get_unchecked(idx0..idx0 + TOKEN_LEN)),
                byte_slice_to_array(&input.get_unchecked(idx1..idx1 + TOKEN_LEN)),
                byte_slice_to_array(&input.get_unchecked(idx2..idx2 + TOKEN_LEN)),
                byte_slice_to_array(&input.get_unchecked(idx3..idx3 + TOKEN_LEN)),
                byte_slice_to_array(&input.get_unchecked(idx4..idx4 + TOKEN_LEN)),
                byte_slice_to_array(&input.get_unchecked(idx5..idx5 + TOKEN_LEN)),
                byte_slice_to_array(&input.get_unchecked(idx6..idx6 + TOKEN_LEN)),
                byte_slice_to_array(&input.get_unchecked(idx7..idx7 + TOKEN_LEN)),
            ]
        };

        let vals = unsafe { parse8_simd(toks) };

        pairs.push(CoordinatePair {
            x0: vals[0],
            y0: vals[1],
            x1: vals[2],
            y1: vals[3],
        });
        pairs.push(CoordinatePair {
            x0: vals[4],
            y0: vals[5],
            x1: vals[6],
            y1: vals[7],
        });

        // step to the next 2-object window; no dependency on q's evolution
        p += WINDOW;
    }

    CoordinatePairs { pairs }
}

#[derive(Serialize)]
pub struct HaversineAnswers {
    pairs: Vec<f64>,
}

#[inline(always)]
pub fn reference_haversine(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let mut lat1 = y0;
    let mut lat2 = y1;
    let lon1 = x0;
    let lon2 = x1;

    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    lat1 = lat1.to_radians();
    lat2 = lat2.to_radians();

    let a = ((d_lat / 2.0).sin()).powf(2.0)
        + (lat1.cos() * lat2.cos() * ((d_lon / 2.0).sin()).powf(2.0));
    let c = 2.0 * (a.sqrt()).asin();

    EARTH_RADIUS * c
}

#[cfg(target_feature = "avx512f")]
pub fn haversine_from_json_coords_parallel(input: &[u8]) -> HaversineAnswers {
    use std::sync::mpsc;
    use std::thread;

    debug_assert!(input.len() >= HEADER_LEN + TRAILER_LEN);
    debug_assert!(&input[0..HEADER_LEN] == b"{pairs:[");
    debug_assert!(&input[input.len() - TRAILER_LEN..] == b"]}");

    let m = object_count_bytes(input);
    debug_assert!(m > 0 && m % 2 == 0);
    let num_batches = m / 2; // 2 objects / batch

    // Discover physical cores (one CoreId per physical core on supported OSes)
    let cores: Vec<core_affinity::CoreId> = core_affinity::get_core_ids().unwrap_or_default();

    // Don’t spawn more workers than batches
    let n_workers = cores.len().min(num_batches);

    let base_start = HEADER_LEN + LABEL_X0_LEN;

    // Channel to collect chunks in (start_obj, Vec<CoordinatePair>) order
    let (tx, rx) = mpsc::channel::<(usize, Vec<f64>)>();

    thread::scope(|scope| {
        // Divide num_batches as evenly as possible across n_workers
        let mut start_batch = 0usize;
        for (wid, core) in cores.iter().copied().take(n_workers).enumerate() {
            // round-robin-ish even split: last workers may get ±1
            let rem = num_batches - start_batch;
            let left = n_workers - wid;
            let take = rem / left + (if rem % left != 0 { 1 } else { 0 });
            let end_batch = start_batch + take;

            let tx = tx.clone();
            let input_ref = &input[..];

            scope.spawn(move || {
                // Pin to the physical core for this worker
                let _ = core_affinity::set_for_current(core);

                let mut out = Vec::with_capacity(take * 2); // 2 objects per batch
                for b in start_batch..end_batch {
                    let p0 = base_start + (2 * b) * OBJ_PLUS_COMMA;

                    // Gather 8 tokens (two objects) with fixed offsets
                    let toks: [&[u8; 17]; 8] = [
                        byte_slice_to_array(&input_ref[p0 + OFFS[0]..p0 + OFFS[0] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[1]..p0 + OFFS[1] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[2]..p0 + OFFS[2] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[3]..p0 + OFFS[3] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[4]..p0 + OFFS[4] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[5]..p0 + OFFS[5] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[6]..p0 + OFFS[6] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[7]..p0 + OFFS[7] + TOKEN_LEN]),
                    ];

                    let vals = unsafe { parse8_simd(toks) };

                    let hav1 = reference_haversine(vals[0], vals[1], vals[2], vals[3]);
                    let hav2 = reference_haversine(vals[4], vals[5], vals[6], vals[7]);
                    out.push(hav1);
                    out.push(hav2);
                }

                let start_obj = 2 * start_batch; // first object index produced by this worker
                let _ = tx.send((start_obj, out));
            });

            start_batch = end_batch;
        }
        drop(tx);
    });

    // Collect, sort, stitch — no copying per element
    let mut chunks = rx.into_iter().collect::<Vec<_>>();
    chunks.sort_unstable_by_key(|(start, _)| *start);

    // Pre-size and copy in-place (tightest collector)
    let mut pairs = Vec::with_capacity(m);
    for (_, v) in chunks {
        // Moves elements from each worker’s Vec into `pairs` (no reallocation because of capacity)
        pairs.extend(v);
    }

    HaversineAnswers { pairs }
}

#[cfg(target_feature = "avx512f")]
pub fn parse_pairs_json_parallel(input: &[u8]) -> CoordinatePairs {
    use std::sync::mpsc;
    use std::thread;

    debug_assert!(input.len() >= HEADER_LEN + TRAILER_LEN);
    debug_assert!(&input[0..HEADER_LEN] == b"{pairs:[");
    debug_assert!(&input[input.len() - TRAILER_LEN..] == b"]}");

    let m = object_count_bytes(input);
    debug_assert!(m > 0 && m % 2 == 0);
    let num_batches = m / 2; // 2 objects / batch

    // Discover physical cores (one CoreId per physical core on supported OSes)
    let cores: Vec<core_affinity::CoreId> = core_affinity::get_core_ids().unwrap_or_default();

    // Don’t spawn more workers than batches
    let n_workers = cores.len().min(num_batches);

    let base_start = HEADER_LEN + LABEL_X0_LEN;

    // Channel to collect chunks in (start_obj, Vec<CoordinatePair>) order
    let (tx, rx) = mpsc::channel::<(usize, Vec<CoordinatePair>)>();

    thread::scope(|scope| {
        // Divide num_batches as evenly as possible across n_workers
        let mut start_batch = 0usize;
        for (wid, core) in cores.iter().copied().take(n_workers).enumerate() {
            // round-robin-ish even split: last workers may get ±1
            let rem = num_batches - start_batch;
            let left = n_workers - wid;
            let take = rem / left + (if rem % left != 0 { 1 } else { 0 });
            let end_batch = start_batch + take;

            let tx = tx.clone();
            let input_ref = &input[..];

            scope.spawn(move || {
                // Pin to the physical core for this worker
                let _ = core_affinity::set_for_current(core);

                let mut out = Vec::with_capacity(take * 2); // 2 objects per batch
                for b in start_batch..end_batch {
                    let p0 = base_start + (2 * b) * OBJ_PLUS_COMMA;

                    // Gather 8 tokens (two objects) with fixed offsets
                    let toks: [&[u8; 17]; 8] = [
                        byte_slice_to_array(&input_ref[p0 + OFFS[0]..p0 + OFFS[0] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[1]..p0 + OFFS[1] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[2]..p0 + OFFS[2] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[3]..p0 + OFFS[3] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[4]..p0 + OFFS[4] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[5]..p0 + OFFS[5] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[6]..p0 + OFFS[6] + TOKEN_LEN]),
                        byte_slice_to_array(&input_ref[p0 + OFFS[7]..p0 + OFFS[7] + TOKEN_LEN]),
                    ];

                    let vals = unsafe { parse8_simd(toks) };

                    out.push(CoordinatePair {
                        x0: vals[0],
                        y0: vals[1],
                        x1: vals[2],
                        y1: vals[3],
                    });
                    out.push(CoordinatePair {
                        x0: vals[4],
                        y0: vals[5],
                        x1: vals[6],
                        y1: vals[7],
                    });
                }

                let start_obj = 2 * start_batch; // first object index produced by this worker
                let _ = tx.send((start_obj, out));
            });

            start_batch = end_batch;
        }
        drop(tx);
    });

    // Collect, sort, stitch — no copying per element
    let mut chunks = rx.into_iter().collect::<Vec<_>>();
    chunks.sort_unstable_by_key(|(start, _)| *start);

    // Pre-size and copy in-place (tightest collector)
    let mut pairs = Vec::with_capacity(m);
    for (_, v) in chunks {
        // Moves elements from each worker’s Vec into `pairs` (no reallocation because of capacity)
        pairs.extend(v);
    }

    CoordinatePairs { pairs }
}

#[cfg(target_feature = "avx512f")]
pub fn parse_pairs_json_fixed_from_file(
    path: impl AsRef<Path>,
) -> std::io::Result<CoordinatePairs> {
    let bytes = std::fs::read(path)?;
    Ok(parse_pairs_json_fixed_bytes(&bytes))
}

#[cfg(target_feature = "avx512f")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_at_once_degrees_format() {
        let out = unsafe {
            parse8_simd([
                b"180.0000000000000", // +180.0 (3 int, 13 frac)
                b"-180.000000000000", // -180.0 (3 int, 12 frac)
                b"90.00000000000000", // +90.0  (2 int, 14 frac)
                b"-9.00000000000000", // -9.0   (1 int, 14 frac)
                b"1.800000000000000", // +1.8   (1 int, 15 frac)
                b"-0.10000000000000", // -0.1  (1 int, 14 frac)
                b"0.000000000000000", //  0.0   (1 int, 15 frac)
                b"123.4000000000000", // +123.4 (3 int, 13 frac)
            ])
        };

        let expected = [180.0, -180.0, 90.0, -9.0, 1.8, -0.1, 0.0, 123.4];
        for i in 0..8 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-9,
                "lane {} mismatch: got {}, expected {}",
                i,
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn example_two_pairs_bytes() {
        let s = b"{pairs:[{\"x0\":180.3949876567845,\"y0\":80.58940234958675,\"x1\":-2.09845738495847,\"y1\":-40.1283940392839},{\"x0\":45.38495069584738,\"y0\":5.094859382738475,\"x1\":20.09584738475867,\"y1\":70.47381273948574}]}";

        let cp = parse_pairs_json_fixed_bytes(s);
        assert_eq!(cp.pairs.len(), 2);

        let p0 = &cp.pairs[0];
        let p1 = &cp.pairs[1];

        assert!((p0.x0 - 180.3949876567845).abs() < 1e-12);
        assert!((p0.y0 - 80.58940234958675).abs() < 1e-12);
        assert!((p0.x1 + 2.09845738495847).abs() < 1e-12);
        assert!((p0.y1 + 40.1283940392839).abs() < 1e-12);

        assert!((p1.x0 - 45.38495069584738).abs() < 1e-12);
        assert!((p1.y0 - 5.094859382738475).abs() < 1e-12);
        assert!((p1.x1 - 20.09584738475867).abs() < 1e-12);
        assert!((p1.y1 - 70.47381273948574).abs() < 1e-12);
    }
}
