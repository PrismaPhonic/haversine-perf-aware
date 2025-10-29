use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use std::path::PathBuf;

use json_coord_parser::CoordinatePairs;

#[cfg(target_feature = "avx512f")]
use json_coord_parser::{parse_pairs_json_fixed_bytes, parse_pairs_json_parallel, parse8_simd};

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn sample_inputs() -> [&'static str; 8] {
    // 8 canonical, properly-formed degree strings from your tests
    [
        "180.0000000000000", // +180.0 (3 int, 13 frac)
        "-180.000000000000", // -180.0 (3 int, 12 frac, tail pad implied)
        "90.00000000000000", // +90.0  (2 int, 14 frac)
        "-9.00000000000000", // -9.0   (1 int, 14 frac)
        "1.800000000000000", // +1.8   (1 int, 15 frac)
        "-0.10000000000000", // -0.1    (1 int, 14 frac)
        "0.000000000000000", //  0.0   (1 int, 15 frac)
        "123.4000000000000", // +123.4 (3 int, 13 frac)
    ]
}

/// Build a large corpus by repeating the 8 canonical inputs.
/// Returns Vec<&'static str> so we can slice it into 8-wide chunks without allocations in the hot loop.
fn build_corpus(multiplier: usize) -> Vec<&'static str> {
    let base = sample_inputs();
    let mut v = Vec::with_capacity(8 * multiplier);
    for _ in 0..multiplier {
        v.extend_from_slice(&base);
    }
    v
}

/// Helper to convert a slice of 8 &str into an array [&str; 8] without copying.
/// (Panics if len != 8; only used with exact-size chunks.)
fn slice8_to_array<'a>(a: &'a [&'a str]) -> [&'a str; 8] {
    assert_eq!(a.len(), 8);
    // SAFETY: we just asserted length is 8, and &[T] is layout-compatible with [T; 8].
    unsafe { *(a.as_ptr() as *const [&str; 8]) }
}

// fn bench_parse(c: &mut Criterion) {
//     // Choose a corpus size big enough to amortize overhead; adjust as you like.
//     // Total strings = 8 * MULT
//     const MULT: usize = 50_000; // => 400k total strings

//     // Pre-build corpus once outside the timed region.
//     let corpus = build_corpus(MULT);
//     let total_items = corpus.len() as u64;

//     let mut group = c.benchmark_group("deg_parse");

//     group.throughput(Throughput::Elements(total_items));

//     // -------------------------
//     // Benchmark: parse8_simd
//     // -------------------------
//     group.bench_function(BenchmarkId::new("simd_parse8", total_items), |b| {
//         b.iter_batched(
//             || &corpus, // input borrowed; no cloning
//             |data| {
//                 let mut acc = 0.0f64;
//                 for chunk in data.chunks_exact(8) {
//                     let arr = slice8_to_array(chunk);
//                     let out = unsafe { parse8_simd(black_box(arr)) };
//                     // Prevent the optimizer removing the work
//                     acc += out.iter().copied().sum::<f64>();
//                 }
//                 black_box(acc)
//             },
//             BatchSize::LargeInput,
//         );
//     });

//     // -------------------------
//     // Benchmark: std::str::parse
//     // -------------------------
//     group.bench_function(BenchmarkId::new("std_parse", total_items), |b| {
//         b.iter_batched(
//             || &corpus,
//             |data| {
//                 let mut acc = 0.0f64;
//                 for s in data.iter().copied() {
//                     // Standard library parse per string
//                     let v: f64 = black_box(s).parse().unwrap();
//                     acc += v;
//                 }
//                 black_box(acc)
//             },
//             BatchSize::LargeInput,
//         );
//     });

//     group.finish();
// }

pub fn pairs_bytes() -> Vec<u8> {
    let path = PathBuf::from("data/pairs.json");
    std::fs::read(&path).unwrap()
}

#[cfg(target_feature = "avx512f")]
fn bench_simd_avx512_parallel(c: &mut Criterion) {
    // Ensure the data is resident before timing starts.
    let pairs: Vec<u8> = pairs_bytes();
    let data = &pairs[..];

    c.bench_function("avx512_parse_json_multithreaded", |b| {
        b.iter(|| {
            // Black-box the input so LLVM can’t constant-fold anything.
            let input = black_box(data);
            let out = parse_pairs_json_parallel(input);
            black_box(out);
        });
    });
}

#[cfg(target_feature = "avx512f")]
fn bench_simd_avx512(c: &mut Criterion) {
    // Ensure the data is resident before timing starts.
    let pairs: Vec<u8> = pairs_bytes();
    let data = &pairs[..];

    c.bench_function("ours_avx512_parse_from_bytes", |b| {
        b.iter(|| {
            // Black-box the input so LLVM can’t constant-fold anything.
            let input = black_box(data);
            // Your parser returns CoordinatePairs directly.
            let out = parse_pairs_json_fixed_bytes(input);
            black_box(out);
        });
    });
}

#[cfg(target_feature = "avx512f")]
fn bench_simd_json(c: &mut Criterion) {
    // Immutable master bytes (read once, outside timed region).
    let master: Vec<u8> = pairs_bytes();
    let len = master.len();

    // Reusable working buffer; same length, allocated once.
    let mut working = vec![0u8; len];

    c.bench_function("simd_json_serde_from_slice", |b| {
        b.iter(|| {
            // Copy master -> working (no alloc, just a memcopy).
            working.copy_from_slice(&master);

            // simd-json needs &mut [u8] because it mutates in place.
            let out: CoordinatePairs = simd_json::serde::from_slice(black_box(&mut working[..]))
                .expect("simd-json parse failed");

            black_box(out);
        });
    });
}

// criterion_group!(benches, bench_parse);
#[cfg(target_feature = "avx512f")]
criterion_group!(
    benches,
    bench_simd_avx512_parallel,
    bench_simd_avx512,
    bench_simd_json
);

#[cfg(target_feature = "avx512f")]
criterion_main!(benches);
