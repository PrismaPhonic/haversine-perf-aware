#![feature(portable_simd, duration_millis_float)]

use json_coord_parser::reference_haversine;
#[cfg(target_feature = "avx512f")]
use json_coord_parser::{haversine_from_json_coords_parallel, parse_pairs_json_parallel};
use memmap2::Mmap;
use rdtsc_timer_rogflow::Profiler;
use rdtsc_timer_rogflow::cpu_timer;
use rdtsc_timer_rogflow::measure_cpu_freq;
use rdtsc_timer_rogflow::time;

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

fn pairs_bytes() -> Mmap {
    let file = File::open("data/pairs.json").unwrap();
    unsafe { Mmap::map(&file).unwrap() }
}

fn main() {
    // haversine_while_parsing();

    // haversine_after_parsing();
    profile_instant();
}

fn haversine_while_parsing() {
    let mut profiler: Profiler<2> = Profiler::new("Haversine while parsing");

    let pairs = time!(profiler, pairs_bytes());

    #[cfg(target_feature = "avx512f")]
    let _out = time!(profiler, haversine_from_json_coords_parallel(&pairs));

    profiler.finalize();
}

fn haversine_after_parsing() {
    let mut profiler: Profiler<3> = Profiler::new("Haversine after parsing");

    let pairs = time!(profiler, pairs_bytes());

    #[cfg(target_feature = "avx512f")]
    let out = time!(profiler, parse_pairs_json_parallel(&pairs));

    #[cfg(target_feature = "avx512f")]
    time!(profiler, "generate_answers", {
        let mut _answers: Vec<f64> = Vec::with_capacity(out.pairs.len());

        for row in out.pairs {
            _answers.push(reference_haversine(row.x0, row.y0, row.x1, row.y1));
        }
    });

    profiler.finalize();
}

fn profile_instant() {
    let mut profiler: Profiler<2> = Profiler::new("Instant vs rdtsc");

    time!(profiler, "Instant::now", {
        let start = Instant::now();
        let end = Instant::now();
        let diff = end - start;
    });

    time!(profiler, "rdtsc", {
        let start = cpu_timer();
        let end = cpu_timer();
        let diff = end - start;
    });

    profiler.finalize();
}
