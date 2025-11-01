use json_coord_parser::reference_haversine;
#[cfg(target_feature = "avx512f")]
use json_coord_parser::{haversine_from_json_coords_parallel, parse_pairs_json_parallel};
use rdtsc_timer_rogflow::cpu_timer;
use rdtsc_timer_rogflow::measure_cpu_freq;

use std::path::PathBuf;

fn pairs_bytes() -> Vec<u8> {
    let path = PathBuf::from("data/pairs.json");
    std::fs::read(&path).unwrap()
}

fn main() {
    #[cfg(target_feature = "avx512f")]
    haversine_while_parsing();

    #[cfg(target_feature = "avx512f")]
    haversine_after_parsing();
}

#[cfg(target_feature = "avx512f")]
fn haversine_while_parsing() {
    let start = cpu_timer();

    let pairs = pairs_bytes();

    let after_read = cpu_timer();

    let _out = haversine_from_json_coords_parallel(&pairs);

    let after_parsing = cpu_timer();

    let cpu_freq = measure_cpu_freq();

    let total_tsc_elapsed = after_parsing - start;

    println!("Haversine while Parsing:\n",);
    println!(
        "Total time: {:.4}ms (CPU freq {cpu_freq})",
        (total_tsc_elapsed as f64 / cpu_freq) * 1e3
    );

    print_time_elapsed("Read", total_tsc_elapsed, start, after_read);
    print_time_elapsed("Parse + Sum", total_tsc_elapsed, after_read, after_parsing);
    println!("\n");
}

fn haversine_after_parsing() {
    let start = cpu_timer();

    let pairs = pairs_bytes();

    let after_read = cpu_timer();

    #[cfg(target_feature = "avx512f")]
    let out = parse_pairs_json_parallel(&pairs);

    let after_parsing = cpu_timer();

    #[cfg(target_feature = "avx512f")]
    let mut _answers: Vec<f64> = Vec::with_capacity(out.pairs.len());

    #[cfg(target_feature = "avx512f")]
    for row in out.pairs {
        _answers.push(reference_haversine(row.x0, row.y0, row.x1, row.y1));
    }

    let after_math = cpu_timer();

    let cpu_freq = measure_cpu_freq();

    let total_tsc_elapsed = after_math - start;

    println!("Haversine after Parsing:\n",);
    println!(
        "Total time: {:.4}ms (CPU freq {cpu_freq})",
        (total_tsc_elapsed as f64 / cpu_freq) * 1e3
    );

    print_time_elapsed("Read", total_tsc_elapsed, start, after_read);
    print_time_elapsed("Parse", total_tsc_elapsed, after_read, after_parsing);
    print_time_elapsed("Parse Sum", total_tsc_elapsed, after_parsing, after_math);
}

fn print_time_elapsed(label: &str, total_tsc_elapsed: u64, begin: u64, end: u64) {
    let elapsed = end - begin;
    let percent = 100.0 * elapsed as f64 / total_tsc_elapsed as f64;
    println!("{label}: {elapsed} ({:.2})", percent);
}
