#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm_lfence, _rdtsc};
use std::mem::MaybeUninit;

use libc::{CLOCK_MONOTONIC_RAW, clock_gettime, timespec};

const ONE_SEC_NS: u64 = 1_000_000_000;

#[inline(always)]
pub fn cpu_timer() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_lfence();
        let t = _rdtsc();
        _mm_lfence();
        t
    }
}

#[repr(C)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct RawInstant {
    secs: u32,
    ns: u32,
}

impl RawInstant {
    // Technically "unsafe" but this is exactly how Instant in the std library
    // works anyways, it just uses a different clock source (realtime);
    pub fn now() -> Self {
        let mut ts: timespec = unsafe { std::mem::zeroed() };
        unsafe { clock_gettime(CLOCK_MONOTONIC_RAW, &mut ts) };
        Self {
            secs: ts.tv_sec as u32,
            ns: ts.tv_nsec as u32,
        }
    }

    pub fn nanos_elapsed(&self, start: RawInstant) -> u64 {
        // Adjust so start is at zero seconds, and end (self) is left adjusted
        // so we can predictably fit all in nanos scaling.
        let end_secs = (self.secs - start.secs) as u64;
        // We assume start secs as zero justified at this point.
        let end_nanos = end_secs
            .saturating_mul(1_000_000_000)
            .saturating_add(self.ns as u64);

        end_nanos - start.ns as u64
    }
}

pub fn measure_cpu_freq() -> f64 {
    let os_start = RawInstant::now();
    let cpu_start = cpu_timer();
    let os_end;

    loop {
        let now_os = RawInstant::now();
        let elapsed_ns = now_os.nanos_elapsed(os_start);
        if elapsed_ns >= ONE_SEC_NS {
            os_end = elapsed_ns;
            break;
        }
    }

    let cpu_end = cpu_timer();

    let os_elapsed_secs = os_end as f64 * 1e-9;
    let cpu_elapsed = cpu_end - cpu_start;

    cpu_elapsed as f64 / os_elapsed_secs
}

// A profiler with a set number of profile spots known at compile time.
pub struct Profiler<const N: usize> {
    labels: [MaybeUninit<&'static str>; N],
    tsc: [MaybeUninit<u64>; N],
    parent_label: &'static str,
    start: u64,
    head: usize,
}

impl<const N: usize> Profiler<N> {
    pub fn new(label: &'static str) -> Self {
        Self {
            labels: [MaybeUninit::uninit(); N],
            tsc: [MaybeUninit::uninit(); N],
            parent_label: label,
            start: cpu_timer(),
            head: 0,
        }
    }

    // TODO: There must be a better way to do this. Right now this requires very
    // correct use - it should be impossible instead to get undefined behavior
    // by not calling these in the perfect sequence.
    //
    // Does NOT currently advance head. Must call record_end before next call of record_start.
    pub fn record_start(&mut self) {
        // This is naive - it assumes correct use.
        debug_assert!(self.head < N);

        self.tsc[self.head].write(cpu_timer());
    }

    pub fn record_end(&mut self, label: &'static str) {
        // This is naive - it assumes correct use.
        debug_assert!(self.head < N);

        let end = cpu_timer();
        let ptr: &mut u64 = unsafe { &mut *self.tsc[self.head].as_mut_ptr() };
        let diff = end - *ptr;
        *ptr = diff;
        self.labels[self.head].write(label);
        self.head += 1;
    }

    /// Consumes self and prints metrics out.
    pub fn finalize(self) {
        // First we consume our MaybeUninit arrays.
        // For now we require ALL get used.
        debug_assert!(self.head == N);

        let end = cpu_timer();
        let total_tsc_elapsed = end - self.start;
        let labels: [&'static str; N] =
            unsafe { *(self.labels.as_ptr() as *const [&'static str; N]) };
        let tsc: [u64; N] = unsafe { *(self.tsc.as_ptr() as *const [u64; N]) };

        let cpu_freq = measure_cpu_freq();

        println!("{}:\n", self.parent_label);
        println!(
            "Total time: {:.4}ms (CPU freq {cpu_freq})",
            (total_tsc_elapsed as f64 / cpu_freq) * 1e3
        );

        for i in 0..N {
            print_time_elapsed(labels[i], total_tsc_elapsed, tsc[i]);
        }

        println!("\n");
    }
}

#[inline(always)]
fn print_time_elapsed(label: &str, total_tsc_elapsed: u64, elapsed: u64) {
    let percent = 100.0 * elapsed as f64 / total_tsc_elapsed as f64;
    println!("{label}: {elapsed} ({:.2})", percent);
}

#[macro_export]
macro_rules! time {
    // Explicit label
    ($prof:expr, $label:expr, $expr:expr) => {{
        let __p = &mut ($prof);
        __p.record_start();
        let __res = $expr;
        __p.record_end($label);
        __res
    }};
    // Label defaults to the source expression
    ($prof:expr, $expr:expr) => {{ $crate::time!($prof, stringify!($expr), $expr) }};
}
