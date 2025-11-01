#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm_lfence, _rdtsc};

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
