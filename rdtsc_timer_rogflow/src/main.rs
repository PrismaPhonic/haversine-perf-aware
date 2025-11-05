use std::{mem::MaybeUninit, time::Instant};

use rdtsc_timer_rogflow::measure_cpu_freq;

fn main() {
    // let now = Instant::now();
    // measure_cpu_freq();
    let now = gettime();
    println!("{:?}", now);
}

pub fn gettime() -> u64 {
    let mut t = MaybeUninit::uninit();
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, t.as_mut_ptr());
    }
    let t = unsafe { t.assume_init() };
    t.tv_nsec as u64
}
