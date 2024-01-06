use core::{panic, slice};
use std::{mem::{transmute, self}, ops::Mul, time::Instant};

use metal::{Device, MTLResourceOptions, MTLSize};
use rand::Rng;
use rayon::prelude::*;

const LIB_DATA: &[u8] = include_bytes!("./shaders/dotprod.metallib");

/// Kernel hostnames
pub enum Kernels<'a> {
    UshortDP(&'a str),
    H16DP(&'a str),
}

/// Compute the dotprod of two vetcors of type U
pub fn dotprod<U: Mul<Output = U> + Copy>(a: &Vec<U>, b: &Vec<U>, kernel: Kernels) -> Vec<U> {
    // get an device instance - logical representation of a gpu. Default on apple silicon is the AGX gpu
    let device = Device::system_default().unwrap();
    // create a new command queue to hold command buffers
    let queue = device.new_command_queue();
    // set up gpu buffers
    let gpu_buffer_a;
    let gpu_buffer_b;
    let gpu_buffer_res;

    match a.len() == b.len() {
        true => {
            gpu_buffer_a = device.new_buffer_with_data(
                unsafe { transmute(a.as_ptr()) },
                a.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            gpu_buffer_b = device.new_buffer_with_data(
                unsafe { transmute(b.as_ptr()) },
                b.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let len = a.len() as u64;
            let size = len * mem::size_of::<U>() as u64;
            gpu_buffer_res =
                device.new_buffer(size, MTLResourceOptions::StorageModeShared);
        }
        false => {
            panic!("cant compute dotprod for vectors of different lengths")
        }
    }
    // retrieve the shader function
    let lib = device.new_library_with_data(LIB_DATA).unwrap();
    let function = match kernel {
        Kernels::H16DP(s) => lib.get_function(s, None).unwrap(),
        Kernels::UshortDP(s) => lib.get_function(s, None).unwrap(),
    };
    // create a compute pipeline. Think of this as the state the gpu needs to be in before performing the actual computation.
    let compute_pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap();
    // we can now create a command buffer and a compute encoder to encode commands into the buffer
    let cmd_buffer = queue.new_command_buffer();
    let compute_encoder = cmd_buffer.new_compute_command_encoder();
    compute_encoder.set_compute_pipeline_state(&compute_pipeline);
    compute_encoder.set_buffers(
        0,
        &[
            Some(&gpu_buffer_a),
            Some(&gpu_buffer_b),
            Some(&gpu_buffer_res),
        ],
        &[0; 3],
    );
    // dispatch - we can specify the number of gpu threads we'd like to launch for this computation.
    // we can also specify the dimensions (or layout) of our thread grid.
    let thread_execution_width = compute_pipeline.thread_execution_width(); // this is simd width
    let _threads_per_threadgroup = compute_pipeline.max_total_threads_per_threadgroup(); // threads per threadgroup

    // calculate threads per grid
    let _grid_width = (a.len() / 1000) as u64; // use for 2d grid of threads
    let _grid_height = (a.len() / 1000) as u64; // use for 2d grid of threads

    let threads_per_grid = MTLSize::new(a.len() as u64, 1, 1);
    let threads_per_thread_group = MTLSize::new(
        thread_execution_width,
        1, // threads_per_threadgroup / thread_execution_width,
        1,
    );
    compute_encoder.dispatch_threads(threads_per_grid, threads_per_thread_group);
    compute_encoder.end_encoding();
    // commmit buffer and do a blocking wait until the gpu is done with this work.
    println!("      Actual time spent performing dotprod on GPU");
    let instant = Instant::now();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
    println!("          Done in {:.2?}", instant.elapsed());

    // get the result and print
    let ptr = gpu_buffer_res.contents() as *const U;
    let len = gpu_buffer_res.length() as usize / mem::size_of::<U>();
    let slice = unsafe { slice::from_raw_parts(ptr, len) };
    slice.to_vec()
}

/// Perform a straightforward dotprod on a cpu. This is much a for loop that sequentially iterates through each element.
pub fn run_cpu<U: Mul<Output = U> + Copy>(a: &Vec<U>, b: &Vec<U>) -> Vec<U> {
    println!("Dotprod on CPU");
    let instant = Instant::now();
    let res: Vec<U> = a.iter().zip(b).map(|(a, b)| *a * *b).collect();
    println!("      Done in {:.2?}", instant.elapsed());
    res
}

/// Parallelizing dotprod computation using rayon on a cpu. Works by leveraging multiple threads.
/// U needs to Send and Sync
pub fn run_cpu_par<U: Mul<Output = U> + Copy + Send + Sync>(a: &Vec<U>, b: &Vec<U>) -> Vec<U> {
    println!("Dotprod on CPU - parallel");
    let instant = Instant::now();
    let res: Vec<U> = a.par_iter().zip(b).map(|(a, b)| *a * *b).collect();
    println!("      Done in {:.2?}", instant.elapsed());
    res
}

/// Launch a dotprod kernel on the GPU.
pub fn run_gpu<U: Mul<Output = U> + Copy>(a: &Vec<U>, b: &Vec<U>, kernel: &str) -> Vec<U> {
    let kernel_hostname = match kernel {
        "dotprod_ushort" => Kernels::UshortDP(kernel),
        "dotprod_half" => Kernels::H16DP(kernel),
        _ => unimplemented!(),
    };
    println!("Dotprod on GPU");
    let instant = Instant::now();
    let res = dotprod(a, b, kernel_hostname);
    println!(
        "      Total time taken in {:.2?} (includes kernel launch and result retreival)",
        instant.elapsed()
    );
    res
}

fn main() {
    let num_elems = 1_000_000usize;
    let mut rng = rand::thread_rng();
    let a = (0..num_elems)
        .into_iter()
        .map(|_| (rng.gen_range(0..32u16)))
        .collect::<Vec<u16>>();
    let b = (0..num_elems)
        .into_iter()
        .map(|_| (rng.gen_range(0..32u16)))
        .collect::<Vec<u16>>();

    let cpu = run_cpu(&a, &b);
    let cpu_par = run_cpu_par(&a, &b);
    let gpu = run_gpu(&a, &b, "dotprod_ushort");

    println!("\n*** verify that all 3 ops produce the same result ***");
    println!(
        "cpu:     {:?}, {:?}",
        &cpu[0..5],
        &cpu[num_elems - 5..num_elems]
    );
    println!(
        "cpu_par: {:?}, {:?}",
        &cpu_par[0..5],
        &cpu_par[num_elems - 5..num_elems]
    );
    println!(
        "gpu:     {:?}, {:?}",
        &gpu[0..5],
        &gpu[num_elems - 5..num_elems]
    );
}
