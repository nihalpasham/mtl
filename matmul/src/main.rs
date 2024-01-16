use core::{mem, mem::transmute, slice};
use std::{
    fmt::Debug,
    ops::{AddAssign, Index, IndexMut, Mul},
    time::Instant,
};

use half::f16;
use metal::{Device, MTLResourceOptions, MTLSize};
use rand::Rng;
use separator::Separatable;

const LIB_DATA: &[u8] = include_bytes!("./shaders/matmul.metallib");

const MTX_WIDTH: usize = 1024;
const MTX_HEIGHT: usize = 1024;

/// Kernel hostnames
pub enum Kernels<'a> {
    SimpleMatMul(&'a str),
    TiledMatMul(&'a str),
}

#[derive(Clone, Debug)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    elems: Vec<T>,
}

impl<T: Copy> Matrix<T> {
    pub fn new(rows: usize, cols: usize, elems: &[T]) -> Self {
        assert_eq!(
            elems.len(),
            rows * cols,
            "no: of elems must equal (rows * cols)"
        );
        Matrix {
            rows,
            cols,
            elems: elems.to_vec(),
        }
    }
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        let idx = self.cols * row + col;
        &self.elems[idx]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let idx = self.cols * row + col;
        &mut self.elems[idx]
    }
}

impl<T> Mul for Matrix<T>
where
    T: AddAssign + Mul<Output = T> + Copy + Default + Debug,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "cant multiply matrices, check dims");
        // instantiate the resultant matrix
        let rows = self.rows;
        let cols = rhs.cols;
        let size = rows * cols;
        let elems: Vec<T> = (0..size).into_iter().map(|_| T::default()).collect();

        let mut result = Matrix::new(rows, cols, elems.as_slice());

        // for each row in self
        for row in 0..self.rows {
            // and for every column in the rhs matrix
            for col in 0..rhs.cols {
                let mut tmp: T = T::default();
                // compute the sum of the dotprod of elems
                for elem in 0..self.rows {
                    tmp += self.elems[row * self.cols + elem] * rhs.elems[elem * rhs.cols + col];
                }
                result[(row, col)] = tmp;
            }
        }
        result
    }
}

/// Multiply matrix `a` with `b` using the GPU
pub fn matmul<U: Mul<Output = U> + Copy>(
    a: &Matrix<U>,
    b: &Matrix<U>,
    kernel: Kernels,
) -> Matrix<U> {
    // get an device instance - logical representation of a gpu. Default on apple silicon is the AGX gpu
    let device = Device::system_default().unwrap();
    // create a new command queue to hold command buffers
    let queue = device.new_command_queue();
    // set up gpu buffers
    let gpu_buffer_a;
    let gpu_buffer_b;
    let gpu_buffer_res;

    match a.cols == b.rows {
        true => {
            let len = a.elems.len() as u64;
            let size = len * mem::size_of::<U>() as u64;
            gpu_buffer_a = device.new_buffer_with_data(
                unsafe { transmute(a.elems.as_ptr()) },
                size,
                MTLResourceOptions::StorageModeShared,
            );
            gpu_buffer_b = device.new_buffer_with_data(
                unsafe { transmute(b.elems.as_ptr()) },
                size,
                MTLResourceOptions::StorageModeShared,
            );
            gpu_buffer_res = device.new_buffer(size, MTLResourceOptions::StorageModeShared);
        }
        false => {
            panic!("cant multiply matrices, please check dims")
        }
    }
    // retrieve the shader function
    let lib = device.new_library_with_data(LIB_DATA).unwrap();
    let function = match kernel {
        Kernels::TiledMatMul(s) => lib.get_function(s, None).unwrap(),
        Kernels::SimpleMatMul(s) => lib.get_function(s, None).unwrap(),
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
    let threads_per_threadgroup = compute_pipeline.max_total_threads_per_threadgroup(); // threads per threadgroup

    // calculate threads per grid
    let grid_width = (a.elems.len() / MTX_WIDTH) as u64; // use for 2d grid of threads
    let grid_height = (a.elems.len() / MTX_HEIGHT) as u64; // use for 2d grid of threads

    let threads_per_grid = MTLSize::new(grid_width, grid_height, 1);
    let threads_per_thread_group = MTLSize::new(
        thread_execution_width,
        threads_per_threadgroup / thread_execution_width,
        1,
    );
    compute_encoder.dispatch_threads(threads_per_grid, threads_per_thread_group);
    compute_encoder.end_encoding();
    // commmit buffer and do a blocking wait until the gpu is done with this work.
    println!("      Actual time spent performing matmul on GPU");
    let instant = Instant::now();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
    println!("          Done in {:.2?}", instant.elapsed());

    // get the result and print
    let ptr = gpu_buffer_res.contents() as *const U;
    let len = gpu_buffer_res.length() as usize / mem::size_of::<U>();
    let elems = unsafe { slice::from_raw_parts(ptr, len) };
    Matrix::new(a.rows, b.cols, elems)
}

/// Launch a matmul kernel on the GPU.
pub fn run_gpu<U: Mul<Output = U> + Copy>(a: &Matrix<U>, b: &Matrix<U>, kernel: &str) -> Matrix<U> {
    let kernel_hostname = match kernel {
        "matmul_w_half" => Kernels::SimpleMatMul(kernel),
        "matmul_w_u16" => Kernels::SimpleMatMul(kernel),
        _ => unimplemented!(),
    };
    println!("Matmul on GPU");
    let instant = Instant::now();
    let res = matmul(a, b, kernel_hostname);
    println!(
        "      Total time taken - {:.2?} (includes kernel launch and result retreival)",
        instant.elapsed()
    );
    res
}

fn main() {
    let m_rows = MTX_WIDTH;
    let m_cols = MTX_HEIGHT;
    let mtx_size = m_rows * m_cols;
    let mut rng = rand::thread_rng();
    let a = (0..mtx_size)
        .into_iter()
        .map(|_| (rng.gen_range(0.0..1.0)))
        .map(|a| half::f16::from_f32(a))
        .collect::<Vec<f16>>();
    let b = (0..mtx_size)
        .into_iter()
        .map(|_| (rng.gen_range(0.0..1.0)))
        .map(|b| half::f16::from_f32(b))
        .collect::<Vec<f16>>();
    let mtx_a = Matrix::new(m_rows, m_cols, a.as_slice());
    let mtx_b = Matrix::new(m_rows, m_cols, b.as_slice());
    
    println!("a: {:?}, {:?}", &a[..5], &a[mtx_size - 5..mtx_size]);
    println!("b: {:?}, {:?}", &b[..5], &b[mtx_size - 5..mtx_size]);
    println!(
    "\n____*** matrix multiplication - matrices of `width {} x height {}` elements of type `{}` ***___\n",
    m_rows.separated_string(), m_cols.separated_string(), std::any::type_name::<f16>(),
);

    let gpu = run_gpu(&mtx_a, &mtx_b, "matmul_w_half");

    println!("Matmul on CPU");
    let instant = Instant::now();
    let cpu = mtx_a * mtx_b;
    println!("      Done in - {:.2?}", instant.elapsed());

    println!("\n____*** verify cpu & gpu produce the same result ***____\n");
    println!(
        "cpu:     {:?}, {:?}",
        &cpu.elems[0..5],
        &cpu.elems[mtx_size - 5..mtx_size]
    );
    println!(
        "gpu:     {:?}, {:?}",
        &gpu.elems[0..5],
        &gpu.elems[mtx_size - 5..mtx_size]
    );
}
