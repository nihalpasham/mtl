### Usage

Launching a metal `compute` kernel is a 2 step process. 

```sh
# every kernel is self contained i.e. is its own crate. Simply `cd` into a kernel's directory and run
# the following, which compiles our shader to an intermediate representation using the metal utility
xcrun -sdk macosx metal -c ./src/shaders/matmul.metal -o ./src/shaders/matmul.air

# next, compile the .air file to generate a .metallib file - which I believe is LLVM IR (need confirmation)
xcrun -sdk macosx metallib ./src/metal/matmul.air -o ./src/metal/matmul.metallib

# lastly, run the rust binary to launch the kernel and examine its output.
cargo run
```
### Example output for matmul

```sh
   Compiling matmul v0.1.0 (/Users/nihal.pasham/devspace/metal/mtl/matmul)
    Finished dev [unoptimized + debuginfo] target(s) in 0.16s
     Running `target/debug/matmul`
a: [0.33935547, 0.010612488, 0.7109375, 0.9404297, 0.6191406], [0.5522461, 0.9848633, 0.55859375, 0.95214844, 0.39819336]
b: [0.17114258, 0.051330566, 0.33325195, 0.85546875, 0.38232422], [0.34301758, 0.9003906, 0.41357422, 0.08544922, 0.6113281]

____*** matrix multiplication - matrices of `width 1,024 x height 1,024` elements of type `half::binary16::f16` ***___

Naive Matmul on GPU
      Actual time spent performing matmul on GPU
          Done in 46.40ms
      Total time taken - 76.89ms (includes kernel launch and result retreival)
Tiled Matmul on GPU
      Actual time spent performing matmul on GPU
          Done in 5.87ms
      Total time taken - 7.21ms (includes kernel launch and result retreival)
Naive Matmul on CPU
      Done in - 29.43s

____*** verify cpu & gpu produce the same result ***____

cpu_naive:     [263.0, 256.0, 260.25, 256.25, 257.25], [251.625, 244.5, 247.5, 250.25, 243.0]
gpu_naive:     [263.0, 256.0, 260.25, 256.25, 257.25], [251.875, 244.625, 247.5, 250.125, 243.0]
gpu_tiled:     [263.0, 256.0, 260.25, 256.25, 257.25], [251.875, 244.625, 247.5, 250.125, 243.0]
```
### GPU speedup Vs. CPU using a naive implementation:

$$(29.43 * 1000/46.40) = 634.26$$ 

### Tiled GPU speedup Vs. GPU using a naive implementation:

$$(46.40/5.87) = 7.90$$ 

### Precision and variance

CPU and GPU matmul's do not produce the exact same result. This can be attributed to floating point types and their precision. The above is for `f16` i.e. 16 bit floating point numbers with half precision.