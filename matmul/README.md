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
    Finished dev [unoptimized + debuginfo] target(s) in 0.15s
     Running `target/debug/matmul`
a: [0.41870117, 0.0039863586, 0.22045898, 0.9082031, 0.7939453], [0.39648438, 0.1005249, 0.6303711, 0.5258789, 0.2286377]
b: [0.40527344, 0.6220703, 0.089416504, 0.25561523, 0.6484375], [0.14367676, 0.43969727, 0.5385742, 0.15527344, 0.0692749]

____*** matrix multiplication - matrices of `width 1,024 x height 1,024` elements of type `half::binary16::f16` ***___

Matmul on GPU
      Actual time spent performing matmul on GPU
          Done in 49.45ms
      Total time taken - 97.03ms (includes kernel launch and result retreival)
Matmul on CPU
      Done in - 29.44s

____*** verify cpu & gpu produce the same result ***____

cpu:     [259.5, 246.0, 251.625, 249.125, 258.75], [248.625, 237.75, 257.5, 248.0, 259.0]
gpu:     [259.5, 246.125, 251.875, 248.875, 258.5], [248.625, 237.75, 257.5, 248.125, 258.75]
```
### GPU Vs. CPU speedup:

```math
(29.44s * 1000)ms \over 49.45ms \right = \left 595.35
```

### Precision and variance

CPU and GPU matmul's do not produce the exact same result. This can be attributed to floating point types and their precision. The above is for `f16` i.e. 16 bit floating point numbers with half precision.