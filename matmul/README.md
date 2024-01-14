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
    Finished dev [unoptimized + debuginfo] target(s) in 0.17s
     Running `target/debug/matmul`
a: [4, 7, 1, 4, 9], [9, 7, 4, 4, 5]
b: [4, 9, 9, 3, 9], [9, 1, 1, 8, 1]

____*** matrix multiplication - matrices of `width 1,024 x height 1,024` elements of type `u32` ***___

Matmul on GPU
      Actual time spent performing matmul on GPU
          Done in 41.59ms
      Total time taken - 65.67ms (includes kernel launch and result retreival)
Matmul on CPU
      Done in - 20.06s

*** verify cpu & gpu produce the same result ***
cpu:     [21010, 20946, 20842, 20345, 21062], [20552, 20594, 21033, 20838, 20439]
gpu:     [21010, 20946, 20842, 20345, 21062], [20552, 20594, 21033, 20838, 20439]
```

