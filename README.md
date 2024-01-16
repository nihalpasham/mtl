# mtl-rs
A playground for experimenting with Apple silicon GPUs and metal-rs bindings

### Available compute kernels 
- dotprod (ushort Vs. half type impls).
- matmul (f16 Vs. u16 type impls)

### Usage

Launching a metal `compute` kernel is a 2 step process. 

```sh
# every kernel is self contained i.e. is its own crate. Simply `cd` into a kernel's directory and run
# the following, which compiles our shader to an intermediate representation using the metal utility
xcrun -sdk macosx metal -c ./src/shaders/dotprod.metal -o ./src/shaders/dotprod.air

# next, compile the .air file to generate a .metallib file - which I believe is LLVM IR (need confirmation)
xcrun -sdk macosx metallib ./src/metal/matrixprod.air -o ./src/metal/matrixprod.metallib

# lastly, run the rust binary to launch the kernel and examine its output.
cargo run
```
### Example output for dotprod

```sh
cargo run
    Finished dev [unoptimized + debuginfo] target(s) in 0.01s
     Running `target/debug/mtl`

____*** vector dotprod of 1,000,000 elements of type `ushort` ***___

Dotprod on CPU
      Done in 15.48ms
Dotprod on CPU - parallel
      Done in 3.38ms
Dotprod on GPU
      Actual time spent performing dotprod on GPU
          Done in 1.00ms
      Total time taken - 24.50ms (includes kernel launch and result retreival)

*** verify that all 3 ops produce the same result ***
cpu:     [391, 0, 35, 116, 810], [48, 155, 126, 48, 0]
cpu_par: [391, 0, 35, 116, 810], [48, 155, 126, 48, 0]
gpu:     [391, 0, 35, 116, 810], [48, 155, 126, 48, 0]

____*** vector dotprod of 1,000,000 elements of type `f16` ***___

Dotprod on CPU
      Done in 18.75ms
Dotprod on CPU - parallel
      Done in 3.37ms
Dotprod on GPU
      Actual time spent performing dotprod on GPU
          Done in 869.25µs
      Total time taken - 2.25ms (includes kernel launch and result retreival) # interesting, looks like metal reuses most objects or resources instantiated from the previous dispatch call (i.e. device, queue etc.). 

*** verify that all 3 ops produce the same result ***
cpu:     [0.72802734, 0.0037574768, 0.0947876, 0.16601563, 0.26367188], [0.31933594, 0.2919922, 0.12042236, 0.20458984, 0.30151367]
cpu_par: [0.72802734, 0.0037574768, 0.0947876, 0.16601563, 0.26367188], [0.31933594, 0.2919922, 0.12042236, 0.20458984, 0.30151367]
gpu:     [0.72802734, 0.0037574768, 0.0947876, 0.16601563, 0.26367188], [0.31933594, 0.2919922, 0.12042236, 0.20458984, 0.30151367]
```

