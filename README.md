# mtl-rs
A playground for experimenting with the metal-rs bindings

## Available compute kernels 
- dotprod (ushort Vs. half type impls).
- matrixmul

## Usage

Launching metal compute kernel involves 2 steps. 

```sh
# every kernel is self contained i.e. is its own crate. Simply `cd` into a kernel's directory and run
# the following, which compiles our shader to an intermediate representation using the metal utility
xcrun -sdk macosx metal -c ./src/shaders/dotprod.metal -o ./src/shaders/dotprod.air

# next, compile the .air file to generate a .metallib file - which I believe is LLVM IR (need confirmation)
xcrun -sdk macosx metallib ./src/metal/matrixprod.air -o ./src/metal/matrixprod.metallib

# lastly, run the rust binary to launch the kernel and examine its output.
cargo run
```
## Example output for dotprod

```sh
cargo run
   Compiling mtl v0.1.0 (/Users/nihal.pasham/devspace/metal/mtl/dotprod)
    Finished dev [unoptimized + debuginfo] target(s) in 0.20s
     Running `target/debug/mtl`
Dotprod on CPU
      Done in 15.65ms
Dotprod on CPU - parallel
      Done in 3.08ms
Dotprod on GPU
      Actual time spent performing dotprod on GPU
          Done in 1.05ms
      Total time taken in 21.71ms (includes kernel launch and result retreival)

*** verify that all 3 ops produce the same result ***
cpu:     [0, 135, 342, 91, 528], [360, 528, 414, 351, 160]
cpu_par: [0, 135, 342, 91, 528], [360, 528, 414, 351, 160]
gpu:     [0, 135, 342, 91, 528], [360, 528, 414, 351, 160]
```

