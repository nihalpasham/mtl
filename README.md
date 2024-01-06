# mtl-rs
A playground for experimenting with the metal-rs bindings

## Available compute kernels 
- dotprod (comparing implementations using ushort and half types).
- matrixmul

## Usage

Launching metal compute kernel involves 2 steps. 

```sh
# every kernel is self contained i.e. is its own crate. Simply `cd` into a kernel's directory and run
# compile shader to intermediate representation using the metal utility
xcrun -sdk macosx metal -c ./src/shaders/dotprod.metal -o ./src/shaders/dotprod.air
# compile .air file to .metallib - which I believe is LLVM IR (need confirmation)
xcrun -sdk macosx metallib ./src/metal/matrixprod.air -o ./src/metal/matrixprod.metallib
# run the rust binary to launch the kernel and examine the output.
cargo run
```
## Example output for dotprod

```sh

```

