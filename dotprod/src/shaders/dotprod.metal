
#include <metal_stdlib>
using namespace metal;

// use a 1d thread grid - for the fastest impl of dotprod (assuming you use 1d threadgroup size of width 32)
//
template<typename T, typename U>
kernel void dotprod(constant T *arrayA [[buffer(0)]],
                    constant T *arrayB [[buffer(1)]],
                    device T *result [[buffer(2)]],
                    U pos [[thread_position_in_grid]])
{
    result[pos] = arrayA[pos] * arrayB[pos];
}

// function specialization. Metal doesn't support generic templates at runtime. 
// You need to explicitly instantiate the template for the specific data type you'll use (e.g., half or uint).
template [[host_name("dotprod_ushort")]] kernel void dotprod<ushort, uint>(constant ushort *, constant ushort *, device ushort *, uint);
template [[host_name("dotprod_half")]] kernel void dotprod<half, uint>(constant half *, constant half *, device half *, uint);

// for a 2d grid of threads - gpu vs cpu (parallel) get slower as we increase the number of elements
// when using this kernel with arrays of 10_000_000 elements, this should return all zeroes. i.e.
// n == 10_000
//
// kernel void dotprod(constant uint *arrayA [[buffer(0)]],
//                     constant uint *arrayB [[buffer(1)]],
//                     device uint *result [[buffer(2)]],
//                     uint2 grid_size [[threads_per_grid]],
//                     uint2 pos [[thread_position_in_grid]])
// {
//     // get grid width
//     uint n = grid_size.x;
//     if (n == 10000) {return;}
//     result[pos.y * n + pos.x] = arrayA[pos.y * n + pos.x] * arrayB[pos.y * n + pos.x];
// }

// finally got 2d working. The issue - to get the grid_size, use attribute [[threads_per_grid]] not [[grid_size]]
// this only works with dispatch_thread_groups()
//
// kernel void dotprod(constant uint *arrayA,
//                     constant uint *arrayB,
//                     device uint *result,
//                     uint2 grid_size [[threads_per_grid]],
//                     uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
//                     uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
//                     uint2 threads_per_threadgroup [[threads_per_threadgroup]])
// {
//     // calculate thread position in grid
//     uint2 pos =
//         (threadgroup_position_in_grid * threads_per_threadgroup) +
//         thread_position_in_threadgroup;
//     // calculate grid size
//     uint n = grid_size.x;
//     result[pos.y * n + pos.x] = arrayA[pos.y * n + pos.x] * arrayB[pos.y * n + pos.x];
// }