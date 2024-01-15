
#include <metal_stdlib>
using namespace metal;

// Performs a standard matmul using a 2d grid of threads. 
// there is some benefit in terms of speedup, if we use 2d threadgroups.
// Note: this example uses 32 x 32 threadgroups.
template <typename T, typename U>
kernel void simple_matmul(constant T *matrixA,
                          constant T *matrixB,
                          device T *matrixC,
                          U grid_size [[threads_per_grid]],
                          U threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                          U thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                          U threads_per_threadgroup [[threads_per_threadgroup]])
{
    // calculate this thread's position in the 2d grid
    uint2 thread_pos_in_grid =
        (threadgroup_position_in_grid * threads_per_threadgroup) + thread_position_in_threadgroup;
    // row and col are the x and y dimensions of this thread in the 2d grid.
    // Example: say row = 3 and col = 6, this thread will perform the dot product on (all) elements of
    //      - row 3 of matrixA and
    //      - col 6 of matrixB and
    //      - computes the sum of the products
    // essentially each thread produces one element of matrixC i.e (3,6)
    uint row = thread_pos_in_grid.x;
    uint col = thread_pos_in_grid.y;
    // calculate grid size
    uint n = grid_size.x;
    // initialize (row, col) element in matrixC to zero.
    matrixC[row * n + col] = 0;
    // perform the dotprod + sum and write value to matrixC
    for (uint elem = 0; elem < n; elem++)
    {
        matrixC[row * n + col] += matrixA[row * n + elem] * matrixB[elem * n + col];
    }
}

// function specialization. Metal doesn't support generic templates at runtime. 
// You need to explicitly instantiate the template for the specific data type you'll use (e.g., half or ushort).
template [[host_name("matmul_w_half")]] kernel void simple_matmul<half, uint2>(
    constant half *,
    constant half *,
    device half *,
    uint2,
    uint2,
    uint2,
    uint2);
template [[host_name("matmul_w_u16")]] kernel void simple_matmul<ushort, uint2>(
    constant ushort *,
    constant ushort *,
    device ushort *,
    uint2,
    uint2,
    uint2,
    uint2);
