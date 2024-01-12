
#include <metal_stdlib>
using namespace metal;

kernel void simple_matmul(constant uint *matrixA,
                    constant uint *matrixB,
                    device uint *matrixC,
                    uint2 grid_size [[threads_per_grid]],
                    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                    uint2 threads_per_threadgroup [[threads_per_threadgroup]])
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
    matrixC[row * n + col ] = 0;
    // perform the dotprod + sum and write value to matrixC  
    for (uint elem = 0; elem < n; elem++) {
        matrixC[row * n + col] += matrixA[row * n + elem] * matrixB[elem * n + col];
    }
}

