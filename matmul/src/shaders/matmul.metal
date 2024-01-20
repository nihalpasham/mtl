
#include <metal_stdlib>
using namespace metal;

// Performs a standard matmul using a 2d grid of threads.
// there is some benefit in terms of speedup, if we use 2d threadgroups.
// Note: this example uses 32 x 32 threadgroups.
template <typename T, typename U>
kernel void simple_matmul(constant T *matrixA [[buffer(0)]],
                          constant T *matrixB [[buffer(1)]],
                          device T *matrixC [[buffer(2)]],
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

constant ushort SHARED_MEM = 1 << 10;

// Performs a tiled matmul using a 2d grid of threads.
// Note: this example uses 32 x 32 threadgroups.
kernel void tiled_matmul(constant half *matrixA [[buffer(0)]],
                         constant half *matrixB [[buffer(1)]],
                         device half *matrixC [[buffer(2)]],
                         uint2 grid_size [[threads_per_grid]],
                         uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                         uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                         uint2 threads_per_threadgroup [[threads_per_threadgroup]])
{
    // calculate this thread's position in the 2d grid
    uint2 thread_pos_in_grid =
        (threadgroup_position_in_grid * threads_per_threadgroup) + thread_position_in_threadgroup;
    // row and col are the x and y dimensions of this thread in the 2d grid.
    uint row = thread_pos_in_grid.y;
    uint col = thread_pos_in_grid.x;
    // x dimension of 2d grid
    uint n = grid_size.x;
    // statically allocate threadgroup memory to store tile A and tile B.
    // Although, this looks like each thread performs its own allocation.
    // that is *not* the case. Its a single allocation for a given threadgroup.
    threadgroup half tile_a[SHARED_MEM];
    threadgroup half tile_b[SHARED_MEM];
    // rename variables
    uint threadIDx = thread_position_in_threadgroup.x;
    uint threadIDy = thread_position_in_threadgroup.y;
    uint blockDimx = threads_per_threadgroup.x;
    // uint blockDimy = threads_per_threadgroup.y;

    // temporary accumulator
    half tmp = 0;
    // Each thread in a threadgroup loads an element (from A) and an element (from B) into the
    // allocated shared tile (i.e. threadgroup) memory.
    for (uint idx = 0; idx < n; idx += blockDimx)
    {
        // in every iteration, threads in a threadgroup load 2 tiles - 
        //      - from matrixA into tile_a (horizontally) and
        //      - from matrixB into tile_b (vertically)
        tile_a[threadIDy * blockDimx + threadIDx] = matrixA[row * n + idx + threadIDx];
        tile_b[threadIDy * blockDimx + threadIDx] = matrixB[idx * n + (threadIDy * n) + col];
        // wait till every thread in a given threadgroup loads its element before we compute the partial
        // dot products. This is an embarrassingly parallel operation.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // We perform a partial matmul using the loaded tiles, in each thread. 
        // More precisely, this thread performs the dotprod on a single row (from tile_a) and column 
        // (from tile_b) and writes the value to tmp
        for (uint elem = 0; elem < blockDimx; elem++)
        {
            tmp += tile_a[threadIDy * blockDimx + elem] * tile_b[elem * blockDimx + threadIDx];
        }
        // wait for all threads in the threadgroup to finish performing their respective dotprods using 
        // currently loaded tiles
        // in the next loop iteration, we'll load new tiles.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    matrixC[row * n  + col] = tmp;
}