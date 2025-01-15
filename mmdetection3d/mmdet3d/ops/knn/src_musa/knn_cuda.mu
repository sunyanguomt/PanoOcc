// Modified from https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/pointops/src/knnquery_heap

#include <cmath>
#include <cstdio>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__device__ void swap_float(float *x, float *y)
{
    float tmp = *x;
    *x = *y;
    *y = tmp;
}


__device__ void swap_int(int *x, int *y)
{
    int tmp = *x;
    *x = *y;
    *y = tmp;
}


__device__ void reheap(float *dist, int *idx, int k)
{
    int root = 0;
    int child = root * 2 + 1;
    while (child < k)
    {
        if(child + 1 < k && dist[child+1] > dist[child])
            child++;
        if(dist[root] > dist[child])
            return;
        swap_float(&dist[root], &dist[child]);
        swap_int(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}


__device__ void heap_sort(float *dist, int *idx, int k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        swap_float(&dist[0], &dist[i]);
        swap_int(&idx[0], &idx[i]);
        reheap(dist, idx, i);
    }
}


// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample)
__global__ void knn_kernel(int b, int n, int m, int nsample, const float *__restrict__ xyz, const float *__restrict__ new_xyz, int *__restrict__ idx, float *__restrict__ dist2) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;
    dist2 += bs_idx * m * nsample + pt_idx * nsample;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    float best_dist[100];
    int best_idx[100];
    for(int i = 0; i < nsample; i++){
        best_dist[i] = 1e10;
        best_idx[i] = 0;
    }
    for(int i = 0; i < n; i++){
        float x = xyz[i * 3 + 0];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < best_dist[0]){
            best_dist[0] = d2;
            best_idx[0] = i;
            reheap(best_dist, best_idx, nsample);
        }
    }
    heap_sort(best_dist, best_idx, nsample);
    for(int i = 0; i < nsample; i++){
        idx[i] = best_idx[i];
        dist2[i] = best_dist[i];
    }
}


void knn_kernel_launcher(int b, int n, int m, int nsample, const float *xyz, const float *new_xyz, int *idx, float *dist2, musaStream_t stream) {
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param idx: (B, m, nsample)

    musaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knn_kernel<<<blocks, threads, 0, stream>>>(b, n, m, nsample, xyz, new_xyz, idx, dist2);
    // musaDeviceSynchronize();  // for using printf in kernel function

    err = musaGetLastError();
    if (musaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", musaGetErrorString(err));
        exit(-1);
    }
}
