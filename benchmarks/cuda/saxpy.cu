#include <stdio.h>

__global__ void comprehension(float *a, float *b, float *result, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int id = index; id < size; id += stride)
    {
        if (id < size)
        {
            result[id] = (2 * a[id]) + b[id];
        }
    }
}

int main(int argc, char const *argv[])
{
    int size = atoi(argv[1]);
    //int bytes = size * sizeof(float);

    float *host_a, *host_b, *host_result;
    host_a = (float *)malloc(size*sizeof(float));
    host_b = (float *)malloc(size*sizeof(float));
    host_result = (float *)malloc(size*sizeof(float));
    // cudaMallocHost((void **)&host_a, size * sizeof(float));
    // cudaMallocHost((void **)&host_b, size * sizeof(float));
    // cudaMallocHost((void **)&host_result, size * sizeof(float));
    if (host_a == NULL || host_b == NULL || host_result == NULL) {
        fprintf(stderr, "malloc failed for size %d\n", size);
        return EXIT_FAILURE;
    }

    // Filling a and b arrays
    for (int i = 0; i < size; i++)
    {
        host_a[i] = i + 1;
        host_b[i] = i + 1;
    }

    float *dev_a, *dev_b, *dev_result;
    cudaError_t err;

    int threadsPerBlock = 256;
    int numberOfBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    float time_h2d, time_kernel, time_d2h;
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&dev_a, size * sizeof(float));
    cudaMalloc((void **)&dev_b, size * sizeof(float));
    cudaMalloc((void **)&dev_result, size * sizeof(float));

    cudaEventRecord(start2, 0);
    cudaMemcpy(dev_a, host_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&time_h2d, start2, stop2);

    cudaEventRecord(start2, 0);
    comprehension<<<numberOfBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_result, size);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&time_kernel, start2, stop2);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(start2, 0);
    volatile float tmp = host_result[0];
    tmp = host_result[size-1]; 
    cudaMemcpy(host_result, dev_result, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&time_d2h, start2, stop2);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("H2D: %.1fms, Kernel: %.1fms, D2H: %.1fms\n", time_h2d, time_kernel, time_d2h);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    printf("CUDA\t%d\t%3.1f\n", size, time);

    free(host_a);
    free(host_b);
    free(host_result);
    // cudaFreeHost(host_a);
    // cudaFreeHost(host_b);
    // cudaFreeHost(host_result);

    return 0;
}
