#include <cuda_runtime.h>
#include <iostream>

__global__ void addOne(float *d_data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] += 1.0f;
    }
}

int main() {
    size_t size = 1 << 28;  // 1GB of f32s, 1GB / 4 = 268,435,456 elements
    float *h_data = (float*)malloc(size * sizeof(float));
    float *d_data;

    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addOne<<<numBlocks, blockSize>>>(d_data, size);

    cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    free(h_data);

    std::cout << "Done" << std::endl;
    return 0;
}
