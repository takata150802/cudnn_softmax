#include <iostream>
#include <vector>
#include <cudnn.h>

#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    std::cout << "CHECK cudaError_t: ";                              \
    if (error != cudaSuccess)                                        \
    {                                                                \
        std::cout << __FILE__                                        \
                  << "("                                             \
                  << __LINE__                                        \
                  << ")"                                             \
                  << ": "                                            \
                  << "Error"                                         \
                  << std::endl;                                      \
        std::cout << "code: "                                        \
                  << error                                           \
                  << ", "                                            \
                  << "reason: "                                      \
                  << cudaGetErrorString(error)                       \
                  << std::endl;                                      \
        exit(1);                                                     \
    }                                                                \
    else                                                             \
    {                                                                \
        std::cout << __FILE__                                        \
                  << "("                                             \
                  << __LINE__                                        \
                  << ")"                                             \
                  << ": "                                            \
                  << "cudaSuccess"                                   \
                  << std::endl;                                      \
    }                                                                \
}

int main(int argc, char *argv[]) {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int n = 5, c = 4, h = 1, w =1;

    std::vector<float> h_y(n * c * h* w, 0);
    std::vector<float> h_dy(n * c * h* w, 1);
    std::vector<float> h_dx(n * c * h* w, -1);

    cudnnTensorDescriptor_t yDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc,
                              CUDNN_TENSOR_NCHW,
                              CUDNN_DATA_FLOAT,
                              n,c,h,w);
    
    cudnnTensorDescriptor_t dyDesc;
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnSetTensor4dDescriptor(dyDesc,
                              CUDNN_TENSOR_NCHW,
                              CUDNN_DATA_FLOAT,
                              n,c,h,w);
    
    cudnnTensorDescriptor_t dxDesc;
    cudnnCreateTensorDescriptor(&dxDesc);
    cudnnSetTensor4dDescriptor(dxDesc,
                              CUDNN_TENSOR_NCHW,
                              CUDNN_DATA_FLOAT,
                              n,c,h,w);
    
    float *y, *dy, *dx;
    size_t size_ = n * c * h * w * sizeof(float);
    cudaMalloc (&y, size_);
    cudaMalloc (&dy, size_);
    cudaMalloc (&dx, size_);
    cudaMemset (dx, 0xff, size_);

    cudaMemcpy(y, h_y.data(), size_, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, h_dy.data(), size_, cudaMemcpyHostToDevice);

    const float alpha = 1, beta = 0;
    cudnnSoftmaxBackward(
            handle,
            CUDNN_SOFTMAX_FAST,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            yDesc,
            y,
            dyDesc,
            dy,
            &beta,
            dxDesc,
            dx);

    cudaMemcpy(h_dx.data(), dx, size_, cudaMemcpyDeviceToHost);

    for (std::vector<float>::const_iterator i = h_dx.begin(); i != h_dx.end(); ++i)
        std::cout << *i << ' ';
    std::cout << std::endl;

    cudaFree(y);
    cudaFree(dy);
    cudaFree(dx);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyTensorDescriptor(dxDesc);
    cudnnDestroy(handle);
    CHECK(cudaDeviceSynchronize());
    return 0;
}
