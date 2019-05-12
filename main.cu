#include <iostream>

#include <cudnn.h>


int main(int argc, char *argv[]) {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int n = 5, c = 4, h = 1, w =1;

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
    cudaMemset (dx, 0, size_);


    const float alpha = 1, beta = 0;
    cudnnSoftmaxBackward(
            handle,
            CUDNN_SOFTMAX_FAST,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            yDesc,
            &y,
            dyDesc,
            &dy,
            &beta,
            dxDesc,
            &dx);

    cudaFree(y);
    cudaFree(dy);
    cudaFree(dx);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyTensorDescriptor(dxDesc);
    cudnnDestroy(handle);
    std::cout << "done" << std::endl;
    return 0;
}
