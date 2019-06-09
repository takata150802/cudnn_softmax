#include <iostream>
#include <vector>
#include <random>
#include <limits>
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

void rand_vector_float (std::vector<float> &v);
void pseudoSoftmaxBackward(const std::vector<float> &y, 
                           const std::vector<float> &dy,
                           std::vector<float> &dx,
                           const int N, const int  C, const int H, const int W);

int main(int argc, char *argv[]) {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int n = 5, c = 4, h = 1, w =1;

    std::vector<float> h_y(n * c * h* w, 0);
    std::vector<float> h_dy(n * c * h* w, 0);
    std::vector<float> h_dx(n * c * h* w, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> h_dx_expct(n * c * h* w, std::numeric_limits<float>::quiet_NaN());
    rand_vector_float(h_y);
    rand_vector_float(h_dy);

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
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            yDesc,
            y,
            dyDesc,
            dy,
            &beta,
            dxDesc,
            dx);

    pseudoSoftmaxBackward(h_y, h_dy, h_dx_expct, n, c, h, w);

    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(h_dx.data(), dx, size_, cudaMemcpyDeviceToHost);

    for (std::vector<float>::const_iterator i = h_dx.begin(); i != h_dx.end(); ++i)
        std::cout << *i << ' ';
    std::cout << std::endl;

    for (std::vector<float>::const_iterator i = h_dx_expct.begin(); i != h_dx_expct.end(); ++i)
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

namespace {
    int getIndex(const int n,
                 const int c,
                 const int h,
                 const int w,
                 const int N,
                 const int C,
                 const int H,
                 const int W
                 ) {
        return n * C * H * W
                 + c * H * W
                     + h * W
                         + w;
    }
}

void pseudoSoftmaxBackward(const std::vector<float> &y, 
                           const std::vector<float> &dy,
                           std::vector<float> &dx,
                           const int N, const int  C, const int H, const int W) {
    int idx, idx_s;
    std::vector<float> sum_(N * 1 * H * W, 0);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    idx = getIndex(n, c, h, w, N, C, H, W);
                    idx_s = getIndex(n, 1, h, w, N, 1, H, W);
                    sum_[idx_s] += y[idx] * dy[idx];
                }
            }
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    idx = getIndex(n, c, h, w, N, C, H, W);
                    idx_s = getIndex(n, 1, h, w, N, 1, H, W);
                    dx[idx] = y[idx] * (dy[idx] - sum_[idx_s]);
                }
            }
        }
    }
    return;
}

std::mt19937 mt(0);
void rand_vector_float (std::vector<float> &v) {
    std::normal_distribution<> rand(0, 5);
    for (std::vector<float>::iterator i = v.begin(); i != v.end(); ++i) {
        *i = rand(mt);
    }
    return;
}
