#include <cudnn.h>
#include <iostream>
#include "src/helper.h"

int main()
{
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t bias_desc;

    cudnnConvolutionFwdAlgo_t falgo;
    cudnnConvolutionBwdFilterAlgo_t b_falgo;
    cudnnConvolutionBwdDataAlgo_t b_dalgo;

    float *d_input = nullptr;
    float *d_output = nullptr;
    float *d_filter = nullptr;
    float *d_bias = nullptr;

    int input_n = 64;
    int input_c = 1;
    int input_h = 28;
    int input_w = 28;

    // output size
    int output_n = input_n;
    int output_c = 20;
    int output_h = 1;
    int output_w = 1;

    // kernel size
    int filter_h = 5;
    int filter_w = 5;

    // alpha, beta
    float one = 1.f;
    float zero = 0.f;

    std::cout << "[" <<  __LINE__ << "]" << std::endl;

    cudnnCreate(&cudnn);

    std::cout << "[" <<  __LINE__ << "]" << std::endl;

    /* Create Resources */
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnCreateTensorDescriptor(&bias_desc);

    std::cout << "[" <<  __LINE__ << "]" << std::endl;

    // Initilziae resources
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, output_c, input_c, filter_h, filter_w);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    0, 0,
                                    1, 1,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &output_n, &output_c, &output_h, &output_w);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_c, 1, 1);

    int weight_size = output_c * input_c * filter_h * filter_w;
    int bias_size = output_c;

    std::cout << "input  size: " << input_n << " " << input_c << " " << input_h << " " << input_w << std::endl;
    std::cout << "output size: " << output_n << " " << output_c << " " << output_h << " " << output_w << std::endl;

    std::cout << "[" <<  __LINE__ << "]" << std::endl;

    // convolution
    size_t workspace_size = 0;
    size_t temp_size = 0;
    float *d_workspace = nullptr;
    cudnnGetConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &falgo);
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, falgo, &temp_size);
    workspace_size = max(workspace_size, temp_size);

    // convolution (bwd - filter)
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, input_desc, output_desc, conv_desc, filter_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &b_falgo);
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, input_desc, output_desc, conv_desc, filter_desc, b_falgo, &temp_size);
    workspace_size = max(workspace_size, temp_size);

    // convolution (bwd - data)
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn, filter_desc, output_desc, conv_desc, input_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &b_dalgo);
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filter_desc, output_desc, conv_desc, input_desc, b_dalgo, &temp_size);
    workspace_size = max(workspace_size, temp_size);

    std::cout << "workspace size: " << workspace_size << std::endl;
    std::cout << "[" <<  __LINE__ << "]" << std::endl;

    // allocate memory space
    cudaMalloc((void**)&d_input,        sizeof(float) * input_n * input_c * input_h * input_w);
    cudaMalloc((void**)&d_filter,       sizeof(float) * weight_size);
    cudaMalloc((void**)&d_output,       sizeof(float) * output_n * output_c * output_h * output_w);
    cudaMalloc((void**)&d_workspace,    sizeof(float) * workspace_size);
    cudaMalloc((void**)&d_bias,         sizeof(float) * bias_size);

    std::cout << "[" <<  __LINE__ << "]" << std::endl;

    // Forward
    checkCudnnErrors(cudnnConvolutionForward(cudnn, &one, input_desc, d_input, filter_desc, d_filter, conv_desc, falgo, d_workspace, workspace_size, &zero, output_desc, d_output));
    checkCudnnErrors(cudnnAddTensor(cudnn, &one, bias_desc, d_bias, &one, output_desc, d_output));
    checkCudaErrors(cudaGetLastError());
    
    std::cout << "[" <<  __LINE__ << "]" << std::endl;

    // backward
    checkCudnnErrors(cudnnConvolutionBackwardBias(cudnn, &one, output_desc, d_output, &zero, bias_desc, d_bias));
    checkCudnnErrors(cudnnConvolutionBackwardFilter(cudnn, &one, input_desc, d_input, output_desc, d_output, conv_desc, b_falgo, d_workspace, workspace_size, &zero, filter_desc, d_filter));
    checkCudnnErrors(cudnnConvolutionBackwardData(cudnn, &one, filter_desc, d_filter, output_desc, d_output, conv_desc, b_dalgo, d_workspace, workspace_size, &zero, input_desc, d_input));
    checkCudaErrors(cudaGetLastError());
    
    std::cout << "[" <<  __LINE__ << "]" << std::endl;

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(bias_desc);

    std::cout << "[" <<  __LINE__ << "]" << std::endl;

    cudaFree(d_input);    
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_workspace);
    cudaFree(d_bias);

    cudnnDestroy(cudnn);

    std::cout << "[" <<  __LINE__ << "]" << std::endl;
}