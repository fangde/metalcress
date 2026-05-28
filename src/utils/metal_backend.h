/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2025, Yafei Ou and Mahdi Tavakoli
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CR_METAL_BACKEND_H
#define CR_METAL_BACKEND_H

#ifdef CR_USE_METAL

#include <cstdint>
#include <cstddef>
#include <cstdio>

// Define Objective-C type wrappers for standard C++ headers
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
typedef id<MTLCommandQueue> MetalCommandQueueType;
typedef id<MTLCommandBuffer> MetalCommandBufferType;
typedef id<MTLBuffer> MetalBufferType;
typedef id<MTLDevice> MetalDeviceType;
typedef id<MTLComputePipelineState> MetalPipelineState;
#else
typedef void* MetalCommandQueueType;
typedef void* MetalCommandBufferType;
typedef void* MetalBufferType;
typedef void* MetalDeviceType;
typedef void* MetalPipelineState;
#endif

// cudaStream_t wrapper
struct CUstream_st {
    MetalCommandQueueType commandQueue;
    MetalCommandBufferType currentCommandBuffer;
};
typedef struct CUstream_st* cudaStream_t;

// Compatibility enums and types
typedef int cudaError_t;
const cudaError_t cudaSuccess = 0;
const cudaError_t cudaErrorUnknown = -1;

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 0,
    cudaMemcpyDeviceToHost = 1,
    cudaMemcpyDeviceToDevice = 2,
    cudaMemcpyDefault = 3
};

// C++ API for Metal memory registry and shader pipeline cache
namespace crmpm {
    namespace metal {
        void init();
        MetalDeviceType getDevice();
        MetalCommandQueueType getCommandQueue();
        
        // Registration for shared/unified memory mapping
        void registerBuffer(const void* ptr, MetalBufferType buffer, size_t size);
        void unregisterBuffer(const void* ptr);
        
        // Find underlying MTLBuffer and compute byte offset for nested/suballocated pointers
        MetalBufferType getBufferForPointer(const void* ptr, size_t* outOffset);
        
        bool isMetalPointer(const void* ptr);
        
        // Pipeline state compiler
        MetalPipelineState getPipelineState(const char* kernelName);
    }
}

// Map standard CUDA runtime API calls to our Metal backend implementation
extern "C" {
    cudaError_t cudaMalloc(void** devPtr, size_t size);
    cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream);
    cudaError_t cudaFree(void* devPtr);
    cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream);
    
    cudaError_t cudaMallocHost(void** ptr, size_t size);
    cudaError_t cudaFreeHost(void* ptr);
    
    cudaError_t cudaMemset(void* devPtr, int value, size_t count);
    cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);
    
    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
    cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
    
    cudaError_t cudaStreamCreate(cudaStream_t* pStream);
    cudaError_t cudaStreamDestroy(cudaStream_t stream);
    cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    cudaError_t cudaDeviceSynchronize();
    
    const char* cudaGetErrorString(cudaError_t error);
}

#ifdef __cplusplus
template <typename T>
inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
    return ::cudaMalloc(reinterpret_cast<void**>(devPtr), size);
}

template <typename T>
inline cudaError_t cudaMallocHost(T** ptr, size_t size) {
    return ::cudaMallocHost(reinterpret_cast<void**>(ptr), size);
}

template <typename T>
inline cudaError_t cudaMallocAsync(T** devPtr, size_t size, cudaStream_t stream) {
    return ::cudaMallocAsync(reinterpret_cast<void**>(devPtr), size, stream);
}
#endif

#endif // CR_USE_METAL

#endif // CR_METAL_BACKEND_H
