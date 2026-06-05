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

#ifdef CR_USE_METAL

#import "metal_backend.h"
#include <map>
#include <mutex>
#include <cstring>
#include <iostream>

namespace crmpm {
    namespace metal {

        struct BufferInfo {
            id<MTLBuffer> buffer;
            uintptr_t start;
            size_t size;
        };

        static id<MTLDevice> gDevice = nil;
        static id<MTLCommandQueue> gCommandQueue = nil;
        static std::map<uintptr_t, BufferInfo> gBufferRegistry;
        static std::mutex gRegistryMutex;

        void init() {
            static std::once_flag initFlag;
            std::call_once(initFlag, []() {
                gDevice = MTLCreateSystemDefaultDevice();
                if (!gDevice) {
                    std::cerr << "[Metal] Error: Failed to create System Default Metal Device!\n";
                    return;
                }
                gCommandQueue = [gDevice newCommandQueue];
                std::cout << "[Metal] Initialized system default device: " 
                          << [[gDevice name] UTF8String] << "\n";
            });
        }

        id<MTLDevice> getDevice() {
            init();
            return gDevice;
        }

        id<MTLCommandQueue> getCommandQueue() {
            init();
            return gCommandQueue;
        }

        void registerBuffer(const void* ptr, id<MTLBuffer> buffer, size_t size) {
            std::lock_guard<std::mutex> lock(gRegistryMutex);
            uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
            gBufferRegistry[start] = {buffer, start, size};
        }

        void unregisterBuffer(const void* ptr) {
            std::lock_guard<std::mutex> lock(gRegistryMutex);
            uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
            gBufferRegistry.erase(start);
        }

        id<MTLBuffer> getBufferForPointer(const void* ptr, size_t* outOffset) {
            std::lock_guard<std::mutex> lock(gRegistryMutex);
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            
            // Search for the registered block containing this address
            for (auto const& [start, info] : gBufferRegistry) {
                if (addr >= info.start && addr < info.start + info.size) {
                    if (outOffset) {
                        *outOffset = addr - info.start;
                    }
                    return info.buffer;
                }
            }
            if (outOffset) {
                *outOffset = 0;
            }
            return nil;
        }

        bool isMetalPointer(const void* ptr) {
            std::lock_guard<std::mutex> lock(gRegistryMutex);
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            for (auto const& [start, info] : gBufferRegistry) {
                if (addr >= info.start && addr < info.start + info.size) {
                    return true;
                }
            }
            return false;
        }
    }
}

// C-style CUDA Compatibility API Implementation
extern "C" {

    cudaError_t cudaMalloc(void** devPtr, size_t size) {
        if (!devPtr) return cudaErrorUnknown;
        
        id<MTLDevice> device = crmpm::metal::getDevice();
        if (!device) return cudaErrorUnknown;
        
        // Allocate unified memory (Shared storage mode in Apple Silicon)
        // Ensure size is at least 16 bytes for alignment
        size_t allocSize = size < 16 ? 16 : size;
        id<MTLBuffer> buffer = [device newBufferWithLength:allocSize 
                                                   options:MTLResourceStorageModeShared];
        if (!buffer) {
            *devPtr = nullptr;
            return cudaErrorUnknown;
        }
        
        void* ptr = [buffer contents];
        std::memset(ptr, 0, size); // Initialize to 0 by default
        
        crmpm::metal::registerBuffer(ptr, buffer, size);
        *devPtr = ptr;
        return cudaSuccess;
    }

    cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
        // Under shared memory architecture, synchronous allocation is perfectly fine and fast
        return cudaMalloc(devPtr, size);
    }

    cudaError_t cudaFree(void* devPtr) {
        if (!devPtr) return cudaSuccess;
        
        crmpm::metal::unregisterBuffer(devPtr);
        
        // In Objective-C ARC, once the std::map reference is released, the buffer will be automatically deallocated
        return cudaSuccess;
    }

    cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
        return cudaFree(devPtr);
    }

    cudaError_t cudaMallocHost(void** ptr, size_t size) {
        // Pinned memory maps directly to unified shared memory in Metal on Mac
        return cudaMalloc(ptr, size);
    }

    cudaError_t cudaFreeHost(void* ptr) {
        return cudaFree(ptr);
    }

    cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
        if (!devPtr) return cudaErrorUnknown;
        std::memset(devPtr, value, count);
        return cudaSuccess;
    }

    cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
        if (!devPtr) return cudaErrorUnknown;
        if (stream) {
            size_t offset = 0;
            id<MTLBuffer> mtlBuf = crmpm::metal::getBufferForPointer(devPtr, &offset);
            if (mtlBuf) {
                id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
                id<MTLBlitCommandEncoder> blit = [buf blitCommandEncoder];
                [blit fillBuffer:mtlBuf range:NSMakeRange(offset, count) value:(uint8_t)value];
                [blit endEncoding];
                return cudaSuccess;
            }
        }
        std::memset(devPtr, value, count);
        return cudaSuccess;
    }

    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
        if (!dst || !src) return cudaErrorUnknown;
        std::memmove(dst, src, count);
        return cudaSuccess;
    }

    cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
        if (!dst || !src) return cudaErrorUnknown;
        if (stream) {
            size_t dstOffset = 0, srcOffset = 0;
            id<MTLBuffer> dstBuf = crmpm::metal::getBufferForPointer(dst, &dstOffset);
            id<MTLBuffer> srcBuf = crmpm::metal::getBufferForPointer(src, &srcOffset);
            if (dstBuf && srcBuf) {
                id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
                id<MTLBlitCommandEncoder> blit = [buf blitCommandEncoder];
                [blit copyFromBuffer:srcBuf sourceOffset:srcOffset toBuffer:dstBuf destinationOffset:dstOffset size:count];
                [blit endEncoding];
                return cudaSuccess;
            } else {
                cudaStreamSynchronize(stream);
            }
        }
        std::memmove(dst, src, count);
        return cudaSuccess;
    }

    cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
        if (!pStream) return cudaErrorUnknown;
        
        id<MTLCommandQueue> queue = crmpm::metal::getCommandQueue();
        if (!queue) return cudaErrorUnknown;
        
        cudaStream_t stream = new CUstream_st();
        stream->commandQueue = queue;
        stream->currentCommandBuffer = [queue commandBuffer];
        
        *pStream = stream;
        return cudaSuccess;
    }

    cudaError_t cudaStreamDestroy(cudaStream_t stream) {
        if (!stream) return cudaSuccess;
        
        delete stream;
        return cudaSuccess;
    }

    cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
        if (!stream) return cudaSuccess;
        
        id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
        if (buf) {
            [buf commit];
            [buf waitUntilCompleted];
            
            // Start a new command buffer for the next asynchronous sequence
            id<MTLCommandQueue> queue = stream->commandQueue;
            stream->currentCommandBuffer = [queue commandBuffer];
        }
        
        return cudaSuccess;
    }

    cudaError_t cudaDeviceSynchronize() {
        id<MTLCommandQueue> queue = crmpm::metal::getCommandQueue();
        if (queue) {
            id<MTLCommandBuffer> buf = [queue commandBuffer];
            [buf commit];
            [buf waitUntilCompleted];
        }
        return cudaSuccess;
    }

    const char* cudaGetErrorString(cudaError_t error) {
        if (error == cudaSuccess) {
            return "cudaSuccess";
        }
        return "cudaErrorUnknown";
    }
}

#endif // CR_USE_METAL
