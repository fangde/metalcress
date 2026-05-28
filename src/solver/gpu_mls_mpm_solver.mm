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

#include "gpu_mls_mpm_solver.cuh"
#include "metal_kernels.h"
#include "check_cuda.cuh"

namespace crmpm
{
    GpuMlsMpmSolver::GpuMlsMpmSolver(int numParticles, float cellSize, Bounds3 gridBound)
    {
        mNumMaxParticles = numParticles;
        mGridBound = gridBound;
        mCellSize = cellSize;
        mInvCellSize = 1 / cellSize;
        Vec3f gridSize = mGridBound.maximum - mGridBound.minimum;
        mNumNodesPerDim = (gridSize / cellSize + 1.0f).cast<int>();
        mNumNodes = mNumNodesPerDim.x() * mNumNodesPerDim.y() * mNumNodesPerDim.z();
        mCellVolume = cellSize * cellSize * cellSize;
    }

    void GpuMlsMpmSolver::initialize()
    {
        dmParticlePositionMass = mParticleData->positionMass;
        dmParticleVelocity = mParticleData->velocity;
        dmParticleMaterialProperties0 = mParticleMaterialData->params0;
        dmParticleMaterialTypes = mParticleMaterialData->type;

        // GPU particle data
        CR_CHECK_CUDA(cudaMallocAsync<float>(&dmParticleInitialVolume, mNumMaxParticles * sizeof(float), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleGradientDeformationTensorColumn0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleGradientDeformationTensorColumn1, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleGradientDeformationTensorColumn2, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleAffineMomentumColumn0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleAffineMomentumColumn1, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleAffineMomentumColumn2, mNumMaxParticles * sizeof(float4), mCudaStream));

        // GPU node data
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmNodeMomentumVelocityMass, mNumNodes * sizeof(float4), mCudaStream));

        // Reset all GPU data to zero
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleInitialVolume, 0, mNumMaxParticles * sizeof(float), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleGradientDeformationTensorColumn0, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleGradientDeformationTensorColumn1, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleGradientDeformationTensorColumn2, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleAffineMomentumColumn0, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleAffineMomentumColumn1, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleAffineMomentumColumn2, 0, mNumMaxParticles * sizeof(float4), mCudaStream));

        CR_CHECK_CUDA(cudaMemsetAsync(dmNodeMomentumVelocityMass, 0, mNumNodes * sizeof(float4), mCudaStream));
    }

    void GpuMlsMpmSolver::computeInitialData(unsigned int numParticlesToCompute,
                                             const unsigned int *CR_RESTRICT indices)
    {
        metal::dispatchMlsMpmComputeInitialGridMass(
            mCudaStream, numParticlesToCompute, indices, mGridBound.minimum, mInvCellSize, mNumNodesPerDim,
            mCellVolume, dmParticlePositionMass, dmParticleGradientDeformationTensorColumn0,
            dmParticleGradientDeformationTensorColumn1, dmParticleGradientDeformationTensorColumn2,
            dmNodeMomentumVelocityMass
        );
        
        metal::dispatchMlsMpmComputeInitialVolume(
            mCudaStream, numParticlesToCompute, indices, mGridBound.minimum, mInvCellSize, mNumNodesPerDim,
            mCellVolume, dmParticlePositionMass, dmNodeMomentumVelocityMass,
            dmParticleGradientDeformationTensorColumn0, dmParticleGradientDeformationTensorColumn1,
            dmParticleGradientDeformationTensorColumn2, dmParticleInitialVolume
        );
    }

    float GpuMlsMpmSolver::step()
    {
        resetGrid();
        particleToGrid();
        updateGrid();
        gridToParticle();
        return mIntegrationStepSize;
    }

    void GpuMlsMpmSolver::_release()
    {
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleInitialVolume, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleGradientDeformationTensorColumn0, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleGradientDeformationTensorColumn1, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleGradientDeformationTensorColumn2, mCudaStream));

        CR_CHECK_CUDA(cudaFreeAsync(dmParticleAffineMomentumColumn0, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleAffineMomentumColumn1, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleAffineMomentumColumn2, mCudaStream));

        CR_CHECK_CUDA(cudaFreeAsync(dmNodeMomentumVelocityMass, mCudaStream));
    }

    void GpuMlsMpmSolver::resetGrid()
    {
        CR_CHECK_CUDA(cudaMemsetAsync(dmNodeMomentumVelocityMass, 0, mNumNodes * sizeof(float4), mCudaStream));
    }

    void GpuMlsMpmSolver::particleToGrid()
    {
        metal::dispatchMlsMpmParticleToGrid(
            mCudaStream, mIntegrationStepSize, mGravity, mGridBound.minimum, mInvCellSize, mCellSize,
            mNumNodesPerDim, mNumActiveParticles, mActiveMask, dmParticlePositionMass, dmParticleInitialVolume,
            dmParticleGradientDeformationTensorColumn0, dmParticleGradientDeformationTensorColumn1,
            dmParticleGradientDeformationTensorColumn2, dmParticleAffineMomentumColumn0,
            dmParticleAffineMomentumColumn1, dmParticleAffineMomentumColumn2, dmParticleMaterialProperties0,
            dmParticleMaterialTypes, mNumShapes, mShapeIds, *mShapeData, *mGeometryData, *mGeometrySdfData,
            dmParticleVelocity, dmNodeMomentumVelocityMass
        );
    }

    void GpuMlsMpmSolver::updateGrid()
    {
        bool useEffectiveMass = (mShapeContactModel != ShapeContactModel::eKinematic);
        metal::dispatchMlsMpmUpdateGrid(
            mCudaStream, useEffectiveMass, mIntegrationStepSize, mNumNodes, mGridBound.minimum, mCellSize,
            mNumNodesPerDim, mNumShapes, mShapeIds, *mShapeData, *mGeometryData, *mGeometrySdfData,
            dmNodeMomentumVelocityMass
        );
    }

    void GpuMlsMpmSolver::gridToParticle()
    {
        metal::dispatchMlsMpmGridToParticle(
            mCudaStream, mGridBound.minimum, mGridBound.maximum, mInvCellSize, mCellSize, mNumNodesPerDim,
            mIntegrationStepSize, mNumActiveParticles, mActiveMask, dmNodeMomentumVelocityMass, mNumShapes,
            mShapeIds, *mShapeData, *mGeometryData, *mGeometrySdfData, dmParticlePositionMass, dmParticleVelocity,
            dmParticleGradientDeformationTensorColumn0, dmParticleGradientDeformationTensorColumn1,
            dmParticleGradientDeformationTensorColumn2, dmParticleAffineMomentumColumn0,
            dmParticleAffineMomentumColumn1, dmParticleAffineMomentumColumn2
        );
    }
}

#endif // CR_USE_METAL
