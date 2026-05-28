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

#ifndef CR_METAL_KERNELS_H
#define CR_METAL_KERNELS_H

#ifdef CR_USE_METAL
 
#include "metal_backend.h"
#include "material_data.h"
#include "bounds3.h"
#include "shape.h"
#include "geometry.h"

namespace crmpm {
    namespace metal {

        // Compile all Metal shaders at startup
        void compileShaders();

        // -------------------
        // Standard MPM Kernels
        // -------------------
        void dispatchStandardMpmComputeInitialGridMass(
            cudaStream_t stream, int numParticles, Vec3f gridBoundMin, float invCellSize,
            Vec3i numNodesPerDim, float gridVolume, const float4* particlePositionMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float4* nodeMomentumVelocityMass
        );

        void dispatchStandardMpmComputeInitialVolume(
            cudaStream_t stream, int numParticles, Vec3f gridBoundMin, float invCellSize,
            Vec3i numNodesPerDim, float gridVolume, const float4* particlePositionMass,
            const float4* nodeMomentumVelocityMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float* particleInitialVolume
        );

        void dispatchStandardMpmParticleToGrid(
            cudaStream_t stream, int numParticles, Vec3f gravity, Vec3f gridBoundMin,
            float invCellSize, Vec3i numNodesPerDim, const float4* particlePositionMass,
            const Vec3f* particleVelocity, const float* particleInitialVolume,
            const float4* particleGradientDeformationTensorColumn0,
            const float4* particleGradientDeformationTensorColumn1,
            const float4* particleGradientDeformationTensorColumn2,
            const float4* particleMaterialProperties0,
            const ParticleMaterialType* particleMaterialTypes,
            float4* nodeMomentumVelocityMass, Vec3f* nodeForce
        );

        void dispatchStandardMpmUpdateGrid(
            cudaStream_t stream, bool useEffectiveMass, int numNodes, Vec3f gridBoundMin,
            float cellSize, Vec3i numNodesPerDim, float integrationStepSize,
            const Vec3f* nodeForce, int numShapes, const int* shapeIds,
            const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData, float4* nodeMomentumVelocityMass
        );

        void dispatchStandardMpmGridToParticle(
            cudaStream_t stream, int numParticles, Vec3f gridBoundMin, Vec3f gridBoundMax,
            float invCellSize, Vec3i numNodesPerDim, float integrationStepSize,
            const float4* nodeMomentumVelocityMass, int numShapes, const int* shapeIds,
            const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData, float4* particlePositionMass,
            Vec3f* particleVelocity,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2
        );

        // -------------------
        // MLS-MPM Kernels
        // -------------------
        void dispatchMlsMpmComputeInitialGridMass(
            cudaStream_t stream, int numParticlesToCompute, const unsigned int* indices,
            Vec3f gridBoundMin, float invCellSize, Vec3i numNodesPerDim, float cellVolume,
            const float4* particlePositionMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float4* nodeMomentumVelocityMass
        );

        void dispatchMlsMpmComputeInitialVolume(
            cudaStream_t stream, int numParticlesToCompute, const unsigned int* indices,
            Vec3f gridBoundMin, float invCellSize, Vec3i numNodesPerDim, float cellVolume,
            const float4* particlePositionMass, const float4* nodeMomentumVelocityMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float* particleInitialVolume
        );

        void dispatchMlsMpmParticleToGrid(
            cudaStream_t stream, float integrationStepSize, Vec3f gravity, Vec3f gridBoundMin,
            float invCellSize, float cellSize, Vec3i numNodesPerDim, int numActiveParticles,
            const unsigned char* activeMask, const float4* particlePositionMass,
            const float* particleInitialVolume,
            const float4* particleGradientDeformationTensorColumn0,
            const float4* particleGradientDeformationTensorColumn1,
            const float4* particleGradientDeformationTensorColumn2,
            const float4* particleAffineMomentumColumn0,
            const float4* particleAffineMomentumColumn1,
            const float4* particleAffineMomentumColumn2,
            const float4* particleMaterialProperties0,
            const ParticleMaterialType* particleMaterialTypes,
            int numShapes, const int* shapeIds,
            const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData,
            Vec3f* particleVelocity, float4* nodeMomentumVelocityMass
        );

        void dispatchMlsMpmUpdateGrid(
            cudaStream_t stream, bool useEffectiveMass, float integrationStepSize, int numNodes,
            Vec3f gridBoundMin, float cellSize, Vec3i numNodesPerDim, int numShapes,
            const int* shapeIds, const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData, float4* nodeMomentumVelocityMass
        );

        void dispatchMlsMpmGridToParticle(
            cudaStream_t stream, Vec3f gridBoundMin, Vec3f gridBoundMax, float invCellSize,
            float cellSize, Vec3i numNodesPerDim, float integrationStepSize, int numActiveParticles,
            const unsigned char* activeMask, const float4* nodeMomentumVelocityMass,
            int numShapes, const int* shapeIds, const ShapeData& shapeData,
            const GeometryData& geometryData, const GeometrySdfData& geometrySdfData,
            float4* particlePositionMass, Vec3f* particleVelocity,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float4* particleAffineMomentumColumn0,
            float4* particleAffineMomentumColumn1,
            float4* particleAffineMomentumColumn2
        );

        // -------------------
        // Position-Based MPM Kernels (PB-MPM)
        // -------------------
        void dispatchPbMpmComputeInitialGridMass(
            cudaStream_t stream, int numParticlesToCompute, const unsigned int* indices,
            Vec3f gridBoundMin, float invCellSize, Vec3i numNodesPerDim, float cellVolume,
            const float4* particlePositionMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float4* nodeMomentumVelocityMass
        );

        void dispatchPbMpmComputeInitialVolume(
            cudaStream_t stream, int numParticlesToCompute, const unsigned int* indices,
            Vec3f gridBoundMin, float invCellSize, Vec3i numNodesPerDim, float cellVolume,
            const float4* particlePositionMass, const float4* nodeMomentumVelocityMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float* particleInitialVolume
        );

        void dispatchPbMpmParticleToGrid(
            cudaStream_t stream, float integrationStepSize, Vec3f gridBoundMin, float invCellSize,
            float cellSize, Vec3i numNodesPerDim, int numActiveParticles, const unsigned char* activeMask,
            const float4* particlePositionMass,
            const float4* particleGradientDeformationTensorColumn0,
            const float4* particleGradientDeformationTensorColumn1,
            const float4* particleGradientDeformationTensorColumn2,
            const float4* particleAffineMomentumColumn0,
            const float4* particleAffineMomentumColumn1,
            const float4* particleAffineMomentumColumn2,
            const float4* particleMaterialProperties0,
            const ParticleMaterialType* particleMaterialTypes,
            int numShapes, const int* shapeIds,
            const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData,
            Vec3f* particleVelocity, float4* nodeMomentumVelocityMass
        );

        void dispatchPbMpmUpdateGrid(
            cudaStream_t stream, bool useEffectiveMass, float integrationStepSize, int numNodes,
            Vec3f gridBoundMin, float cellSize, Vec3i numNodesPerDim, int numShapes,
            const int* shapeIds, const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData, float4* nodeMomentumVelocityMass
        );

        void dispatchPbMpmGridToParticle(
            cudaStream_t stream, Vec3f gridBoundMin, Vec3f gridBoundMax, float invCellSize,
            float cellSize, float cellVolume, Vec3i numNodesPerDim, float integrationStepSize,
            int numActiveParticles, const unsigned char* activeMask,
            const float4* nodeMomentumVelocityMass, int numShapes, const int* shapeIds,
            const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData, float4* particlePositionMass,
            Vec3f* particleVelocity,
            float4* particleAffineMomentumColumn0,
            float4* particleAffineMomentumColumn1,
            float4* particleAffineMomentumColumn2
        );

        void dispatchPbMpmIntegrateParticle(
            cudaStream_t stream, Vec3f gridBoundMin, Vec3f gridBoundMax, float integrationStepSize,
            Vec3f gravity, int numActiveParticles, const unsigned char* activeMask,
            const float4* particleAffineMomentumColumn0,
            const float4* particleAffineMomentumColumn1,
            const float4* particleAffineMomentumColumn2,
            const ParticleMaterialType* particleMaterialTypes,
            int numShapes, const int* shapeIds, const ShapeData& shapeData,
            const GeometryData& geometryData, const GeometrySdfData& geometrySdfData,
            float4* particlePositionMass, Vec3f* particleVelocity,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2
        );

    }
}

#endif // CR_USE_METAL

#endif // CR_METAL_KERNELS_H
