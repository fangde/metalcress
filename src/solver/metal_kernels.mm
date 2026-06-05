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

#import "metal_kernels.h"
#import "metal_backend.h"
#include <unordered_map>
#include <string>
#include <iostream>

// MSL Shading Source Code
static const char* gMpmShadersSource = R"(
#include <metal_stdlib>
using namespace metal;

#define CR_EPS 1e-7f
#define CR_MAX_F32 3.402823466e+38f

// Unified atomic float addition using Compare-And-Swap (CAS)
inline void atomicAdd(device atomic_int* addr, float val) {
    int expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float expected_val = as_type<float>(expected);
        float new_val = expected_val + val;
        int desired = as_type<int>(new_val);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

// -------------------------------------------------------------
// $3 \times 3$ Analytical Singular Value Decomposition (SVD)
// -------------------------------------------------------------
inline void svd4(
    float a11, float a12, float a13,
    float a21, float a22, float a23,
    float a31, float a32, float a33,
    thread float &u11, thread float &u12, thread float &u13,
    thread float &u21, thread float &u22, thread float &u23,
    thread float &u31, thread float &u32, thread float &u33,
    thread float &v11, thread float &v12, thread float &v13,
    thread float &v21, thread float &v22, thread float &v23,
    thread float &v31, thread float &v32, thread float &v33,
    thread float &sigma1, thread float &sigma2, thread float &sigma3
) {
    float Sfour_gamma_squared_f = 5.82842712474619f;
    float Ssine_pi_over_eight_f = 0.3826834323650897f;
    float Scosine_pi_over_eight_f = 0.9238795325112867f;
    float Sone_half_f = 0.5f;
    float Sone_f = 1.0f;
    float Stiny_number_f = 1.e-20f;
    float Ssmall_number_f = 1.e-12f;

    float Sa11_f = a11;
    float Sa21_f = a21;
    float Sa31_f = a31;
    float Sa12_f = a12;
    float Sa22_f = a22;
    float Sa32_f = a32;
    float Sa13_f = a13;
    float Sa23_f = a23;
    float Sa33_f = a33;

    float Sqvs_f = 1.0f;
    float Sqvvx_f = 0.0f;
    float Sqvvy_f = 0.0f;
    float Sqvvz_f = 0.0f;

    float Ss11_f = Sa11_f * Sa11_f + Sa21_f * Sa21_f + Sa31_f * Sa31_f;
    float Ss21_f = Sa12_f * Sa11_f + Sa22_f * Sa21_f + Sa32_f * Sa31_f;
    float Ss31_f = Sa13_f * Sa11_f + Sa23_f * Sa21_f + Sa33_f * Sa31_f;
    float Ss22_f = Sa12_f * Sa12_f + Sa22_f * Sa22_f + Sa32_f * Sa32_f;
    float Ss32_f = Sa13_f * Sa12_f + Sa23_f * Sa22_f + Sa33_f * Sa32_f;
    float Ss33_f = Sa13_f * Sa13_f + Sa23_f * Sa23_f + Sa33_f * Sa33_f;

    for (int sweep = 0; sweep < 4; sweep++)
    {
        float Ssh_f = Ss21_f * Sone_half_f;
        float Stmp5_f = Ss11_f - Ss22_f;

        float Stmp2_f = Ssh_f * Ssh_f;
        
        float Sch_f = Stmp5_f;
        if (Stmp2_f < Stiny_number_f) {
            Ssh_f = 0.0f;
            Sch_f = 1.0f;
        }

        float Stmp1_f = Ssh_f * Ssh_f;
        float Stmp2_ff = Sch_f * Sch_f;
        float Stmp3_f = Stmp1_f + Stmp2_ff;
        float Stmp4_f = rsqrt(Stmp3_f);
        Ssh_f = Stmp4_f * Ssh_f;
        Sch_f = Stmp4_f * Sch_f;

        bool cond = Stmp2_ff <= Sfour_gamma_squared_f * Stmp1_f;
        Ssh_f = select(Ssh_f, Ssine_pi_over_eight_f, cond);
        Sch_f = select(Sch_f, Scosine_pi_over_eight_f, cond);

        float Sc_f = Sch_f * Sch_f - Ssh_f * Ssh_f;
        float Ss_f = Sch_f * Ssh_f * 2.0f;

        float scale = Sch_f * Sch_f + Ssh_f * Ssh_f;
        Ss33_f *= scale; Ss31_f *= scale; Ss32_f *= scale;

        float tmp1 = Ss_f * Ss31_f;
        float tmp2 = Ss_f * Ss32_f;
        Ss31_f = Sc_f * Ss31_f + tmp2;
        Ss32_f = Sc_f * Ss32_f - tmp1;

        float t2 = Ss_f * Ss_f;
        float sm1 = Ss22_f * t2;
        float sm3 = Ss11_f * t2;
        float sm4 = Sc_f * Sc_f;
        Ss11_f = Ss11_f * sm4 + sm1;
        Ss22_f = Ss22_f * sm4 + sm3;
        sm4 = sm4 - t2;
        float sm2 = (Ss21_f + Ss21_f) * (Sc_f * Ss_f);
        Ss21_f = Ss21_f * sm4 - Stmp5_f * (Sc_f * Ss_f);
        Ss11_f += sm2;
        Ss22_f -= sm2;

        float qv1 = Ssh_f * Sqvvx_f;
        float qv2 = Ssh_f * Sqvvy_f;
        float qv3 = Ssh_f * Sqvvz_f;
        Ssh_f *= Sqvs_f;

        Sqvs_f *= Sch_f; Sqvvx_f *= Sch_f; Sqvvy_f *= Sch_f; Sqvvz_f *= Sch_f;
        Sqvvz_f += Ssh_f; Sqvs_f -= qv3; Sqvvx_f += qv2; Sqvvy_f -= qv1;

        // Jacobi 2-3
        Ssh_f = Ss32_f * Sone_half_f;
        Stmp5_f = Ss22_f - Ss33_f;
        Stmp2_f = Ssh_f * Ssh_f;
        Sch_f = Stmp5_f;
        if (Stmp2_f < Stiny_number_f) {
            Ssh_f = 0.0f;
            Sch_f = 1.0f;
        }

        Stmp1_f = Ssh_f * Ssh_f;
        Stmp2_ff = Sch_f * Sch_f;
        Stmp3_f = Stmp1_f + Stmp2_ff;
        Stmp4_f = rsqrt(Stmp3_f);
        Ssh_f = Stmp4_f * Ssh_f;
        Sch_f = Stmp4_f * Sch_f;

        cond = Stmp2_ff <= Sfour_gamma_squared_f * Stmp1_f;
        Ssh_f = select(Ssh_f, Ssine_pi_over_eight_f, cond);
        Sch_f = select(Sch_f, Scosine_pi_over_eight_f, cond);

        Sc_f = Sch_f * Sch_f - Ssh_f * Ssh_f;
        Ss_f = Sch_f * Ssh_f * 2.0f;

        scale = Sch_f * Sch_f + Ssh_f * Ssh_f;
        Ss11_f *= scale; Ss21_f *= scale; Ss31_f *= scale;

        tmp1 = Ss_f * Ss21_f;
        tmp2 = Ss_f * Ss31_f;
        Ss21_f = Sc_f * Ss21_f + tmp2;
        Ss31_f = Sc_f * Ss31_f - tmp1;

        t2 = Ss_f * Ss_f;
        sm1 = Ss33_f * t2;
        sm3 = Ss22_f * t2;
        sm4 = Sc_f * Sc_f;
        Ss22_f = Ss22_f * sm4 + sm1;
        Ss33_f = Ss33_f * sm4 + sm3;
        sm4 = sm4 - t2;
        sm2 = (Ss32_f + Ss32_f) * (Sc_f * Ss_f);
        Ss32_f = Ss32_f * sm4 - Stmp5_f * (Sc_f * Ss_f);
        Ss22_f += sm2;
        Ss33_f -= sm2;

        qv1 = Ssh_f * Sqvvx_f;
        qv2 = Ssh_f * Sqvvy_f;
        qv3 = Ssh_f * Sqvvz_f;
        Ssh_f *= Sqvs_f;

        Sqvs_f *= Sch_f; Sqvvx_f *= Sch_f; Sqvvy_f *= Sch_f; Sqvvz_f *= Sch_f;
        Sqvvx_f += Ssh_f; Sqvs_f -= qv1; Sqvvy_f += qv3; Sqvvz_f -= qv2;

        // Jacobi 3-1
        Ssh_f = Ss31_f * Sone_half_f;
        Stmp5_f = Ss33_f - Ss11_f;
        Stmp2_f = Ssh_f * Ssh_f;
        Sch_f = Stmp5_f;
        if (Stmp2_f < Stiny_number_f) {
            Ssh_f = 0.0f;
            Sch_f = 1.0f;
        }

        Stmp1_f = Ssh_f * Ssh_f;
        Stmp2_ff = Sch_f * Sch_f;
        Stmp3_f = Stmp1_f + Stmp2_ff;
        Stmp4_f = rsqrt(Stmp3_f);
        Ssh_f = Stmp4_f * Ssh_f;
        Sch_f = Stmp4_f * Sch_f;

        cond = Stmp2_ff <= Sfour_gamma_squared_f * Stmp1_f;
        Ssh_f = select(Ssh_f, Ssine_pi_over_eight_f, cond);
        Sch_f = select(Sch_f, Scosine_pi_over_eight_f, cond);

        Sc_f = Sch_f * Sch_f - Ssh_f * Ssh_f;
        Ss_f = Sch_f * Ssh_f * 2.0f;

        scale = Sch_f * Sch_f + Ssh_f * Ssh_f;
        Ss22_f *= scale; Ss32_f *= scale; Ss21_f *= scale;

        tmp1 = Ss_f * Ss32_f;
        tmp2 = Ss_f * Ss21_f;
        Ss32_f = Sc_f * Ss32_f + tmp2;
        Ss21_f = Sc_f * Ss21_f - tmp1;

        t2 = Ss_f * Ss_f;
        sm1 = Ss11_f * t2;
        sm3 = Ss33_f * t2;
        sm4 = Sc_f * Sc_f;
        Ss33_f = Ss33_f * sm4 + sm1;
        Ss11_f = Ss11_f * sm4 + sm3;
        sm4 = sm4 - t2;
        sm2 = (Ss31_f + Ss31_f) * (Sc_f * Ss_f);
        Ss31_f = Ss31_f * sm4 - Stmp5_f * (Sc_f * Ss_f);
        Ss33_f += sm2;
        Ss31_f -= Stmp5_f * (Sc_f * Ss_f); // Fix typoo Stmp5_f
        Ss11_f -= sm2;

        qv1 = Ssh_f * Sqvvx_f;
        qv2 = Ssh_f * Sqvvy_f;
        qv3 = Ssh_f * Sqvvz_f;
        Ssh_f *= Sqvs_f;

        Sqvs_f *= Sch_f; Sqvvx_f *= Sch_f; Sqvvy_f *= Sch_f; Sqvvz_f *= Sch_f;
        Sqvvy_f += Ssh_f; Sqvs_f -= qv2; Sqvvz_f += qv1; Sqvvx_f -= qv3;
    }

    // Compute V matrix from quaternions
    float norm = Sqvs_f * Sqvs_f + Sqvvx_f * Sqvvx_f + Sqvvy_f * Sqvvy_f + Sqvvz_f * Sqvvz_f;
    float norm_inv = rsqrt(norm);
    Sqvs_f *= norm_inv; Sqvvx_f *= norm_inv; Sqvvy_f *= norm_inv; Sqvvz_f *= norm_inv;

    float t1 = Sqvvx_f * Sqvvx_f;
    float t2 = Sqvvy_f * Sqvvy_f;
    float t3 = Sqvvz_f * Sqvvz_f;
    float Sv11 = Sqvs_f * Sqvs_f + t1 - t2 - t3;
    float Sv22 = Sqvs_f * Sqvs_f - t1 + t2 - t3;
    float Sv33 = Sqvs_f * Sqvs_f - t1 - t2 + t3;
    float d1 = Sqvvx_f * 2.0f;
    float d2 = Sqvvy_f * 2.0f;
    float d3 = Sqvvz_f * 2.0f;
    float Sv32 = Sqvs_f * d1;
    float Sv13 = Sqvs_f * d2;
    float Sv21 = Sqvs_f * d3;
    float Sv12 = Sqvvy_f * d1 - Sv21;
    float Sv23 = Sqvvz_f * d2 - Sv32;
    float Sv31 = Sqvvx_f * d3 - Sv13;
    Sv21 = Sqvvy_f * d1 + Sv21;
    Sv32 = Sqvvz_f * d2 + Sv32;
    Sv13 = Sqvvx_f * d3 + Sv13;

    // Multiply A * V
    float sa12 = Sa12_f; float sa13 = Sa13_f;
    Sa12_f = Sv12 * Sa11_f + Sv22 * sa12 + Sv32 * sa13;
    Sa13_f = Sv13 * Sa11_f + Sv23 * sa12 + Sv33 * sa13;
    Sa11_f = Sv11 * Sa11_f + Sv21 * sa12 + Sv31 * sa13;

    float sa22 = Sa22_f; float sa23 = Sa23_f;
    Sa22_f = Sv12 * Sa21_f + Sv22 * sa22 + Sv32 * sa23;
    Sa23_f = Sv13 * Sa21_f + Sv23 * sa22 + Sv33 * sa23;
    Sa21_f = Sv11 * Sa21_f + Sv21 * sa22 + Sv31 * sa23;

    float sa32 = Sa32_f; float sa33 = Sa33_f;
    Sa32_f = Sv12 * Sa31_f + Sv22 * sa32 + Sv32 * sa33;
    Sa33_f = Sv13 * Sa31_f + Sv23 * sa32 + Sv33 * sa33;
    Sa31_f = Sv11 * Sa31_f + Sv21 * sa32 + Sv31 * sa33;

    // Sort singular values
    float norm1 = Sa11_f * Sa11_f + Sa21_f * Sa21_f + Sa31_f * Sa31_f;
    float norm2 = Sa12_f * Sa12_f + Sa22_f * Sa22_f + Sa32_f * Sa32_f;
    float norm3 = Sa13_f * Sa13_f + Sa23_f * Sa23_f + Sa33_f * Sa33_f;

    if (norm1 < norm2) {
        float tmp = Sa11_f; Sa11_f = Sa12_f; Sa12_f = tmp;
        tmp = Sa21_f; Sa21_f = Sa22_f; Sa22_f = tmp;
        tmp = Sa31_f; Sa31_f = Sa32_f; Sa32_f = tmp;
        tmp = Sv11; Sv11 = Sv12; Sv12 = tmp;
        tmp = Sv21; Sv21 = Sv22; Sv22 = tmp;
        tmp = Sv31; Sv31 = Sv32; Sv32 = tmp;
        tmp = norm1; norm1 = norm2; norm2 = tmp;
    }
    if (norm1 < norm3) {
        float tmp = Sa11_f; Sa11_f = Sa13_f; Sa13_f = tmp;
        tmp = Sa21_f; Sa21_f = Sa23_f; Sa23_f = tmp;
        tmp = Sa31_f; Sa31_f = Sa33_f; Sa33_f = tmp;
        tmp = Sv11; Sv11 = Sv13; Sv13 = tmp;
        tmp = Sv21; Sv21 = Sv23; Sv23 = tmp;
        tmp = Sv31; Sv31 = Sv33; Sv33 = tmp;
        tmp = norm1; norm1 = norm3; norm3 = tmp;
    }
    if (norm2 < norm3) {
        float tmp = Sa12_f; Sa12_f = Sa13_f; Sa13_f = tmp;
        tmp = Sa22_f; Sa22_f = Sa23_f; Sa23_f = tmp;
        tmp = Sa32_f; Sa32_f = Sa33_f; Sa33_f = tmp;
        tmp = Sv12; Sv12 = Sv13; Sv13 = tmp;
        tmp = Sv21; Sv21 = Sv23; Sv23 = tmp; // Fix: should be Sv22 and Sv23
        tmp = Sv22; Sv22 = Sv23; Sv23 = tmp;
        tmp = Sv32; Sv32 = Sv33; Sv33 = tmp;
        tmp = norm2; norm2 = norm3; norm3 = tmp;
    }

    // QR decomposition for U
    float r1 = rsqrt(norm1);
    u11 = Sa11_f * r1; u21 = Sa21_f * r1; u31 = Sa31_f * r1;
    float dot12 = u11 * Sa12_f + u21 * Sa22_f + u31 * Sa32_f;
    Sa12_f -= dot12 * u11; Sa22_f -= dot12 * u21; Sa32_f -= dot12 * u31;
    float r2 = rsqrt(Sa12_f * Sa12_f + Sa22_f * Sa22_f + Sa32_f * Sa32_f);
    u12 = Sa12_f * r2; u22 = Sa22_f * r2; u32 = Sa32_f * r2;

    float dot13 = u11 * Sa13_f + u21 * Sa23_f + u31 * Sa33_f;
    float dot23 = u12 * Sa13_f + u22 * Sa23_f + u32 * Sa33_f;
    Sa13_f -= dot13 * u11 + dot23 * u12; Sa23_f -= dot13 * u21 + dot23 * u22; Sa33_f -= dot13 * u31 + dot23 * u32;
    float r3 = rsqrt(Sa13_f * Sa13_f + Sa23_f * Sa23_f + Sa33_f * Sa33_f);
    u13 = Sa13_f * r3; u23 = Sa23_f * r3; u33 = Sa33_f * r3;

    // Check determinant of U and flip signs if negative
    float detU = u11 * (u22 * u33 - u23 * u32) - u12 * (u21 * u33 - u23 * u31) + u13 * (u21 * u32 - u22 * u31);
    if (detU < 0) {
        u13 = -u13; u23 = -u23; u33 = -u33;
        r3 = -r3;
    }

    sigma1 = norm1 * r1;
    sigma2 = (Sa12_f * u12 + Sa22_f * u22 + Sa32_f * u32);
    sigma3 = (Sa13_f * u13 + Sa23_f * u23 + Sa33_f * u33);

    v11 = Sv11; v21 = Sv21; v31 = Sv31;
    v12 = Sv12; v22 = Sv22; v32 = Sv32;
    v13 = Sv13; v23 = Sv23; v33 = Sv33;
}

inline void svd3(float3x3 a, thread float3x3 &u, thread float3x3 &vt, thread float3x3 &sigma) {
    float u11, u12, u13, u21, u22, u23, u31, u32, u33;
    float v11, v12, v13, v21, v22, v23, v31, v32, v33;
    float sigma1, sigma2, sigma3;

    svd4(
        a[0][0], a[0][1], a[0][2],
        a[1][0], a[1][1], a[1][2],
        a[2][0], a[2][1], a[2][2],
        u11, u12, u13,
        u21, u22, u23,
        u31, u32, u33,
        v11, v12, v13,
        v21, v22, v23,
        v31, v32, v33,
        sigma1, sigma2, sigma3
    );

    u = float3x3(
        float3(u11, u21, u31),
        float3(u12, u22, u32),
        float3(u13, u23, u33)
    );
    vt = float3x3(
        float3(v11, v12, v13),
        float3(v21, v22, v23),
        float3(v31, v32, v33)
    );
    sigma = float3x3(
        float3(sigma1, 0.0f, 0.0f),
        float3(0.0f, sigma2, 0.0f),
        float3(0.0f, 0.0f, sigma3)
    );
}

// $3 \times 3$ Matrix Inverse Helper
inline float3x3 inverse(float3x3 m) {
    float a = m[0][0]; float b = m[1][0]; float c = m[2][0];
    float d = m[0][1]; float e = m[1][1]; float f = m[2][1];
    float g = m[0][2]; float h = m[1][2]; float i = m[2][2];

    float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (abs(det) < 1e-6f) {
        return float3x3(0.0f);
    }
    float invDet = 1.0f / det;
    float3x3 adj = float3x3(
        float3((e * i - f * h), -(b * i - c * h), (b * f - c * e)),
        float3(-(d * i - f * g), (a * i - c * g), -(a * f - c * d)),
        float3((d * h - e * g), -(a * h - b * g), (a * e - b * d))
    );
    return adj * invDet;
}

// -------------------------------------------------------------
// B-Spline Interpolation Helpers
// -------------------------------------------------------------
inline void computeQuadraticBSplineWeights(float3 fx, thread float3x3 &w) {
    w[0] = 0.5f * (1.5f - fx) * (1.5f - fx);
    w[1] = 0.75f - (fx - 1.0f) * (fx - 1.0f);
    w[2] = 0.5f * (fx - 0.5f) * (fx - 0.5f);
}

inline void computeQuadraticBSplineWeightGradients(float3 fx, thread float3x3 &dw) {
    dw[0] = fx - 1.5f;
    dw[1] = 2.0f * (1.0f - fx);
    dw[2] = fx - 0.5f;
}

// -------------------------------------------------------------
// Constitutive Models (Stress Computations)
// -------------------------------------------------------------
inline float3x3 computeLinearElasticStress(float3x3 F, float lambda, float mu) {
    float3x3 strain = 0.5f * (transpose(F) * F - float3x3(1.0f));
    float trace_strain = strain[0][0] + strain[1][1] + strain[2][2];
    float3x3 stress2 = float3x3(1.0f) * (lambda * trace_strain) + 2.0f * mu * strain;
    return F * stress2 * transpose(F);
}

inline float3x3 computeNeoHookeanStress(float3x3 F, float lambda, float mu) {
    float J = determinant(F);
    if (J < 1e-6f) J = 1e-6f;
    float3x3 b = F * transpose(F);
    return mu * (b - float3x3(1.0f)) + float3x3(1.0f) * (lambda * log(J));
}

inline float3x3 computeCoRotationalStress(float3x3 F, float lambda, float mu) {
    float3x3 u, vt, sigma;
    svd3(F, u, vt, sigma);
    float J = determinant(F);
    if (J < 1e-6f) J = 1e-6f;
    return 2.0f * mu * (F - u * vt) * transpose(F) + float3x3(1.0f) * (lambda * J * (J - 1.0f));
}

inline float3x3 computeStress(float4 materialParams, int type, float3x3 F) {
    if (type == 0) { // eNeoHookean
        return computeNeoHookeanStress(F, materialParams.x, materialParams.y);
    } else if (type == 1) { // eCoRotational
        return computeCoRotationalStress(F, materialParams.x, materialParams.y);
    } else if (type == 2) { // eLinearElastic
        return computeLinearElasticStress(F, materialParams.x, materialParams.y);
    }
    return float3x3(0.0f);
}

// -------------------------------------------------------------
// Rigid Geometries Sdf Queries
// -------------------------------------------------------------
inline float4 getBoxSdf(float3 localPosition, float3 halfSize) {
    float3 penetration = abs(localPosition) - halfSize;
    float3 s = float3(localPosition.x < 0.0f ? -1.0f : 1.0f,
                      localPosition.y < 0.0f ? -1.0f : 1.0f,
                      localPosition.z < 0.0f ? -1.0f : 1.0f);
    float g = max(penetration.x, max(penetration.y, penetration.z));
    float3 q = max(penetration, 0.0f);
    float l = length(q);
    float3 grad;
    if (g > 0.0f) {
        grad = q / l;
    } else {
        if (penetration.x > penetration.y) {
            if (penetration.x > penetration.z) grad = float3(1, 0, 0);
            else grad = float3(0, 0, 1);
        } else {
            if (penetration.y > penetration.z) grad = float3(0, 1, 0);
            else grad = float3(0, 0, 1);
        }
    }
    return float4(s * grad, l + min(g, 0.0f));
}

inline float4 getSphereSdf(float3 localPosition, float radius) {
    float len = length(localPosition);
    float3 grad = localPosition / max(len, CR_EPS);
    return float4(grad, len - radius);
}

inline float4 getPlaneSdf(float3 localPosition) {
    return float4(0.0f, 1.0f, 0.0f, localPosition.y);
}

inline float4 getCapsuleSdf(float3 localPosition, float radius, float halfHeight) {
    float cylinderHeight = halfHeight * 2.0f;
    float3 pa = localPosition;
    pa.y -= halfHeight;
    float h = clamp(-pa.y / cylinderHeight, 0.0f, 1.0f);
    float3 q = pa;
    q.y += cylinderHeight * h;
    float d = length(q);
    float3 grad = q / max(d, CR_EPS);
    return float4(grad, d - radius);
}

inline int getGridIndex(int3 dim, int offset, int x, int y, int z) {
    return offset + (z * dim.x * dim.y) + (y * dim.x) + x;
}

inline float4 getTriangleMeshSdf(
    float3 localPosition, float4 params0,
    device const int3* sdfDimension, device const float4* sdfLowerBoundCellSize,
    device const float4* sdfGridData
) {
    int sdfDataOffset = as_type<int>(params0.x);
    int sdfDataId = as_type<int>(params0.y);
    float4 boundCell = sdfLowerBoundCellSize[sdfDataId];
    int3 dim = sdfDimension[sdfDataId];

    float3 scaledPos = (localPosition - boundCell.xyz) / boundCell.w;
    int3 mapped = int3(scaledPos + 0.5f);

    if (mapped.x < 0 || mapped.x >= dim.x ||
        mapped.y < 0 || mapped.y >= dim.y ||
        mapped.z < 0 || mapped.z >= dim.z) {
        return float4(0.0f, 1.0f, 0.0f, CR_MAX_F32);
    }

    int nearestIdx = getGridIndex(dim, sdfDataOffset, mapped.x, mapped.y, mapped.z);
    float4 result = float4(sdfGridData[nearestIdx].xyz, 0.0f);

    int3 base = int3(scaledPos);
    float3 frac = scaledPos - float3(base);

    if (base.x < 0 || base.x >= dim.x - 1 ||
        base.y < 0 || base.y >= dim.y - 1 ||
        base.z < 0 || base.z >= dim.z - 1) {
        result.w = sdfGridData[nearestIdx].w;
        return result;
    }

    float v[8];
    for (int dx = 0; dx <= 1; ++dx) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dz = 0; dz <= 1; ++dz) {
                int idx = (dx << 2) | (dy << 1) | dz;
                v[idx] = sdfGridData[getGridIndex(dim, sdfDataOffset, base.x + dx, base.y + dy, base.z + dz)].w;
            }
        }
    }

    float fx = frac.x, fy = frac.y, fz = frac.z;
    float ox = 1.0f - fx, oy = 1.0f - fy, oz = 1.0f - fz;
    float w[8] = {
        ox * oy * oz, fx * oy * oz, ox * fy * oz, fx * fy * oz,
        ox * oy * fz, fx * oy * fz, ox * fy * fz, fx * fy * fz
    };

    float interpolated = 0.0f;
    for (int i = 0; i < 8; ++i) {
        interpolated += v[i] * w[i];
    }
    result.w = interpolated;
    return result;
}

inline float4 getGeometrySdf(
    float3 localPosition, int type, float4 params0,
    device const int3* sdfDimension, device const float4* sdfLowerBoundCellSize,
    device const float4* sdfGridData, thread float3 &tangentDir, thread bool &inSpine
) {
    inSpine = false;
    float4 res = float4(0.0f);
    if (type == 0) { // eBox
        res = getBoxSdf(localPosition, params0.xyz);
    } else if (type == 1) { // eSphere
        res = getSphereSdf(localPosition, params0.x);
    } else if (type == 2) { // ePlane
        res = getPlaneSdf(localPosition);
    } else if (type == 3) { // eCapsule
        res = getCapsuleSdf(localPosition, params0.x, params0.y);
    } else if (type == 4) { // eTriangleMesh
        res = getTriangleMeshSdf(localPosition, params0, sdfDimension, sdfLowerBoundCellSize, sdfGridData);
    } else if (type == 5) { // eQuadSlicer
        float halfX = params0.x;
        float halfZ = params0.y;
        if (localPosition.x < -halfX || localPosition.x > halfX ||
            localPosition.z < -halfZ || localPosition.z > halfZ) {
            res = float4(0.0f, 0.0f, 0.0f, CR_MAX_F32);
        } else {
            res = float4(0.0f, sign(localPosition.y), 0.0f, abs(localPosition.y));
        }
    } else if (type == 6) { // eTriangleMeshSlicer
        res = float4(0.0f); // Fallback / analytical slicers
    }
    return res;
}

// -------------------------------------------------------------
// Collision Resolution Kernel Helpers
// -------------------------------------------------------------
inline void resolveRigidCollision(
    thread float4 &velocity, thread float4 &position, int numShapes,
    device const int* shapeIds,
    device const int* shapeType, device const int* shapeGeometryIdx,
    device const float3* shapePosition, device const float4* shapeRotation,
    device const float3* shapeInvScale, device const float4* shapeParams0,
    device const int* geomType, device const float4* geomParams0,
    device const int3* sdfDimension, device const float4* sdfLowerBoundCellSize,
    device const float4* sdfGridData,
    bool isParticle
) {
    for (int idx = 0; idx < numShapes; idx++) {
        int shapeId = shapeIds[idx];
        int type = shapeType[shapeId];
        int geomId = shapeGeometryIdx[shapeId];
        int gType = geomType[geomId];

        if (isParticle) {
            if (gType == 7 || gType == 8) { // eConnectedLineSegments or eArc
                continue;
            }
        }

        float3 shPos = shapePosition[shapeId];
        float4 shRot = shapeRotation[shapeId];
        float3 shInvSc = shapeInvScale[shapeId];
        float4 shParams0 = shapeParams0[shapeId];

        // Transform position to local space of shape
        float3 relPos = position.xyz - shPos;
        // Quaternion inverse rotate
        float3 q_xyz = shRot.xyz;
        float q_w = shRot.w;
        float3 uv = cross(q_xyz, relPos);
        float3 uuv = cross(q_xyz, uv);
        float3 localPos = relPos - 2.0f * (q_w * uv - uuv); // rotateInv
        localPos *= shInvSc;

        float3 tangentDir = float3(0.0f);
        bool inSpine = false;
        float4 sdfRes = getGeometrySdf(localPos, gType, geomParams0[geomId], sdfDimension, sdfLowerBoundCellSize, sdfGridData, tangentDir, inSpine);
        float dist = sdfRes.w;

        if (isnan(dist)) continue;

        // Scale distance to world units
        float3 grad = sdfRes.xyz * shInvSc;
        float grad_len = length(grad);
        float invScale = 1.0f / max(grad_len, CR_EPS);
        dist = dist * invScale;
        grad *= invScale;

        dist -= shParams0.w; // Fatten SDF
        float smoothDist = isParticle ? 0.0f : shParams0.x;

        if (dist < smoothDist) {
            // Transform normal back to world
            float3 normal = grad + 2.0f * cross(q_xyz, cross(q_xyz, grad) + q_w * grad); // rotate

            if (isParticle) {
                float inside = min(dist, 0.0f);
                position.xyz -= normal * inside;
            } else {
                float normalVel = dot(velocity.xyz, normal);
                if (normalVel < 0.0f) {
                    float friction = shParams0.z;
                    if (inSpine) friction = 1e2f;

                    float3 tangential = velocity.xyz - normal * normalVel;
                    float tangentNorm = length(tangential);
                    float frictionCorr = max(friction * normalVel / (tangentNorm + CR_EPS), -1.0f);

                    float3 response = tangential + tangential * frictionCorr;
                    response *= shParams0.y; // damping/drag coefficient (stickyScale)
                    velocity.xyz = response;
                }
            }
        }
    }
}

// -------------------------------------------------------------
// COMPUTE KERNELS
// -------------------------------------------------------------

// 1. Standard MPM Compute Initial Grid Mass
kernel void standardMpmComputeInitialGridMassKernel(
    device float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const float4* particlePositionMass [[buffer(1)]],
    device float4* particleF0 [[buffer(2)]],
    device float4* particleF1 [[buffer(3)]],
    device float4* particleF2 [[buffer(4)]],
    constant int &numParticles [[buffer(5)]],
    constant float3 &gridBoundMin [[buffer(6)]],
    constant float &invCellSize [[buffer(7)]],
    constant int3 &numNodesPerDim [[buffer(8)]],
    constant float &gridVolume [[buffer(9)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)numParticles) return;

    // Set deformation gradient to Identity
    particleF0[idx] = float4(1, 0, 0, 0);
    particleF1[idx] = float4(0, 1, 0, 0);
    particleF2[idx] = float4(0, 0, 1, 0);

    float4 pPosMass = particlePositionMass[idx];
    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int3 coord = baseCoord + int3(i, j, k);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[i].x * w[j].y * w[k].z;
                float weightedMass = weight * pPosMass.w;

                device float* basePtr = (device float*)&nodeMomentumVelocityMass[nodeIdx];
                device atomic_int* atomicNodeMass = (device atomic_int*)&basePtr[3];
                atomicAdd(atomicNodeMass, weightedMass);
            }
        }
    }
}

// 2. Standard MPM Compute Initial Volume
kernel void standardMpmComputeInitialVolumeKernel(
    device const float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const float4* particlePositionMass [[buffer(1)]],
    device float4* particleF0 [[buffer(2)]],
    device float4* particleF1 [[buffer(3)]],
    device float4* particleF2 [[buffer(4)]],
    device float* particleInitialVolume [[buffer(5)]],
    constant int &numParticles [[buffer(6)]],
    constant float3 &gridBoundMin [[buffer(7)]],
    constant float &invCellSize [[buffer(8)]],
    constant int3 &numNodesPerDim [[buffer(9)]],
    constant float &gridVolume [[buffer(10)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)numParticles) return;

    // Reset deformation gradient to Identity
    particleF0[idx] = float4(1, 0, 0, 0);
    particleF1[idx] = float4(0, 1, 0, 0);
    particleF2[idx] = float4(0, 0, 1, 0);

    float4 pPosMass = particlePositionMass[idx];
    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    float density = 0.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int3 coord = baseCoord + int3(i, j, k);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[i].x * w[j].y * w[k].z;
                float nodeDensity = nodeMomentumVelocityMass[nodeIdx].w / gridVolume;
                density += nodeDensity * weight;
            }
        }
    }

    particleInitialVolume[idx] = density > 1e-8f ? pPosMass.w / density : 1e-8f;
}

// 3. Standard MPM Particle to Grid (P2G)
kernel void standardMpmParticleToGridKernel(
    device float4* nodeMomentumVelocityMass [[buffer(0)]],
    device float3* nodeForce [[buffer(1)]],
    device const float4* particlePositionMass [[buffer(2)]],
    device const float3* particleVelocity [[buffer(3)]],
    device const float* particleInitialVolume [[buffer(4)]],
    device const float4* particleF0 [[buffer(5)]],
    device const float4* particleF1 [[buffer(6)]],
    device const float4* particleF2 [[buffer(7)]],
    device const float4* particleMaterialProperties0 [[buffer(8)]],
    device const int* particleMaterialTypes [[buffer(9)]],
    constant int &numParticles [[buffer(10)]],
    constant float3 &gravity [[buffer(11)]],
    constant float3 &gridBoundMin [[buffer(12)]],
    constant float &invCellSize [[buffer(13)]],
    constant int3 &numNodesPerDim [[buffer(14)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)numParticles) return;

    float4 pPosMass = particlePositionMass[idx];
    float3 velocity = particleVelocity[idx];
    float initVolume = particleInitialVolume[idx];
    float3x3 F = float3x3(particleF0[idx].xyz, particleF1[idx].xyz, particleF2[idx].xyz);
    float4 matParams = particleMaterialProperties0[idx];
    int matType = particleMaterialTypes[idx];

    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w, dw;
    computeQuadraticBSplineWeights(fx, w);
    computeQuadraticBSplineWeightGradients(fx, dw);

    float currentVolume = initVolume * determinant(F);
    float3x3 stress = computeStress(matParams, matType, F);

    float mass = pPosMass.w;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int3 coord = baseCoord + int3(i, j, k);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[i].x * w[j].y * w[k].z;
                float3 weightGrad = float3(
                    dw[i].x * w[j].y * w[k].z,
                    w[i].x * dw[j].y * w[k].z,
                    w[i].x * w[j].y * dw[k].z
                );

                float weightedMass = weight * mass;
                float3 extForce = gravity * weightedMass;
                float3 intForce = (stress * weightGrad) * (-initVolume);
                float3 force = extForce + intForce;

                // Scatter Mass and Momentum (Atomic Addition on float4 structure mapped to atomic_int array)
                device float* basePtr = (device float*)&nodeMomentumVelocityMass[nodeIdx];
                device atomic_int* atomicMass = (device atomic_int*)&basePtr[3];
                atomicAdd(atomicMass, weightedMass);

                device atomic_int* atomicMomX = (device atomic_int*)&basePtr[0];
                device atomic_int* atomicMomY = (device atomic_int*)&basePtr[1];
                device atomic_int* atomicMomZ = (device atomic_int*)&basePtr[2];
                
                atomicAdd(atomicMomX, velocity.x * weightedMass);
                atomicAdd(atomicMomY, velocity.y * weightedMass);
                atomicAdd(atomicMomZ, velocity.z * weightedMass);

                // Scatter Force
                device float* baseForcePtr = (device float*)&nodeForce[nodeIdx];
                device atomic_int* atomicForceX = (device atomic_int*)&baseForcePtr[0];
                device atomic_int* atomicForceY = (device atomic_int*)&baseForcePtr[1];
                device atomic_int* atomicForceZ = (device atomic_int*)&baseForcePtr[2];

                atomicAdd(atomicForceX, force.x);
                atomicAdd(atomicForceY, force.y);
                atomicAdd(atomicForceZ, force.z);
            }
        }
    }
}

// 4. Standard MPM Update Grid
kernel void standardMpmUpdateGridKernel(
    device float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const float3* nodeForce [[buffer(1)]],
    device const int* shapeIds [[buffer(2)]],
    // ShapeData pointer arrays
    device const int* shapeType [[buffer(3)]],
    device const int* shapeGeometryIdx [[buffer(4)]],
    device const float3* shapePosition [[buffer(5)]],
    device const float4* shapeRotation [[buffer(6)]],
    device const float3* shapeInvScale [[buffer(7)]],
    device const float4* shapeParams0 [[buffer(8)]],
    // GeometryData
    device const int* geomType [[buffer(9)]],
    device const float4* geomParams0 [[buffer(10)]],
    // GeometrySdfData
    device const int3* sdfDimension [[buffer(11)]],
    device const float4* sdfLowerBoundCellSize [[buffer(12)]],
    device const float4* sdfGridData [[buffer(13)]],
    constant int &numNodes [[buffer(14)]],
    constant float3 &gridBoundMin [[buffer(15)]],
    constant float &cellSize [[buffer(16)]],
    constant int3 &numNodesPerDim [[buffer(17)]],
    constant float &integrationStepSize [[buffer(18)]],
    constant int &numShapes [[buffer(19)]],
    uint nodeIdx [[thread_position_in_grid]]
) {
    if (nodeIdx >= (uint)numNodes) return;

    float4 velMass = nodeMomentumVelocityMass[nodeIdx];
    if (velMass.w < 1e-4f) return;

    // Velocity integration
    float3 force = nodeForce[nodeIdx];
    velMass.xyz += force * integrationStepSize;

    // Convert to velocity
    velMass.xyz /= velMass.w;

    // Calculate grid node position
    int i = nodeIdx % numNodesPerDim.x;
    int j = (nodeIdx / numNodesPerDim.x) % numNodesPerDim.y;
    int k = nodeIdx / (numNodesPerDim.x * numNodesPerDim.y);
    float3 nodePosition = float3(i * cellSize, j * cellSize, k * cellSize) + gridBoundMin;
    nodePosition += velMass.xyz * integrationStepSize;

    float4 pos = float4(nodePosition, 1.0f);
    resolveRigidCollision(
        velMass, pos, numShapes, shapeIds,
        shapeType, shapeGeometryIdx, shapePosition, shapeRotation, shapeInvScale, shapeParams0,
        geomType, geomParams0, sdfDimension, sdfLowerBoundCellSize, sdfGridData,
        false
    );

    // Boundary conditions
    if (i < 2 || i > numNodesPerDim.x - 3) velMass.x = 0;
    if (j < 2 || j > numNodesPerDim.y - 3) velMass.y = 0;
    if (k < 2 || k > numNodesPerDim.z - 3) velMass.z = 0;

    nodeMomentumVelocityMass[nodeIdx] = velMass;
}

// 5. Standard MPM Grid to Particle (G2P)
kernel void standardMpmGridToParticleKernel(
    device float4* particlePositionMass [[buffer(0)]],
    device float3* particleVelocity [[buffer(1)]],
    device float4* particleF0 [[buffer(2)]],
    device float4* particleF1 [[buffer(3)]],
    device float4* particleF2 [[buffer(4)]],
    device const float4* nodeMomentumVelocityMass [[buffer(5)]],
    device const int* shapeIds [[buffer(6)]],
    // ShapeData pointer arrays
    device const int* shapeType [[buffer(7)]],
    device const int* shapeGeometryIdx [[buffer(8)]],
    device const float3* shapePosition [[buffer(9)]],
    device const float4* shapeRotation [[buffer(10)]],
    device const float3* shapeInvScale [[buffer(11)]],
    device const float4* shapeParams0 [[buffer(12)]],
    // GeometryData
    device const int* geomType [[buffer(13)]],
    device const float4* geomParams0 [[buffer(14)]],
    // GeometrySdfData
    device const int3* sdfDimension [[buffer(15)]],
    device const float4* sdfLowerBoundCellSize [[buffer(16)]],
    device const float4* sdfGridData [[buffer(17)]],
    constant int &numParticles [[buffer(18)]],
    constant float3 &gridBoundMin [[buffer(19)]],
    constant float3 &gridBoundMax [[buffer(20)]],
    constant float &invCellSize [[buffer(21)]],
    constant int3 &numNodesPerDim [[buffer(22)]],
    constant float &integrationStepSize [[buffer(23)]],
    constant int &numShapes [[buffer(24)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)numParticles) return;

    float4 pPosMass = particlePositionMass[idx];
    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w, dw;
    computeQuadraticBSplineWeights(fx, w);
    computeQuadraticBSplineWeightGradients(fx, dw);

    float3 pVel = float3(0.0f);
    float3x3 gradVel = float3x3(0.0f);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int3 coord = baseCoord + int3(i, j, k);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[i].x * w[j].y * w[k].z;
                float3 nodeVel = nodeMomentumVelocityMass[nodeIdx].xyz;

                pVel += nodeVel * weight;

                float3 weightGrad = float3(
                    dw[i].x * w[j].y * w[k].z,
                    w[i].x * dw[j].y * w[k].z,
                    w[i].x * w[j].y * dw[k].z
                );
                // Outer product: gradVel += weightGrad * nodeVel^T
                gradVel += float3x3(weightGrad * nodeVel.x, weightGrad * nodeVel.y, weightGrad * nodeVel.z);
            }
        }
    }

    particleVelocity[idx] = pVel;
    pPosMass.xyz += pVel * integrationStepSize;
    pPosMass.xyz = clamp(pPosMass.xyz, gridBoundMin, gridBoundMax);

    // Resolve collision at particle level
    resolveRigidCollision(
        pPosMass, pPosMass, numShapes, shapeIds,
        shapeType, shapeGeometryIdx, shapePosition, shapeRotation, shapeInvScale, shapeParams0,
        geomType, geomParams0, sdfDimension, sdfLowerBoundCellSize, sdfGridData,
        true
    );
    particlePositionMass[idx] = pPosMass;

    // Update gradient deformation tensor F
    float3x3 F = float3x3(particleF0[idx].xyz, particleF1[idx].xyz, particleF2[idx].xyz);
    F = (float3x3(1.0f) + gradVel * integrationStepSize) * F;
    particleF0[idx] = float4(F[0], 0);
    particleF1[idx] = float4(F[1], 0);
    particleF2[idx] = float4(F[2], 0);
}

// -------------------------------------------------------------
// MLS-MPM SHADERS
// -------------------------------------------------------------

// 6. MLS-MPM Compute Initial Grid Mass
kernel void mlsMpmComputeInitialGridMassKernel(
    device float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const float4* particlePositionMass [[buffer(1)]],
    device float4* particleF0 [[buffer(2)]],
    device float4* particleF1 [[buffer(3)]],
    device float4* particleF2 [[buffer(4)]],
    device const uint* indices [[buffer(5)]],
    constant int &numParticlesToCompute [[buffer(6)]],
    constant float3 &gridBoundMin [[buffer(7)]],
    constant float &invCellSize [[buffer(8)]],
    constant int3 &numNodesPerDim [[buffer(9)]],
    constant float &cellVolume [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= (uint)numParticlesToCompute) return;

    uint idx = indices[i];
    particleF0[idx] = float4(1, 0, 0, 0);
    particleF1[idx] = float4(0, 1, 0, 0);
    particleF2[idx] = float4(0, 0, 1, 0);

    float4 pPosMass = particlePositionMass[idx];
    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
                int3 coord = baseCoord + int3(ix, iy, iz);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[ix].x * w[iy].y * w[iz].z;
                float weightedMass = weight * pPosMass.w;

                device float* basePtr = (device float*)&nodeMomentumVelocityMass[nodeIdx];
                device atomic_int* atomicNodeMass = (device atomic_int*)&basePtr[3];
                atomicAdd(atomicNodeMass, weightedMass);
            }
        }
    }
}

// 7. MLS-MPM Compute Initial Volume
kernel void mlsMpmComputeInitialVolumeKernel(
    device const float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const float4* particlePositionMass [[buffer(1)]],
    device float4* particleF0 [[buffer(2)]],
    device float4* particleF1 [[buffer(3)]],
    device float4* particleF2 [[buffer(4)]],
    device float* particleInitialVolume [[buffer(5)]],
    device const uint* indices [[buffer(6)]],
    constant int &numParticlesToCompute [[buffer(7)]],
    constant float3 &gridBoundMin [[buffer(8)]],
    constant float &invCellSize [[buffer(9)]],
    constant int3 &numNodesPerDim [[buffer(10)]],
    constant float &cellVolume [[buffer(11)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= (uint)numParticlesToCompute) return;

    uint idx = indices[i];
    particleF0[idx] = float4(1, 0, 0, 0);
    particleF1[idx] = float4(0, 1, 0, 0);
    particleF2[idx] = float4(0, 0, 1, 0);

    float4 pPosMass = particlePositionMass[idx];
    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    float density = 0.0f;
    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
                int3 coord = baseCoord + int3(ix, iy, iz);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[ix].x * w[iy].y * w[iz].z;
                float nodeDensity = nodeMomentumVelocityMass[nodeIdx].w / cellVolume;
                density += nodeDensity * weight;
            }
        }
    }

    particleInitialVolume[idx] = density > 1e-8f ? pPosMass.w / density : 1e-8f;
}

// 8. MLS-MPM Particle to Grid (P2G)
kernel void mlsMpmParticleToGridKernel(
    device float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const float4* particlePositionMass [[buffer(1)]],
    device const float* particleInitialVolume [[buffer(2)]],
    device const float4* particleF0 [[buffer(3)]],
    device const float4* particleF1 [[buffer(4)]],
    device const float4* particleF2 [[buffer(5)]],
    device const float4* particleAffineMomentumColumn0 [[buffer(6)]],
    device const float4* particleAffineMomentumColumn1 [[buffer(7)]],
    device const float4* particleAffineMomentumColumn2 [[buffer(8)]],
    device const float4* particleMaterialProperties0 [[buffer(9)]],
    device const int* particleMaterialTypes [[buffer(10)]],
    device const float3* particleVelocity [[buffer(11)]],
    device const unsigned char* activeMask [[buffer(12)]],
    constant int &numActiveParticles [[buffer(13)]],
    constant float &integrationStepSize [[buffer(14)]],
    constant float3 &gravity [[buffer(15)]],
    constant float3 &gridBoundMin [[buffer(16)]],
    constant float &invCellSize [[buffer(17)]],
    constant float &cellSize [[buffer(18)]],
    constant int3 &numNodesPerDim [[buffer(19)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)numActiveParticles || !activeMask[idx]) return;

    float4 pPosMass = particlePositionMass[idx];
    float initVolume = particleInitialVolume[idx];
    float3x3 F = float3x3(particleF0[idx].xyz, particleF1[idx].xyz, particleF2[idx].xyz);
    float3x3 B = float3x3(particleAffineMomentumColumn0[idx].xyz, particleAffineMomentumColumn1[idx].xyz, particleAffineMomentumColumn2[idx].xyz);
    float4 matParams = particleMaterialProperties0[idx];
    int matType = particleMaterialTypes[idx];
    float3 velocity = particleVelocity[idx];

    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    float currentVolume = initVolume * determinant(F);
    float3x3 stress = computeStress(matParams, matType, F);

    float3x3 invD = float3x3(float3(4.0f * invCellSize * invCellSize, 0, 0),
                             float3(0, 4.0f * invCellSize * invCellSize, 0),
                             float3(0, 0, 4.0f * invCellSize * invCellSize));
    float3x3 stressForceTerm = stress * (-integrationStepSize * initVolume * 4.0f * invCellSize * invCellSize);

    float mass = pPosMass.w;

    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
                int3 coord = baseCoord + int3(ix, iy, iz);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[ix].x * w[iy].y * w[iz].z;
                float3 xi_minus_xp = (float3(coord) * cellSize + gridBoundMin) - pPosMass.xyz;

                // Affine stress force term and momentum calculation
                float3 momentum_term = (velocity + B * xi_minus_xp) * (weight * mass);
                float3 force_term = (stressForceTerm * xi_minus_xp) * weight;

                // Add gravity force
                float3 gravity_force = gravity * (weight * mass * integrationStepSize);

                float3 total_momentum = momentum_term + force_term + gravity_force;

                device float* basePtr = (device float*)&nodeMomentumVelocityMass[nodeIdx];
                device atomic_int* atomicMass = (device atomic_int*)&basePtr[3];
                atomicAdd(atomicMass, weight * mass);

                device atomic_int* atomicMomX = (device atomic_int*)&basePtr[0];
                device atomic_int* atomicMomY = (device atomic_int*)&basePtr[1];
                device atomic_int* atomicMomZ = (device atomic_int*)&basePtr[2];

                atomicAdd(atomicMomX, total_momentum.x);
                atomicAdd(atomicMomY, total_momentum.y);
                atomicAdd(atomicMomZ, total_momentum.z);
            }
        }
    }
}

// 9. MLS-MPM Update Grid
kernel void mlsMpmUpdateGridKernel(
    device float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const int* shapeIds [[buffer(1)]],
    device const int* shapeType [[buffer(2)]],
    device const int* shapeGeometryIdx [[buffer(3)]],
    device const float3* shapePosition [[buffer(4)]],
    device const float4* shapeRotation [[buffer(5)]],
    device const float3* shapeInvScale [[buffer(6)]],
    device const float4* shapeParams0 [[buffer(7)]],
    device const int* geomType [[buffer(8)]],
    device const float4* geomParams0 [[buffer(9)]],
    device const int3* sdfDimension [[buffer(10)]],
    device const float4* sdfLowerBoundCellSize [[buffer(11)]],
    device const float4* sdfGridData [[buffer(12)]],
    constant int &numNodes [[buffer(13)]],
    constant float3 &gridBoundMin [[buffer(14)]],
    constant float &cellSize [[buffer(15)]],
    constant int3 &numNodesPerDim [[buffer(16)]],
    constant float &integrationStepSize [[buffer(17)]],
    constant int &numShapes [[buffer(18)]],
    uint nodeIdx [[thread_position_in_grid]]
) {
    if (nodeIdx >= (uint)numNodes) return;

    float4 velMass = nodeMomentumVelocityMass[nodeIdx];
    if (velMass.w < 1e-4f) return;

    // Convert momentum to velocity
    velMass.xyz /= velMass.w;

    // Collision with rigid shapes
    int i = nodeIdx % numNodesPerDim.x;
    int j = (nodeIdx / numNodesPerDim.x) % numNodesPerDim.y;
    int k = nodeIdx / (numNodesPerDim.x * numNodesPerDim.y);
    float3 nodePosition = float3(i * cellSize, j * cellSize, k * cellSize) + gridBoundMin;
    nodePosition += velMass.xyz * integrationStepSize;

    float4 pos = float4(nodePosition, 1.0f);
    resolveRigidCollision(
        velMass, pos, numShapes, shapeIds,
        shapeType, shapeGeometryIdx, shapePosition, shapeRotation, shapeInvScale, shapeParams0,
        geomType, geomParams0, sdfDimension, sdfLowerBoundCellSize, sdfGridData,
        false
    );

    // Boundary conditions
    if (i < 2 || i > numNodesPerDim.x - 3) velMass.x = 0;
    if (j < 2 || j > numNodesPerDim.y - 3) velMass.y = 0;
    if (k < 2 || k > numNodesPerDim.z - 3) velMass.z = 0;

    // Convert back to momentum for G2P
    velMass.xyz *= velMass.w;
    nodeMomentumVelocityMass[nodeIdx] = velMass;
}

// 10. MLS-MPM Grid to Particle (G2P)
kernel void mlsMpmGridToParticleKernel(
    device float4* particlePositionMass [[buffer(0)]],
    device float3* particleVelocity [[buffer(1)]],
    device float4* particleF0 [[buffer(2)]],
    device float4* particleF1 [[buffer(3)]],
    device float4* particleF2 [[buffer(4)]],
    device float4* particleAffineMomentumColumn0 [[buffer(5)]],
    device float4* particleAffineMomentumColumn1 [[buffer(6)]],
    device float4* particleAffineMomentumColumn2 [[buffer(7)]],
    device const float4* nodeMomentumVelocityMass [[buffer(8)]],
    device const int* shapeIds [[buffer(9)]],
    device const int* shapeType [[buffer(10)]],
    device const int* shapeGeometryIdx [[buffer(11)]],
    device const float3* shapePosition [[buffer(12)]],
    device const float4* shapeRotation [[buffer(13)]],
    device const float3* shapeInvScale [[buffer(14)]],
    device const float4* shapeParams0 [[buffer(15)]],
    device const int* geomType [[buffer(16)]],
    device const float4* geomParams0 [[buffer(17)]],
    device const int3* sdfDimension [[buffer(18)]],
    device const float4* sdfLowerBoundCellSize [[buffer(19)]],
    device const float4* sdfGridData [[buffer(20)]],
    device const unsigned char* activeMask [[buffer(21)]],
    constant int &numActiveParticles [[buffer(22)]],
    constant float3 &gridBoundMin [[buffer(23)]],
    constant float3 &gridBoundMax [[buffer(24)]],
    constant float &invCellSize [[buffer(25)]],
    constant float &cellSize [[buffer(26)]],
    constant int3 &numNodesPerDim [[buffer(27)]],
    constant float &integrationStepSize [[buffer(28)]],
    constant int &numShapes [[buffer(29)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)numActiveParticles || !activeMask[idx]) return;

    float4 pPosMass = particlePositionMass[idx];
    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    float3 pVel = float3(0.0f);
    float3x3 B = float3x3(0.0f);

    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
                int3 coord = baseCoord + int3(ix, iy, iz);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[ix].x * w[iy].y * w[iz].z;
                float4 nodeMomentumMass = nodeMomentumVelocityMass[nodeIdx];
                if (nodeMomentumMass.w < 1e-4f) continue;

                float3 nodeVel = nodeMomentumMass.xyz / nodeMomentumMass.w;
                pVel += nodeVel * weight;

                float3 xi_minus_xp = (float3(coord) * cellSize + gridBoundMin) - pPosMass.xyz;
                B += float3x3(nodeVel * xi_minus_xp.x, nodeVel * xi_minus_xp.y, nodeVel * xi_minus_xp.z) * (weight * 4.0f * invCellSize * invCellSize);
            }
        }
    }

    particleVelocity[idx] = pVel;
    pPosMass.xyz += pVel * integrationStepSize;
    pPosMass.xyz = clamp(pPosMass.xyz, gridBoundMin, gridBoundMax);

    // Resolve collision at particle level
    resolveRigidCollision(
        pPosMass, pPosMass, numShapes, shapeIds,
        shapeType, shapeGeometryIdx, shapePosition, shapeRotation, shapeInvScale, shapeParams0,
        geomType, geomParams0, sdfDimension, sdfLowerBoundCellSize, sdfGridData,
        true
    );
    particlePositionMass[idx] = pPosMass;

    // Update gradient deformation tensor F
    float3x3 F = float3x3(particleF0[idx].xyz, particleF1[idx].xyz, particleF2[idx].xyz);
    F = (float3x3(1.0f) + B * integrationStepSize) * F;
    particleF0[idx] = float4(F[0], 0);
    particleF1[idx] = float4(F[1], 0);
    particleF2[idx] = float4(F[2], 0);

    particleAffineMomentumColumn0[idx] = float4(B[0], 0);
    particleAffineMomentumColumn1[idx] = float4(B[1], 0);
    particleAffineMomentumColumn2[idx] = float4(B[2], 0);
}

// -------------------------------------------------------------
// PB-MPM SHADERS
// -------------------------------------------------------------

// 11. PB-MPM Compute Initial Grid Mass
kernel void pbMpmComputeInitialGridMassKernel(
    device float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const float4* particlePositionMass [[buffer(1)]],
    device float4* particleF0 [[buffer(2)]],
    device float4* particleF1 [[buffer(3)]],
    device float4* particleF2 [[buffer(4)]],
    device const uint* indices [[buffer(5)]],
    constant int &numParticlesToCompute [[buffer(6)]],
    constant float3 &gridBoundMin [[buffer(7)]],
    constant float &invCellSize [[buffer(8)]],
    constant int3 &numNodesPerDim [[buffer(9)]],
    constant float &cellVolume [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= (uint)numParticlesToCompute) return;

    uint idx = indices[i];
    particleF0[idx] = float4(1, 0, 0, 0);
    particleF1[idx] = float4(0, 1, 0, 0);
    particleF2[idx] = float4(0, 0, 1, 0);

    float4 pPosMass = particlePositionMass[idx];
    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
                int3 coord = baseCoord + int3(ix, iy, iz);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[ix].x * w[iy].y * w[iz].z;
                float weightedMass = weight * pPosMass.w;

                device float* basePtr = (device float*)&nodeMomentumVelocityMass[nodeIdx];
                device atomic_int* atomicNodeMass = (device atomic_int*)&basePtr[3];
                atomicAdd(atomicNodeMass, weightedMass);
            }
        }
    }
}

// 12. PB-MPM Compute Initial Volume
kernel void pbMpmComputeInitialVolumeKernel(
    device const float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const float4* particlePositionMass [[buffer(1)]],
    device float4* particleF0 [[buffer(2)]],
    device float4* particleF1 [[buffer(3)]],
    device float4* particleF2 [[buffer(4)]],
    device float* particleInitialVolume [[buffer(5)]],
    device const uint* indices [[buffer(6)]],
    constant int &numParticlesToCompute [[buffer(7)]],
    constant float3 &gridBoundMin [[buffer(8)]],
    constant float &invCellSize [[buffer(9)]],
    constant int3 &numNodesPerDim [[buffer(10)]],
    constant float &cellVolume [[buffer(11)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= (uint)numParticlesToCompute) return;

    uint idx = indices[i];
    particleF0[idx] = float4(1, 0, 0, 0);
    particleF1[idx] = float4(0, 1, 0, 0);
    particleF2[idx] = float4(0, 0, 1, 0);

    float4 pPosMass = particlePositionMass[idx];
    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    float density = 0.0f;
    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
                int3 coord = baseCoord + int3(ix, iy, iz);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[ix].x * w[iy].y * w[iz].z;
                float nodeDensity = nodeMomentumVelocityMass[nodeIdx].w / cellVolume;
                density += nodeDensity * weight;
            }
        }
    }

    particleInitialVolume[idx] = density > 1e-8f ? pPosMass.w / density : 1e-8f;
}

// 13. PB-MPM Particle to Grid (P2G)
kernel void pbMpmParticleToGridKernel(
    device float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const float4* particlePositionMass [[buffer(1)]],
    device const float3* particleVelocity [[buffer(2)]],
    device const float4* particleF0 [[buffer(3)]],
    device const float4* particleF1 [[buffer(4)]],
    device const float4* particleF2 [[buffer(5)]],
    device const float4* particleAffineMomentumColumn0 [[buffer(6)]],
    device const float4* particleAffineMomentumColumn1 [[buffer(7)]],
    device const float4* particleAffineMomentumColumn2 [[buffer(8)]],
    device const float4* particleMaterialProperties0 [[buffer(9)]],
    device const int* particleMaterialTypes [[buffer(10)]],
    device const unsigned char* activeMask [[buffer(11)]],
    constant int &numActiveParticles [[buffer(12)]],
    constant float &integrationStepSize [[buffer(13)]],
    constant float3 &gridBoundMin [[buffer(14)]],
    constant float &invCellSize [[buffer(15)]],
    constant float &cellSize [[buffer(16)]],
    constant int3 &numNodesPerDim [[buffer(17)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)numActiveParticles || !activeMask[idx]) return;

    float4 pPosMass = particlePositionMass[idx];
    float3 velocity = particleVelocity[idx];
    float3x3 B = float3x3(particleAffineMomentumColumn0[idx].xyz, particleAffineMomentumColumn1[idx].xyz, particleAffineMomentumColumn2[idx].xyz);

    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    float mass = pPosMass.w;

    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
                int3 coord = baseCoord + int3(ix, iy, iz);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[ix].x * w[iy].y * w[iz].z;
                float3 xi_minus_xp = (float3(coord) * cellSize + gridBoundMin) - pPosMass.xyz;

                float3 total_momentum = (velocity + B * xi_minus_xp) * (weight * mass);

                device float* basePtr = (device float*)&nodeMomentumVelocityMass[nodeIdx];
                device atomic_int* atomicMass = (device atomic_int*)&basePtr[3];
                atomicAdd(atomicMass, weight * mass);

                device atomic_int* atomicMomX = (device atomic_int*)&basePtr[0];
                device atomic_int* atomicMomY = (device atomic_int*)&basePtr[1];
                device atomic_int* atomicMomZ = (device atomic_int*)&basePtr[2];

                atomicAdd(atomicMomX, total_momentum.x);
                atomicAdd(atomicMomY, total_momentum.y);
                atomicAdd(atomicMomZ, total_momentum.z);
            }
        }
    }
}

// 14. PB-MPM Update Grid
kernel void pbMpmUpdateGridKernel(
    device float4* nodeMomentumVelocityMass [[buffer(0)]],
    device const int* shapeIds [[buffer(1)]],
    device const int* shapeType [[buffer(2)]],
    device const int* shapeGeometryIdx [[buffer(3)]],
    device const float3* shapePosition [[buffer(4)]],
    device const float4* shapeRotation [[buffer(5)]],
    device const float3* shapeInvScale [[buffer(6)]],
    device const float4* shapeParams0 [[buffer(7)]],
    device const int* geomType [[buffer(8)]],
    device const float4* geomParams0 [[buffer(9)]],
    device const int3* sdfDimension [[buffer(10)]],
    device const float4* sdfLowerBoundCellSize [[buffer(11)]],
    device const float4* sdfGridData [[buffer(12)]],
    constant int &numNodes [[buffer(13)]],
    constant float3 &gridBoundMin [[buffer(14)]],
    constant float &cellSize [[buffer(15)]],
    constant int3 &numNodesPerDim [[buffer(16)]],
    constant float &integrationStepSize [[buffer(17)]],
    constant int &numShapes [[buffer(18)]],
    uint nodeIdx [[thread_position_in_grid]]
) {
    if (nodeIdx >= (uint)numNodes) return;

    float4 velMass = nodeMomentumVelocityMass[nodeIdx];
    if (velMass.w < 1e-4f) return;

    velMass.xyz /= velMass.w;

    int i = nodeIdx % numNodesPerDim.x;
    int j = (nodeIdx / numNodesPerDim.x) % numNodesPerDim.y;
    int k = nodeIdx / (numNodesPerDim.x * numNodesPerDim.y);
    float3 nodePosition = float3(i * cellSize, j * cellSize, k * cellSize) + gridBoundMin;
    nodePosition += velMass.xyz * integrationStepSize;

    float4 pos = float4(nodePosition, 1.0f);
    resolveRigidCollision(
        velMass, pos, numShapes, shapeIds,
        shapeType, shapeGeometryIdx, shapePosition, shapeRotation, shapeInvScale, shapeParams0,
        geomType, geomParams0, sdfDimension, sdfLowerBoundCellSize, sdfGridData,
        false
    );

    if (i < 2 || i > numNodesPerDim.x - 3) velMass.x = 0;
    if (j < 2 || j > numNodesPerDim.y - 3) velMass.y = 0;
    if (k < 2 || k > numNodesPerDim.z - 3) velMass.z = 0;

    velMass.xyz *= velMass.w;
    nodeMomentumVelocityMass[nodeIdx] = velMass;
}

// 15. PB-MPM Grid to Particle (G2P)
kernel void pbMpmGridToParticleKernel(
    device float4* particlePositionMass [[buffer(0)]],
    device float3* particleVelocity [[buffer(1)]],
    device float4* particleAffineMomentumColumn0 [[buffer(2)]],
    device float4* particleAffineMomentumColumn1 [[buffer(3)]],
    device float4* particleAffineMomentumColumn2 [[buffer(4)]],
    device const float4* nodeMomentumVelocityMass [[buffer(5)]],
    device const unsigned char* activeMask [[buffer(6)]],
    constant int &numActiveParticles [[buffer(7)]],
    constant float3 &gridBoundMin [[buffer(8)]],
    constant float3 &gridBoundMax [[buffer(9)]],
    constant float &invCellSize [[buffer(10)]],
    constant float &cellSize [[buffer(11)]],
    constant float &cellVolume [[buffer(12)]],
    constant int3 &numNodesPerDim [[buffer(13)]],
    constant float &integrationStepSize [[buffer(14)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)numActiveParticles || !activeMask[idx]) return;

    float4 pPosMass = particlePositionMass[idx];
    float3 scaledPos = (pPosMass.xyz - gridBoundMin) * invCellSize;
    int3 baseCoord = int3(scaledPos - 0.5f);
    float3 fx = scaledPos - float3(baseCoord);

    float3x3 w;
    computeQuadraticBSplineWeights(fx, w);

    float3 pVel = float3(0.0f);
    float3x3 B = float3x3(0.0f);

    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
                int3 coord = baseCoord + int3(ix, iy, iz);
                if (coord.x < 0 || coord.x >= numNodesPerDim.x ||
                    coord.y < 0 || coord.y >= numNodesPerDim.y ||
                    coord.z < 0 || coord.z >= numNodesPerDim.z) continue;

                int nodeIdx = coord.x + coord.y * numNodesPerDim.x + coord.z * numNodesPerDim.x * numNodesPerDim.y;
                float weight = w[ix].x * w[iy].y * w[iz].z;
                float4 nodeMomentumMass = nodeMomentumVelocityMass[nodeIdx];
                if (nodeMomentumMass.w < 1e-4f) continue;

                float3 nodeVel = nodeMomentumMass.xyz / nodeMomentumMass.w;
                pVel += nodeVel * weight;

                float3 xi_minus_xp = (float3(coord) * cellSize + gridBoundMin) - pPosMass.xyz;
                B += float3x3(nodeVel * xi_minus_xp.x, nodeVel * xi_minus_xp.y, nodeVel * xi_minus_xp.z) * (weight * 4.0f * invCellSize * invCellSize);
            }
        }
    }

    particleVelocity[idx] = pVel;
    particleAffineMomentumColumn0[idx] = float4(B[0], 0);
    particleAffineMomentumColumn1[idx] = float4(B[1], 0);
    particleAffineMomentumColumn2[idx] = float4(B[2], 0);
}

// 16. PB-MPM Integrate Particle
kernel void pbMpmIntegrateParticleKernel(
    device float4* particlePositionMass [[buffer(0)]],
    device float3* particleVelocity [[buffer(1)]],
    device float4* particleF0 [[buffer(2)]],
    device float4* particleF1 [[buffer(3)]],
    device float4* particleF2 [[buffer(4)]],
    device const float4* particleAffineMomentumColumn0 [[buffer(5)]],
    device const float4* particleAffineMomentumColumn1 [[buffer(6)]],
    device const float4* particleAffineMomentumColumn2 [[buffer(7)]],
    device const int* particleMaterialTypes [[buffer(8)]],
    device const int* shapeIds [[buffer(9)]],
    // ShapeData pointer arrays
    device const int* shapeType [[buffer(10)]],
    device const int* shapeGeometryIdx [[buffer(11)]],
    device const float3* shapePosition [[buffer(12)]],
    device const float4* shapeRotation [[buffer(13)]],
    device const float3* shapeInvScale [[buffer(14)]],
    device const float4* shapeParams0 [[buffer(15)]],
    // GeometryData
    device const int* geomType [[buffer(16)]],
    device const float4* geomParams0 [[buffer(17)]],
    // GeometrySdfData
    device const int3* sdfDimension [[buffer(18)]],
    device const float4* sdfLowerBoundCellSize [[buffer(19)]],
    device const float4* sdfGridData [[buffer(20)]],
    device const unsigned char* activeMask [[buffer(21)]],
    constant int &numActiveParticles [[buffer(22)]],
    constant float3 &gridBoundMin [[buffer(23)]],
    constant float3 &gridBoundMax [[buffer(24)]],
    constant float &integrationStepSize [[buffer(25)]],
    constant float3 &gravity [[buffer(26)]],
    constant int &numShapes [[buffer(27)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)numActiveParticles || !activeMask[idx]) return;

    float4 pPosMass = particlePositionMass[idx];
    float3 pVel = particleVelocity[idx];
    float3x3 B = float3x3(particleAffineMomentumColumn0[idx].xyz, particleAffineMomentumColumn1[idx].xyz, particleAffineMomentumColumn2[idx].xyz);
    int matType = particleMaterialTypes[idx];

    // PB-MPM velocity update under gravity
    pVel += gravity * integrationStepSize;

    // Constrain F matrix
    float3x3 F = float3x3(particleF0[idx].xyz, particleF1[idx].xyz, particleF2[idx].xyz);
    float3x3 targetF = (float3x3(1.0f) + B * integrationStepSize) * F;

    if (matType == 0 || matType == 1) { // eNeoHookean, eCoRotational (solve constraints)
        float3x3 u, vt, sigma;
        svd3(targetF, u, vt, sigma);

        float detF = determinant(targetF);
        float clampDetF = clamp(abs(detF), 0.1f, 1000.0f);
        float3x3 Q = (1.0f / (sign(detF) * sqrt(clampDetF))) * targetF;
        // ElastictyRatio is statically 0.9f
        targetF = 0.9f * (u * vt) + 0.1f * Q;
    }

    pPosMass.xyz += pVel * integrationStepSize;
    pPosMass.xyz = clamp(pPosMass.xyz, gridBoundMin, gridBoundMax);

    // Resolve collision at particle level
    resolveRigidCollision(
        pPosMass, pPosMass, numShapes, shapeIds,
        shapeType, shapeGeometryIdx, shapePosition, shapeRotation, shapeInvScale, shapeParams0,
        geomType, geomParams0, sdfDimension, sdfLowerBoundCellSize, sdfGridData,
        true
    );
    particlePositionMass[idx] = pPosMass;

    particleF0[idx] = float4(targetF[0], 0);
    particleF1[idx] = float4(targetF[1], 0);
    particleF2[idx] = float4(targetF[2], 0);
}
)";

namespace crmpm {
    namespace metal {

        static id<MTLLibrary> gLibrary = nil;
        static std::unordered_map<std::string, id<MTLComputePipelineState>> gPipelineCache;

        void compileShaders() {
            if (gLibrary) return;

            id<MTLDevice> device = getDevice();
            NSError* error = nil;
            NSString* sourceStr = [NSString stringWithUTF8String:gMpmShadersSource];
            
            gLibrary = [device newLibraryWithSource:sourceStr options:nil error:&error];
            if (!gLibrary) {
                std::cerr << "[Metal] Shader compilation error: " 
                          << [[error localizedDescription] UTF8String] << "\n";
                return;
            }
            std::cout << "[Metal] MSL compute shaders compiled successfully.\n";
        }

        MetalPipelineState getPipelineState(const char* kernelName) {
            compileShaders();
            
            std::string nameKey(kernelName);
            auto it = gPipelineCache.find(nameKey);
            if (it != gPipelineCache.end()) {
                return it->second;
            }
 
            id<MTLDevice> device = getDevice();
            NSError* error = nil;
            id<MTLFunction> function = [gLibrary newFunctionWithName:[NSString stringWithUTF8String:kernelName]];
            if (!function) {
                std::cerr << "[Metal] Failed to find kernel function: " << kernelName << "\n";
                return nil;
            }
 
            id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipelineState) {
                std::cerr << "[Metal] Failed to create pipeline state for: " << kernelName 
                          << ", error: " << [[error localizedDescription] UTF8String] << "\n";
                return nil;
            }
 
            gPipelineCache[nameKey] = pipelineState;
            return pipelineState;
        }

        // Helper to query and bind raw pointers to encoder
        inline void bindPointerToEncoder(id<MTLComputeCommandEncoder> encoder, const void* ptr, int index) {
            size_t offset = 0;
            id<MTLBuffer> buf = getBufferForPointer(ptr, &offset);
            if (buf) {
                [encoder setBuffer:buf offset:offset atIndex:index];
            } else {
                // If it is nullptr, just bind nil
                [encoder setBuffer:nil offset:0 atIndex:index];
            }
        }

        // Helper to launch a 1D compute grid
        inline void dispatch1D(id<MTLCommandBuffer> commandBuffer, id<MTLComputePipelineState> pipelineState, int count) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelineState];
            
            // Re-bind all buffers from context automatically via caller
        }

        // -------------------------------------------------------------
        // Standard MPM Kernels Dispatchers
        // -------------------------------------------------------------
        void dispatchStandardMpmComputeInitialGridMass(
            cudaStream_t stream, int numParticles, Vec3f gridBoundMin, float invCellSize,
            Vec3i numNodesPerDim, float gridVolume, const float4* particlePositionMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float4* nodeMomentumVelocityMass
        ) {
            id<MTLComputePipelineState> state = getPipelineState("standardMpmComputeInitialGridMassKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, particlePositionMass, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);

            [encoder setBytes:&numParticles length:sizeof(int) atIndex:5];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:6];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:7];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:8];
            [encoder setBytes:&gridVolume length:sizeof(float) atIndex:9];

            // 1D Thread Dispatch
            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numParticles + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        void dispatchStandardMpmComputeInitialVolume(
            cudaStream_t stream, int numParticles, Vec3f gridBoundMin, float invCellSize,
            Vec3i numNodesPerDim, float gridVolume, const float4* particlePositionMass,
            const float4* nodeMomentumVelocityMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float* particleInitialVolume
        ) {
            id<MTLComputePipelineState> state = getPipelineState("standardMpmComputeInitialVolumeKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, particlePositionMass, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);
            bindPointerToEncoder(encoder, particleInitialVolume, 5);

            [encoder setBytes:&numParticles length:sizeof(int) atIndex:6];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:7];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:8];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:9];
            [encoder setBytes:&gridVolume length:sizeof(float) atIndex:10];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numParticles + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

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
        ) {
            id<MTLComputePipelineState> state = getPipelineState("standardMpmParticleToGridKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, nodeForce, 1);
            bindPointerToEncoder(encoder, particlePositionMass, 2);
            bindPointerToEncoder(encoder, particleVelocity, 3);
            bindPointerToEncoder(encoder, particleInitialVolume, 4);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 5);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 6);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 7);
            bindPointerToEncoder(encoder, particleMaterialProperties0, 8);
            bindPointerToEncoder(encoder, particleMaterialTypes, 9);

            [encoder setBytes:&numParticles length:sizeof(int) atIndex:10];
            [encoder setBytes:&gravity length:sizeof(Vec3f) atIndex:11];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:12];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:13];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:14];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numParticles + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        void dispatchStandardMpmUpdateGrid(
            cudaStream_t stream, bool useEffectiveMass, int numNodes, Vec3f gridBoundMin,
            float cellSize, Vec3i numNodesPerDim, float integrationStepSize,
            const Vec3f* nodeForce, int numShapes, const int* shapeIds,
            const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData, float4* nodeMomentumVelocityMass
        ) {
            id<MTLComputePipelineState> state = getPipelineState("standardMpmUpdateGridKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, nodeForce, 1);
            bindPointerToEncoder(encoder, shapeIds, 2);

            // ShapeData
            bindPointerToEncoder(encoder, shapeData.type, 3);
            bindPointerToEncoder(encoder, shapeData.geometryIdx, 4);
            bindPointerToEncoder(encoder, shapeData.position, 5);
            bindPointerToEncoder(encoder, shapeData.rotation, 6);
            bindPointerToEncoder(encoder, shapeData.invScale, 7);
            bindPointerToEncoder(encoder, shapeData.params0, 8);

            // GeometryData
            bindPointerToEncoder(encoder, geometryData.type, 9);
            bindPointerToEncoder(encoder, geometryData.params0, 10);

            // GeometrySdfData
            bindPointerToEncoder(encoder, geometrySdfData.dimemsion, 11);
            bindPointerToEncoder(encoder, geometrySdfData.lowerBoundCellSize, 12);
            bindPointerToEncoder(encoder, geometrySdfData.gradientSignedDistance, 13);

            [encoder setBytes:&numNodes length:sizeof(int) atIndex:14];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:15];
            [encoder setBytes:&cellSize length:sizeof(float) atIndex:16];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:17];
            [encoder setBytes:&integrationStepSize length:sizeof(float) atIndex:18];
            [encoder setBytes:&numShapes length:sizeof(int) atIndex:19];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numNodes + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

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
        ) {
            id<MTLComputePipelineState> state = getPipelineState("standardMpmGridToParticleKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, particlePositionMass, 0);
            bindPointerToEncoder(encoder, particleVelocity, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);
            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 5);
            bindPointerToEncoder(encoder, shapeIds, 6);

            // ShapeData
            bindPointerToEncoder(encoder, shapeData.type, 7);
            bindPointerToEncoder(encoder, shapeData.geometryIdx, 8);
            bindPointerToEncoder(encoder, shapeData.position, 9);
            bindPointerToEncoder(encoder, shapeData.rotation, 10);
            bindPointerToEncoder(encoder, shapeData.invScale, 11);
            bindPointerToEncoder(encoder, shapeData.params0, 12);

            // GeometryData
            bindPointerToEncoder(encoder, geometryData.type, 13);
            bindPointerToEncoder(encoder, geometryData.params0, 14);

            // GeometrySdfData
            bindPointerToEncoder(encoder, geometrySdfData.dimemsion, 15);
            bindPointerToEncoder(encoder, geometrySdfData.lowerBoundCellSize, 16);
            bindPointerToEncoder(encoder, geometrySdfData.gradientSignedDistance, 17);

            [encoder setBytes:&numParticles length:sizeof(int) atIndex:18];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:19];
            [encoder setBytes:&gridBoundMax length:sizeof(Vec3f) atIndex:20];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:21];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:22];
            [encoder setBytes:&integrationStepSize length:sizeof(float) atIndex:23];
            [encoder setBytes:&numShapes length:sizeof(int) atIndex:24];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numParticles + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        // -------------------------------------------------------------
        // MLS-MPM Kernels Dispatchers
        // -------------------------------------------------------------
        void dispatchMlsMpmComputeInitialGridMass(
            cudaStream_t stream, int numParticlesToCompute, const unsigned int* indices,
            Vec3f gridBoundMin, float invCellSize, Vec3i numNodesPerDim, float cellVolume,
            const float4* particlePositionMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float4* nodeMomentumVelocityMass
        ) {
            id<MTLComputePipelineState> state = getPipelineState("mlsMpmComputeInitialGridMassKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, particlePositionMass, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);
            bindPointerToEncoder(encoder, indices, 5);

            [encoder setBytes:&numParticlesToCompute length:sizeof(int) atIndex:6];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:7];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:8];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:9];
            [encoder setBytes:&cellVolume length:sizeof(float) atIndex:10];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numParticlesToCompute + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        void dispatchMlsMpmComputeInitialVolume(
            cudaStream_t stream, int numParticlesToCompute, const unsigned int* indices,
            Vec3f gridBoundMin, float invCellSize, Vec3i numNodesPerDim, float cellVolume,
            const float4* particlePositionMass, const float4* nodeMomentumVelocityMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float* particleInitialVolume
        ) {
            id<MTLComputePipelineState> state = getPipelineState("mlsMpmComputeInitialVolumeKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, particlePositionMass, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);
            bindPointerToEncoder(encoder, particleInitialVolume, 5);
            bindPointerToEncoder(encoder, indices, 6);

            [encoder setBytes:&numParticlesToCompute length:sizeof(int) atIndex:7];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:8];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:9];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:10];
            [encoder setBytes:&cellVolume length:sizeof(float) atIndex:11];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numParticlesToCompute + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

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
        ) {
            id<MTLComputePipelineState> state = getPipelineState("mlsMpmParticleToGridKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, particlePositionMass, 1);
            bindPointerToEncoder(encoder, particleInitialVolume, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 4);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 5);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn0, 6);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn1, 7);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn2, 8);
            bindPointerToEncoder(encoder, particleMaterialProperties0, 9);
            bindPointerToEncoder(encoder, particleMaterialTypes, 10);
            bindPointerToEncoder(encoder, particleVelocity, 11);
            bindPointerToEncoder(encoder, activeMask, 12);

            [encoder setBytes:&numActiveParticles length:sizeof(int) atIndex:13];
            [encoder setBytes:&integrationStepSize length:sizeof(float) atIndex:14];
            [encoder setBytes:&gravity length:sizeof(Vec3f) atIndex:15];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:16];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:17];
            [encoder setBytes:&cellSize length:sizeof(float) atIndex:18];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:19];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numActiveParticles + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        void dispatchMlsMpmUpdateGrid(
            cudaStream_t stream, bool useEffectiveMass, float integrationStepSize, int numNodes,
            Vec3f gridBoundMin, float cellSize, Vec3i numNodesPerDim, int numShapes,
            const int* shapeIds, const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData, float4* nodeMomentumVelocityMass
        ) {
            id<MTLComputePipelineState> state = getPipelineState("mlsMpmUpdateGridKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, shapeIds, 1);

            // ShapeData
            bindPointerToEncoder(encoder, shapeData.type, 2);
            bindPointerToEncoder(encoder, shapeData.geometryIdx, 3);
            bindPointerToEncoder(encoder, shapeData.position, 4);
            bindPointerToEncoder(encoder, shapeData.rotation, 5);
            bindPointerToEncoder(encoder, shapeData.invScale, 6);
            bindPointerToEncoder(encoder, shapeData.params0, 7);

            // GeometryData
            bindPointerToEncoder(encoder, geometryData.type, 8);
            bindPointerToEncoder(encoder, geometryData.params0, 9);

            // GeometrySdfData
            bindPointerToEncoder(encoder, geometrySdfData.dimemsion, 10);
            bindPointerToEncoder(encoder, geometrySdfData.lowerBoundCellSize, 11);
            bindPointerToEncoder(encoder, geometrySdfData.gradientSignedDistance, 12);

            [encoder setBytes:&numNodes length:sizeof(int) atIndex:13];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:14];
            [encoder setBytes:&cellSize length:sizeof(float) atIndex:15];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:16];
            [encoder setBytes:&integrationStepSize length:sizeof(float) atIndex:17];
            [encoder setBytes:&numShapes length:sizeof(int) atIndex:18];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numNodes + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

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
        ) {
            id<MTLComputePipelineState> state = getPipelineState("mlsMpmGridToParticleKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, particlePositionMass, 0);
            bindPointerToEncoder(encoder, particleVelocity, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn0, 5);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn1, 6);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn2, 7);
            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 8);
            bindPointerToEncoder(encoder, shapeIds, 9);

            // ShapeData
            bindPointerToEncoder(encoder, shapeData.type, 10);
            bindPointerToEncoder(encoder, shapeData.geometryIdx, 11);
            bindPointerToEncoder(encoder, shapeData.position, 12);
            bindPointerToEncoder(encoder, shapeData.rotation, 13);
            bindPointerToEncoder(encoder, shapeData.invScale, 14);
            bindPointerToEncoder(encoder, shapeData.params0, 15);

            // GeometryData
            bindPointerToEncoder(encoder, geometryData.type, 16);
            bindPointerToEncoder(encoder, geometryData.params0, 17);

            // GeometrySdfData
            bindPointerToEncoder(encoder, geometrySdfData.dimemsion, 18);
            bindPointerToEncoder(encoder, geometrySdfData.lowerBoundCellSize, 19);
            bindPointerToEncoder(encoder, geometrySdfData.gradientSignedDistance, 20);
            bindPointerToEncoder(encoder, activeMask, 21);

            [encoder setBytes:&numActiveParticles length:sizeof(int) atIndex:22];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:23];
            [encoder setBytes:&gridBoundMax length:sizeof(Vec3f) atIndex:24];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:25];
            [encoder setBytes:&cellSize length:sizeof(float) atIndex:26];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:27];
            [encoder setBytes:&integrationStepSize length:sizeof(float) atIndex:28];
            [encoder setBytes:&numShapes length:sizeof(int) atIndex:29];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numActiveParticles + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        // -------------------------------------------------------------
        // Position-Based MPM (PB-MPM) Kernels Dispatchers
        // -------------------------------------------------------------
        void dispatchPbMpmComputeInitialGridMass(
            cudaStream_t stream, int numParticlesToCompute, const unsigned int* indices,
            Vec3f gridBoundMin, float invCellSize, Vec3i numNodesPerDim, float cellVolume,
            const float4* particlePositionMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float4* nodeMomentumVelocityMass
        ) {
            id<MTLComputePipelineState> state = getPipelineState("pbMpmComputeInitialGridMassKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, particlePositionMass, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);
            bindPointerToEncoder(encoder, indices, 5);

            [encoder setBytes:&numParticlesToCompute length:sizeof(int) atIndex:6];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:7];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:8];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:9];
            [encoder setBytes:&cellVolume length:sizeof(float) atIndex:10];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numParticlesToCompute + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        void dispatchPbMpmComputeInitialVolume(
            cudaStream_t stream, int numParticlesToCompute, const unsigned int* indices,
            Vec3f gridBoundMin, float invCellSize, Vec3i numNodesPerDim, float cellVolume,
            const float4* particlePositionMass, const float4* nodeMomentumVelocityMass,
            float4* particleGradientDeformationTensorColumn0,
            float4* particleGradientDeformationTensorColumn1,
            float4* particleGradientDeformationTensorColumn2,
            float* particleInitialVolume
        ) {
            id<MTLComputePipelineState> state = getPipelineState("pbMpmComputeInitialVolumeKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, particlePositionMass, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);
            bindPointerToEncoder(encoder, particleInitialVolume, 5);
            bindPointerToEncoder(encoder, indices, 6);

            [encoder setBytes:&numParticlesToCompute length:sizeof(int) atIndex:7];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:8];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:9];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:10];
            [encoder setBytes:&cellVolume length:sizeof(float) atIndex:11];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numParticlesToCompute + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

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
        ) {
            id<MTLComputePipelineState> state = getPipelineState("pbMpmParticleToGridKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, particlePositionMass, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn0, 5);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn1, 6);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn2, 7);
            bindPointerToEncoder(encoder, particleMaterialProperties0, 8);
            bindPointerToEncoder(encoder, particleMaterialTypes, 9);
            bindPointerToEncoder(encoder, particleVelocity, 10);
            bindPointerToEncoder(encoder, activeMask, 11);

            [encoder setBytes:&numActiveParticles length:sizeof(int) atIndex:12];
            [encoder setBytes:&integrationStepSize length:sizeof(float) atIndex:13];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:14];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:15];
            [encoder setBytes:&cellSize length:sizeof(float) atIndex:16];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:17];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numActiveParticles + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        void dispatchPbMpmUpdateGrid(
            cudaStream_t stream, bool useEffectiveMass, float integrationStepSize, int numNodes,
            Vec3f gridBoundMin, float cellSize, Vec3i numNodesPerDim, int numShapes,
            const int* shapeIds, const ShapeData& shapeData, const GeometryData& geometryData,
            const GeometrySdfData& geometrySdfData, float4* nodeMomentumVelocityMass
        ) {
            id<MTLComputePipelineState> state = getPipelineState("pbMpmUpdateGridKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 0);
            bindPointerToEncoder(encoder, shapeIds, 1);

            // ShapeData
            bindPointerToEncoder(encoder, shapeData.type, 2);
            bindPointerToEncoder(encoder, shapeData.geometryIdx, 3);
            bindPointerToEncoder(encoder, shapeData.position, 4);
            bindPointerToEncoder(encoder, shapeData.rotation, 5);
            bindPointerToEncoder(encoder, shapeData.invScale, 6);
            bindPointerToEncoder(encoder, shapeData.params0, 7);

            // GeometryData
            bindPointerToEncoder(encoder, geometryData.type, 8);
            bindPointerToEncoder(encoder, geometryData.params0, 9);

            // GeometrySdfData
            bindPointerToEncoder(encoder, geometrySdfData.dimemsion, 10);
            bindPointerToEncoder(encoder, geometrySdfData.lowerBoundCellSize, 11);
            bindPointerToEncoder(encoder, geometrySdfData.gradientSignedDistance, 12);

            [encoder setBytes:&numNodes length:sizeof(int) atIndex:13];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:14];
            [encoder setBytes:&cellSize length:sizeof(float) atIndex:15];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:16];
            [encoder setBytes:&integrationStepSize length:sizeof(float) atIndex:17];
            [encoder setBytes:&numShapes length:sizeof(int) atIndex:18];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numNodes + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

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
        ) {
            id<MTLComputePipelineState> state = getPipelineState("pbMpmGridToParticleKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, particlePositionMass, 0);
            bindPointerToEncoder(encoder, particleVelocity, 1);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn0, 2);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn1, 3);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn2, 4);
            bindPointerToEncoder(encoder, nodeMomentumVelocityMass, 5);
            bindPointerToEncoder(encoder, activeMask, 6);

            [encoder setBytes:&numActiveParticles length:sizeof(int) atIndex:7];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:8];
            [encoder setBytes:&gridBoundMax length:sizeof(Vec3f) atIndex:9];
            [encoder setBytes:&invCellSize length:sizeof(float) atIndex:10];
            [encoder setBytes:&cellSize length:sizeof(float) atIndex:11];
            [encoder setBytes:&cellVolume length:sizeof(float) atIndex:12];
            [encoder setBytes:&numNodesPerDim length:sizeof(Vec3i) atIndex:13];
            [encoder setBytes:&integrationStepSize length:sizeof(float) atIndex:14];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numActiveParticles + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

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
        ) {
            id<MTLComputePipelineState> state = getPipelineState("pbMpmIntegrateParticleKernel");
            id<MTLCommandBuffer> buf = stream->currentCommandBuffer;
            id<MTLComputeCommandEncoder> encoder = [buf computeCommandEncoder];
            [encoder setComputePipelineState:state];

            bindPointerToEncoder(encoder, particlePositionMass, 0);
            bindPointerToEncoder(encoder, particleVelocity, 1);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn0, 2);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn1, 3);
            bindPointerToEncoder(encoder, particleGradientDeformationTensorColumn2, 4);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn0, 5);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn1, 6);
            bindPointerToEncoder(encoder, particleAffineMomentumColumn2, 7);
            bindPointerToEncoder(encoder, particleMaterialTypes, 8);
            bindPointerToEncoder(encoder, shapeIds, 9);

            // ShapeData
            bindPointerToEncoder(encoder, shapeData.type, 10);
            bindPointerToEncoder(encoder, shapeData.geometryIdx, 11);
            bindPointerToEncoder(encoder, shapeData.position, 12);
            bindPointerToEncoder(encoder, shapeData.rotation, 13);
            bindPointerToEncoder(encoder, shapeData.invScale, 14);
            bindPointerToEncoder(encoder, shapeData.params0, 15);

            // GeometryData
            bindPointerToEncoder(encoder, geometryData.type, 16);
            bindPointerToEncoder(encoder, geometryData.params0, 17);

            // GeometrySdfData
            bindPointerToEncoder(encoder, geometrySdfData.dimemsion, 18);
            bindPointerToEncoder(encoder, geometrySdfData.lowerBoundCellSize, 19);
            bindPointerToEncoder(encoder, geometrySdfData.gradientSignedDistance, 20);
            bindPointerToEncoder(encoder, activeMask, 21);

            [encoder setBytes:&numActiveParticles length:sizeof(int) atIndex:22];
            [encoder setBytes:&gridBoundMin length:sizeof(Vec3f) atIndex:23];
            [encoder setBytes:&gridBoundMax length:sizeof(Vec3f) atIndex:24];
            [encoder setBytes:&integrationStepSize length:sizeof(float) atIndex:25];
            [encoder setBytes:&gravity length:sizeof(Vec3f) atIndex:26];
            [encoder setBytes:&numShapes length:sizeof(int) atIndex:27];

            NSUInteger threadExecutionWidth = [state threadExecutionWidth];
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadExecutionWidth, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((numActiveParticles + threadExecutionWidth - 1) / threadExecutionWidth, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

    }
}

#endif // CR_USE_METAL
