#pragma once

#include "ggml.h"

// GGML internal header

#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h> // memcpy
#include <math.h>   // fabsf

#ifdef __cplusplus
extern "C" {
#endif

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#ifndef __SSE3__
#define __SSE3__
#endif
#endif

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// 16-bit float
// on Arm, we use __fp16
// on x86, we use uint16_t
#if defined(__ARM_NEON) && !defined(_MSC_VER)

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>

#define LM_GGML_COMPUTE_FP16_TO_FP32(x) ((float) (x))
#define LM_GGML_COMPUTE_FP32_TO_FP16(x) (x)

#define LM_GGML_FP16_TO_FP32(x) ((float) (x))
#define LM_GGML_FP32_TO_FP16(x) (x)

#else

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#else
#ifdef __POWER9_VECTOR__
#include <altivec.h>
#undef bool
#define bool _Bool
#else
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__) || defined(__SSE3__)
#if !defined(__riscv)
#include <immintrin.h>
#endif
#endif
#endif
#endif
#endif

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

#ifdef __F16C__

#ifdef _MSC_VER
#define LM_GGML_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define LM_GGML_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#else
#define LM_GGML_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define LM_GGML_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#endif

#elif defined(__POWER9_VECTOR__)

#define LM_GGML_COMPUTE_FP16_TO_FP32(x) lm_ggml_compute_fp16_to_fp32(x)
#define LM_GGML_COMPUTE_FP32_TO_FP16(x) lm_ggml_compute_fp32_to_fp16(x)
/* the inline asm below is about 12% faster than the lookup method */
#define LM_GGML_FP16_TO_FP32(x) LM_GGML_COMPUTE_FP16_TO_FP32(x)
#define LM_GGML_FP32_TO_FP16(x) LM_GGML_COMPUTE_FP32_TO_FP16(x)

static inline float lm_ggml_compute_fp16_to_fp32(lm_ggml_fp16_t h) {
    register float f;
    register double d;
    __asm__(
        "mtfprd %0,%2\n"
        "xscvhpdp %0,%0\n"
        "frsp %1,%0\n" :
        /* temp */ "=d"(d),
        /* out */  "=f"(f):
        /* in */   "r"(h));
    return f;
}

static inline lm_ggml_fp16_t lm_ggml_compute_fp32_to_fp16(float f) {
    register double d;
    register lm_ggml_fp16_t r;
    __asm__( /* xscvdphp can work on double or single precision */
        "xscvdphp %0,%2\n"
        "mffprd %1,%0\n" :
        /* temp */ "=d"(d),
        /* out */  "=r"(r):
        /* in */   "f"(f));
    return r;
}

#else

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float lm_ggml_compute_fp16_to_fp32(lm_ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline lm_ggml_fp16_t lm_ggml_compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define LM_GGML_COMPUTE_FP16_TO_FP32(x) lm_ggml_compute_fp16_to_fp32(x)
#define LM_GGML_COMPUTE_FP32_TO_FP16(x) lm_ggml_compute_fp32_to_fp16(x)

#endif // __F16C__

#endif // __ARM_NEON

// precomputed f32 table for f16 (256 KB)
// defined in ggml.c, initialized in lm_ggml_init()
extern float lm_ggml_table_f32_f16[1 << 16];

// On ARM NEON, it's quicker to directly convert x -> x instead of calling into lm_ggml_lookup_fp16_to_fp32,
// so we define LM_GGML_FP16_TO_FP32 and LM_GGML_FP32_TO_FP16 elsewhere for NEON.
// This is also true for POWER9.
#if !defined(LM_GGML_FP16_TO_FP32) || !defined(LM_GGML_FP32_TO_FP16)

inline static float lm_ggml_lookup_fp16_to_fp32(lm_ggml_fp16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return lm_ggml_table_f32_f16[s];
}

#define LM_GGML_FP16_TO_FP32(x) lm_ggml_lookup_fp16_to_fp32(x)
#define LM_GGML_FP32_TO_FP16(x) LM_GGML_COMPUTE_FP32_TO_FP16(x)

#endif

    // TODO: backend v2 PR

#ifdef __cplusplus
}
#endif