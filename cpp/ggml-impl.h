#pragma once

// GGML internal header

#include "ggml.h"

#include <assert.h>
#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// required for mmap as gguf only guarantees 32-byte alignment
#define TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef __cplusplus
#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif
#endif

static inline int lm_ggml_up32(int n) {
    return (n + 31) & ~31;
}

//static inline int lm_ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

static inline int lm_ggml_up(int n, int m) {
    // assert m is a power of 2
    LM_GGML_ASSERT((m & (m - 1)) == 0);
    return (n + m - 1) & ~(m - 1);
}

//
// logging
//

LM_GGML_ATTRIBUTE_FORMAT(2, 3)
void lm_ggml_log_internal        (enum lm_ggml_log_level level, const char * format, ...);
void lm_ggml_log_callback_default(enum lm_ggml_log_level level, const char * text, void * user_data);

#define LM_GGML_LOG(...)       lm_ggml_log_internal(LM_GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define LM_GGML_LOG_INFO(...)  lm_ggml_log_internal(LM_GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LM_GGML_LOG_WARN(...)  lm_ggml_log_internal(LM_GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LM_GGML_LOG_ERROR(...) lm_ggml_log_internal(LM_GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LM_GGML_LOG_DEBUG(...) lm_ggml_log_internal(LM_GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LM_GGML_LOG_CONT(...)  lm_ggml_log_internal(LM_GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define LM_GGML_DEBUG 0

#if (LM_GGML_DEBUG >= 1)
#define LM_GGML_PRINT_DEBUG(...) LM_GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define LM_GGML_PRINT_DEBUG(...)
#endif

#if (LM_GGML_DEBUG >= 5)
#define LM_GGML_PRINT_DEBUG_5(...) LM_GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define LM_GGML_PRINT_DEBUG_5(...)
#endif

#if (LM_GGML_DEBUG >= 10)
#define LM_GGML_PRINT_DEBUG_10(...) LM_GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define LM_GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

static void lm_ggml_set_op_params(struct lm_ggml_tensor * tensor, const void * params, size_t params_size) {
    LM_GGML_ASSERT(tensor != NULL); // silence -Warray-bounds warnings
    assert(params_size <= LM_GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

static int32_t lm_ggml_get_op_params_i32(const struct lm_ggml_tensor * tensor, uint32_t i) {
    assert(i < LM_GGML_MAX_OP_PARAMS / sizeof(int32_t));
    return ((const int32_t *)(tensor->op_params))[i];
}

static float lm_ggml_get_op_params_f32(const struct lm_ggml_tensor * tensor, uint32_t i) {
    assert(i < LM_GGML_MAX_OP_PARAMS / sizeof(float));
    return ((const float *)(tensor->op_params))[i];
}

static void lm_ggml_set_op_params_i32(struct lm_ggml_tensor * tensor, uint32_t i, int32_t value) {
    assert(i < LM_GGML_MAX_OP_PARAMS / sizeof(int32_t));
    ((int32_t *)(tensor->op_params))[i] = value;
}

static void lm_ggml_set_op_params_f32(struct lm_ggml_tensor * tensor, uint32_t i, float value) {
    assert(i < LM_GGML_MAX_OP_PARAMS / sizeof(float));
    ((float *)(tensor->op_params))[i] = value;
}

struct lm_ggml_map_custom1_op_params {
    lm_ggml_custom1_op_t  fun;
    int                n_tasks;
    void             * userdata;
};


struct lm_ggml_map_custom2_op_params {
    lm_ggml_custom2_op_t   fun;
    int                 n_tasks;
    void              * userdata;
};


struct lm_ggml_map_custom3_op_params {
    lm_ggml_custom3_op_t fun;
    int n_tasks;
    void * userdata;
};

// bitset

typedef uint32_t lm_ggml_bitset_t;

static_assert(sizeof(lm_ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(lm_ggml_bitset_t)*8)
#define BITSET_MASK (sizeof(lm_ggml_bitset_t)*8 - 1)

static size_t lm_ggml_bitset_size(size_t n) {
    return (n + BITSET_MASK) >> BITSET_SHR;
}

static inline bool lm_ggml_bitset_get(const lm_ggml_bitset_t * bitset, size_t i) {
    return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

static inline void lm_ggml_bitset_set(lm_ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

static inline void lm_ggml_bitset_clear(lm_ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define LM_GGML_HASHSET_FULL ((size_t)-1)
#define LM_GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

struct lm_ggml_hash_set {
    size_t size;
    lm_ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    struct lm_ggml_tensor ** keys; // actual tensors in the set, keys[i] is only defined if lm_ggml_bitset_get(used, i)
};

struct lm_ggml_hash_set lm_ggml_hash_set_new(size_t size);
void                 lm_ggml_hash_set_free(struct lm_ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t lm_ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
void lm_ggml_hash_set_reset(struct lm_ggml_hash_set * hash_set);

// returns true if key is in the hash set
static bool lm_ggml_hash_contains(const struct lm_ggml_hash_set * hash_set, struct lm_ggml_tensor * key);

// returns LM_GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
static size_t lm_ggml_hash_find(const struct lm_ggml_hash_set * hash_set, struct lm_ggml_tensor * key);

// returns LM_GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
static size_t lm_ggml_hash_insert(struct lm_ggml_hash_set * hash_set, struct lm_ggml_tensor * key);

// return index, asserts if table is full
static size_t lm_ggml_hash_find_or_insert(struct lm_ggml_hash_set * hash_set, struct lm_ggml_tensor * key);

// hash function for lm_ggml_tensor
static inline size_t lm_ggml_hash(const struct lm_ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    return (size_t)(uintptr_t)p >> 4;
}

static size_t lm_ggml_hash_find(const struct lm_ggml_hash_set * hash_set, struct lm_ggml_tensor * key) {
    size_t h = lm_ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    while (lm_ggml_bitset_get(hash_set->used, i) && hash_set->keys[i] != key) {
        i = (i + 1) % hash_set->size;
        if (i == h) {
            // visited all hash table entries -> not found
            return LM_GGML_HASHSET_FULL;
        }
    }
    return i;
}

static bool lm_ggml_hash_contains(const struct lm_ggml_hash_set * hash_set, struct lm_ggml_tensor * key) {
    size_t i = lm_ggml_hash_find(hash_set, key);
    return i != LM_GGML_HASHSET_FULL && lm_ggml_bitset_get(hash_set->used, i);
}

static size_t lm_ggml_hash_insert(struct lm_ggml_hash_set * hash_set, struct lm_ggml_tensor * key) {
    size_t h = lm_ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    do {
        if (!lm_ggml_bitset_get(hash_set->used, i)) {
            lm_ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
        if (hash_set->keys[i] == key) {
            return LM_GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    // visited all hash table entries -> not found
    LM_GGML_ABORT("fatal error");
}

static size_t lm_ggml_hash_find_or_insert(struct lm_ggml_hash_set * hash_set, struct lm_ggml_tensor * key) {
    size_t h = lm_ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    do {
        if (!lm_ggml_bitset_get(hash_set->used, i)) {
            lm_ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
        if (hash_set->keys[i] == key) {
            return i;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    // visited all hash table entries -> not found
    LM_GGML_ABORT("fatal error");
}

// computation graph

enum lm_ggml_cgraph_eval_order {
    LM_GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
    LM_GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
    LM_GGML_CGRAPH_EVAL_ORDER_COUNT
};

struct lm_ggml_cgraph {
    int size;
    int n_nodes;
    int n_leafs;

    struct lm_ggml_tensor ** nodes;
    struct lm_ggml_tensor ** grads;
    struct lm_ggml_tensor ** leafs;

    struct lm_ggml_hash_set visited_hash_set;

    enum lm_ggml_cgraph_eval_order order;
};

struct lm_ggml_cgraph lm_ggml_graph_view(struct lm_ggml_cgraph * cgraph, int i0, int i1);

// Memory allocation

void * lm_ggml_aligned_malloc(size_t size);
void lm_ggml_aligned_free(void * ptr, size_t size);

// TODO: move to threading file
void lm_ggml_critical_section_start(void);
void lm_ggml_critical_section_end(void);

#ifdef __cplusplus
}
#endif
