#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef LM_GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef LM_GGML_USE_LLAMAFILE
#endif

#ifdef LM_GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define LM_GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define LM_GGML_CACHE_ALIGN __attribute__((aligned(LM_GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define LM_GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define LM_GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED LM_GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float lm_ggml_table_f32_f16[1 << 16];

#if defined(__ARM_ARCH)
struct lm_ggml_arm_arch_features_type {
    int sve_cnt;
} lm_ggml_arm_arch_features = { 0 };
#endif


#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define LM_GGML_CACHE_ALIGN __declspec(align(LM_GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t lm_ggml_thread_t;

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct lm_ggml_type_traits_cpu type_traits_cpu[LM_GGML_TYPE_COUNT] = {
    [LM_GGML_TYPE_F32] = {
        .vec_dot                  = (lm_ggml_vec_dot_t) lm_ggml_vec_dot_f32,
        .vec_dot_type             = LM_GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_F16] = {
        .from_float               = (lm_ggml_from_float_t) lm_ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (lm_ggml_vec_dot_t) lm_ggml_vec_dot_f16,
        .vec_dot_type             = LM_GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = lm_ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [LM_GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = lm_ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = LM_GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [LM_GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = lm_ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = lm_ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = LM_GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = lm_ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [LM_GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = LM_GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = lm_ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = lm_ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = lm_ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [LM_GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = lm_ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = lm_ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [LM_GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = lm_ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = lm_ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in lm_ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = lm_ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = lm_ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = lm_ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = lm_ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = lm_ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = lm_ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = lm_ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [LM_GGML_TYPE_BF16] = {
        .from_float               = (lm_ggml_from_float_t) lm_ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (lm_ggml_vec_dot_t) lm_ggml_vec_dot_bf16,
        .vec_dot_type             = LM_GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = lm_ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = lm_ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
};

const struct lm_ggml_type_traits_cpu * lm_ggml_get_type_traits_cpu(enum lm_ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          lm_ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE lm_ggml_cond_t;
typedef SRWLOCK            lm_ggml_mutex_t;

#define lm_ggml_mutex_init(m)   InitializeSRWLock(m)
#define lm_ggml_mutex_destroy(m)
#define lm_ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define lm_ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define lm_ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define lm_ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define lm_ggml_cond_init(c)    InitializeConditionVariable(c)
#define lm_ggml_cond_destroy(c)
#define lm_ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define lm_ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define lm_ggml_thread_create pthread_create
#define lm_ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     lm_ggml_cond_t;
typedef pthread_mutex_t    lm_ggml_mutex_t;

#define lm_ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define lm_ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define lm_ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define lm_ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define lm_ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define lm_ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define lm_ggml_lock_init(x)    UNUSED(x)
#define lm_ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define lm_ggml_lock_lock(x)    _mm_pause()
#else
#define lm_ggml_lock_lock(x)    UNUSED(x)
#endif
#define lm_ggml_lock_unlock(x)  UNUSED(x)

#define LM_GGML_LOCK_INITIALIZER 0
#define lm_ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define lm_ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define lm_ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define lm_ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define lm_ggml_thread_create pthread_create
#define lm_ggml_thread_join   pthread_join

#endif

// Threadpool def
struct lm_ggml_threadpool {
    lm_ggml_mutex_t mutex;       // mutex for cond.var
    lm_ggml_cond_t  cond;        // cond.var for waiting for new work

    struct lm_ggml_cgraph * cgraph;
    struct lm_ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // incremented when there is work to be done (i.e each graph)
    atomic_int LM_GGML_CACHE_ALIGN n_barrier;
    atomic_int LM_GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int LM_GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int abort;         // Used for aborting processing of a graph

    struct lm_ggml_compute_state * workers;   // per thread state
    int          n_threads_max; // number of threads in the pool
    atomic_int   n_threads_cur; // number of threads used in the current graph

    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum lm_ggml_status ec;
};

// Per-thread state
struct lm_ggml_compute_state {
#ifndef LM_GGML_USE_OPENMP
    lm_ggml_thread_t thrd;
    bool cpumask[LM_GGML_MAX_N_THREADS];
    int  last_graph;
    bool pending;
#endif
    struct lm_ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void lm_ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void lm_ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#else
static inline void lm_ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define LM_GGML_NUMA_MAX_NODES 8
#define LM_GGML_NUMA_MAX_CPUS 512

struct lm_ggml_numa_node {
    uint32_t cpus[LM_GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct lm_ggml_numa_nodes {
    enum lm_ggml_numa_strategy numa_strategy;
    struct lm_ggml_numa_node nodes[LM_GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct lm_ggml_state {
    struct lm_ggml_numa_nodes numa;
};

static struct lm_ggml_state g_state = {0};

void lm_ggml_barrier(struct lm_ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed);
    if (n_threads == 1) {
        return;
    }

#ifdef LM_GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (fill seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        lm_ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef LM_GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void lm_ggml_threadpool_chunk_set(struct lm_ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int lm_ggml_threadpool_chunk_add(struct lm_ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t lm_ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t lm_ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void lm_ggml_numa_init(enum lm_ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "lm_ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    LM_GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = lm_ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < LM_GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        LM_GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < LM_GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        LM_GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    LM_GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    LM_GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct lm_ggml_numa_node * node = &g_state.numa.nodes[n];
        LM_GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            LM_GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                LM_GGML_PRINT_DEBUG(" %u", c);
            }
        }
        LM_GGML_PRINT_DEBUG("\n");
    }

    if (lm_ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                LM_GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool lm_ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#endif

static void lm_ggml_init_arm_arch_features(void) {
#if defined(__linux__) && defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
    lm_ggml_arm_arch_features.sve_cnt = PR_SVE_VL_LEN_MASK & prctl(PR_SVE_GET_VL);
#endif
}

#endif // __ARM_ARCH

struct lm_ggml_tensor * lm_ggml_new_i32(struct lm_ggml_context * ctx, int32_t value) {
    LM_GGML_ASSERT(!lm_ggml_get_no_alloc(ctx));

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, 1);

    lm_ggml_set_i32(result, value);

    return result;
}

struct lm_ggml_tensor * lm_ggml_new_f32(struct lm_ggml_context * ctx, float value) {
    LM_GGML_ASSERT(!lm_ggml_get_no_alloc(ctx));

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, 1);

    lm_ggml_set_f32(result, value);

    return result;
}

struct lm_ggml_tensor * lm_ggml_set_i32 (struct lm_ggml_tensor * tensor, int32_t value) {
    const int n     = lm_ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_f16(nc, (lm_ggml_fp16_t *)(data + i*n1), LM_GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case LM_GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_bf16(nc, (lm_ggml_bf16_t *)(data + i*n1), LM_GGML_FP32_TO_BF16(value));
                }
            } break;
        case LM_GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct lm_ggml_tensor * lm_ggml_set_f32(struct lm_ggml_tensor * tensor, float value) {
    const int n     = lm_ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_f16(nc, (lm_ggml_fp16_t *)(data + i*n1), LM_GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case LM_GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(lm_ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_bf16(nc, (lm_ggml_bf16_t *)(data + i*n1), LM_GGML_FP32_TO_BF16(value));
                }
            } break;
        case LM_GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t lm_ggml_get_i32_1d(const struct lm_ggml_tensor * tensor, int i) {
    if (!lm_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        lm_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return lm_ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_I16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_I32:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_F16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                return LM_GGML_CPU_FP16_TO_FP32(((lm_ggml_fp16_t *)(tensor->data))[i]);
            }
        case LM_GGML_TYPE_BF16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(lm_ggml_bf16_t));
                return LM_GGML_BF16_TO_FP32(((lm_ggml_bf16_t *)(tensor->data))[i]);
            }
        case LM_GGML_TYPE_F32:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

void lm_ggml_set_i32_1d(const struct lm_ggml_tensor * tensor, int i, int32_t value) {
    if (!lm_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        lm_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        lm_ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_I16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_I32:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_F16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                ((lm_ggml_fp16_t *)(tensor->data))[i] = LM_GGML_CPU_FP32_TO_FP16(value);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(lm_ggml_bf16_t));
                ((lm_ggml_bf16_t *)(tensor->data))[i] = LM_GGML_FP32_TO_BF16(value);
            } break;
        case LM_GGML_TYPE_F32:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

int32_t lm_ggml_get_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case LM_GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case LM_GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case LM_GGML_TYPE_F16:
            return LM_GGML_CPU_FP16_TO_FP32(((lm_ggml_fp16_t *) data)[0]);
        case LM_GGML_TYPE_BF16:
            return LM_GGML_BF16_TO_FP32(((lm_ggml_bf16_t *) data)[0]);
        case LM_GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            LM_GGML_ABORT("fatal error");
    }
}

void lm_ggml_set_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_F16:
            {
                ((lm_ggml_fp16_t *)(data))[0] = LM_GGML_CPU_FP32_TO_FP16(value);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                ((lm_ggml_bf16_t *)(data))[0] = LM_GGML_FP32_TO_BF16(value);
            } break;
        case LM_GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

float lm_ggml_get_f32_1d(const struct lm_ggml_tensor * tensor, int i) {
    if (!lm_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        lm_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return lm_ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_F16:
            {
                return LM_GGML_CPU_FP16_TO_FP32(((lm_ggml_fp16_t *)(tensor->data))[i]);
            }
        case LM_GGML_TYPE_BF16:
            {
                return LM_GGML_BF16_TO_FP32(((lm_ggml_bf16_t *)(tensor->data))[i]);
            }
        case LM_GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

void lm_ggml_set_f32_1d(const struct lm_ggml_tensor * tensor, int i, float value) {
    if (!lm_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        lm_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        lm_ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_F16:
            {
                ((lm_ggml_fp16_t *)(tensor->data))[i] = LM_GGML_CPU_FP32_TO_FP16(value);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                ((lm_ggml_bf16_t *)(tensor->data))[i] = LM_GGML_FP32_TO_BF16(value);
            } break;
        case LM_GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

float lm_ggml_get_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case LM_GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case LM_GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case LM_GGML_TYPE_F16:
            return LM_GGML_CPU_FP16_TO_FP32(((lm_ggml_fp16_t *) data)[0]);
        case LM_GGML_TYPE_BF16:
            return LM_GGML_BF16_TO_FP32(((lm_ggml_bf16_t *) data)[0]);
        case LM_GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            LM_GGML_ABORT("fatal error");
    }
}

void lm_ggml_set_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_F16:
            {
                ((lm_ggml_fp16_t *)(data))[0] = LM_GGML_CPU_FP32_TO_FP16(value);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                ((lm_ggml_bf16_t *)(data))[0] = LM_GGML_FP32_TO_BF16(value);
            } break;
        case LM_GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// lm_ggml_compute_forward_mul_mat

static void lm_ggml_compute_forward_mul_mat_one_chunk(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst,
    const enum lm_ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = lm_ggml_is_contiguous(src1);

    lm_ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum lm_ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = lm_ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_mul_mat(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum lm_ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    lm_ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    LM_GGML_ASSERT(ne0 == ne01);
    LM_GGML_ASSERT(ne1 == ne11);
    LM_GGML_ASSERT(ne2 == ne12);
    LM_GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    LM_GGML_ASSERT(nb00 == lm_ggml_type_size(src0->type));
    LM_GGML_ASSERT(nb10 == lm_ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    LM_GGML_ASSERT(nb0 <= nb1);
    LM_GGML_ASSERT(nb1 <= nb2);
    LM_GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if LM_GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = lm_ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/lm_ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/lm_ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/lm_ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/lm_ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = lm_ggml_type_size(vec_dot_type);
        const size_t nbw1 = lm_ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = lm_ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    lm_ggml_barrier(params->threadpool);

#if LM_GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = lm_ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/lm_ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/lm_ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/lm_ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/lm_ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || lm_ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        lm_ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// lm_ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void lm_ggml_compute_forward_mul_mat_id_one_chunk(
    struct lm_ggml_tensor * dst,
    const struct lm_ggml_tensor * src0,
    const struct lm_ggml_tensor * src1,
    const struct lm_ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const enum lm_ggml_type type = src0->type;

    lm_ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum lm_ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) LM_GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void lm_ggml_compute_forward_mul_mat_id(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];
    const struct lm_ggml_tensor * ids = dst->src[2];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum lm_ggml_type type = src0->type;

    const bool src1_cont = lm_ggml_is_contiguous(src1);

    enum lm_ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    lm_ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    LM_GGML_ASSERT(nb00 == lm_ggml_type_size(type));
    LM_GGML_ASSERT(nb10 == lm_ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    LM_GGML_ASSERT(nb0 <= nb1);
    LM_GGML_ASSERT(nb1 <= nb2);
    LM_GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, lm_ggml_row_size(vec_dot_type, lm_ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    LM_GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = lm_ggml_type_size(vec_dot_type);
        const size_t nbw1 = lm_ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = lm_ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    lm_ggml_barrier(params->threadpool);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = lm_ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

#if defined(__aarch64__)
        // disable for ARM
        const bool disable_chunking = true;
#else
        // disable for NUMA
        const bool disable_chunking = lm_ggml_is_numa();
#endif // defined(__aarch64__)

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            lm_ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

static void lm_ggml_compute_forward(struct lm_ggml_compute_params * params, struct lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(params);

    if (tensor->op == LM_GGML_OP_NONE || lm_ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (lm_ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case LM_GGML_OP_DUP:
            {
                lm_ggml_compute_forward_dup(params, tensor);
            } break;
        case LM_GGML_OP_ADD:
            {
                lm_ggml_compute_forward_add(params, tensor);
            } break;
        case LM_GGML_OP_ADD1:
            {
                lm_ggml_compute_forward_add1(params, tensor);
            } break;
        case LM_GGML_OP_ACC:
            {
                lm_ggml_compute_forward_acc(params, tensor);
            } break;
        case LM_GGML_OP_SUB:
            {
                lm_ggml_compute_forward_sub(params, tensor);
            } break;
        case LM_GGML_OP_MUL:
            {
                lm_ggml_compute_forward_mul(params, tensor);
            } break;
        case LM_GGML_OP_DIV:
            {
                lm_ggml_compute_forward_div(params, tensor);
            } break;
        case LM_GGML_OP_SQR:
            {
                lm_ggml_compute_forward_sqr(params, tensor);
            } break;
        case LM_GGML_OP_SQRT:
            {
                lm_ggml_compute_forward_sqrt(params, tensor);
            } break;
        case LM_GGML_OP_LOG:
            {
                lm_ggml_compute_forward_log(params, tensor);
            } break;
        case LM_GGML_OP_SIN:
            {
                lm_ggml_compute_forward_sin(params, tensor);
            } break;
        case LM_GGML_OP_COS:
            {
                lm_ggml_compute_forward_cos(params, tensor);
            } break;
        case LM_GGML_OP_SUM:
            {
                lm_ggml_compute_forward_sum(params, tensor);
            } break;
        case LM_GGML_OP_SUM_ROWS:
            {
                lm_ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case LM_GGML_OP_MEAN:
            {
                lm_ggml_compute_forward_mean(params, tensor);
            } break;
        case LM_GGML_OP_ARGMAX:
            {
                lm_ggml_compute_forward_argmax(params, tensor);
            } break;
        case LM_GGML_OP_COUNT_EQUAL:
            {
                lm_ggml_compute_forward_count_equal(params, tensor);
            } break;
        case LM_GGML_OP_REPEAT:
            {
                lm_ggml_compute_forward_repeat(params, tensor);
            } break;
        case LM_GGML_OP_REPEAT_BACK:
            {
                lm_ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case LM_GGML_OP_CONCAT:
            {
                lm_ggml_compute_forward_concat(params, tensor);
            } break;
        case LM_GGML_OP_SILU_BACK:
            {
                lm_ggml_compute_forward_silu_back(params, tensor);
            } break;
        case LM_GGML_OP_NORM:
            {
                lm_ggml_compute_forward_norm(params, tensor);
            } break;
        case LM_GGML_OP_RMS_NORM:
            {
                lm_ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case LM_GGML_OP_RMS_NORM_BACK:
            {
                lm_ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case LM_GGML_OP_GROUP_NORM:
            {
                lm_ggml_compute_forward_group_norm(params, tensor);
            } break;
        case LM_GGML_OP_L2_NORM:
            {
                lm_ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case LM_GGML_OP_MUL_MAT:
            {
                lm_ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case LM_GGML_OP_MUL_MAT_ID:
            {
                lm_ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case LM_GGML_OP_OUT_PROD:
            {
                lm_ggml_compute_forward_out_prod(params, tensor);
            } break;
        case LM_GGML_OP_SCALE:
            {
                lm_ggml_compute_forward_scale(params, tensor);
            } break;
        case LM_GGML_OP_SET:
            {
                lm_ggml_compute_forward_set(params, tensor);
            } break;
        case LM_GGML_OP_CPY:
            {
                lm_ggml_compute_forward_cpy(params, tensor);
            } break;
        case LM_GGML_OP_CONT:
            {
                lm_ggml_compute_forward_cont(params, tensor);
            } break;
        case LM_GGML_OP_RESHAPE:
            {
                lm_ggml_compute_forward_reshape(params, tensor);
            } break;
        case LM_GGML_OP_VIEW:
            {
                lm_ggml_compute_forward_view(params, tensor);
            } break;
        case LM_GGML_OP_PERMUTE:
            {
                lm_ggml_compute_forward_permute(params, tensor);
            } break;
        case LM_GGML_OP_TRANSPOSE:
            {
                lm_ggml_compute_forward_transpose(params, tensor);
            } break;
        case LM_GGML_OP_GET_ROWS:
            {
                lm_ggml_compute_forward_get_rows(params, tensor);
            } break;
        case LM_GGML_OP_GET_ROWS_BACK:
            {
                lm_ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case LM_GGML_OP_DIAG:
            {
                lm_ggml_compute_forward_diag(params, tensor);
            } break;
        case LM_GGML_OP_DIAG_MASK_INF:
            {
                lm_ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case LM_GGML_OP_DIAG_MASK_ZERO:
            {
                lm_ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case LM_GGML_OP_SOFT_MAX:
            {
                lm_ggml_compute_forward_soft_max(params, tensor);
            } break;
        case LM_GGML_OP_SOFT_MAX_BACK:
            {
                lm_ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case LM_GGML_OP_ROPE:
            {
                lm_ggml_compute_forward_rope(params, tensor);
            } break;
        case LM_GGML_OP_ROPE_BACK:
            {
                lm_ggml_compute_forward_rope_back(params, tensor);
            } break;
        case LM_GGML_OP_CLAMP:
            {
                lm_ggml_compute_forward_clamp(params, tensor);
            } break;
        case LM_GGML_OP_CONV_TRANSPOSE_1D:
            {
                lm_ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case LM_GGML_OP_IM2COL:
            {
                lm_ggml_compute_forward_im2col(params, tensor);
            } break;
        case LM_GGML_OP_IM2COL_BACK:
            {
                lm_ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case LM_GGML_OP_CONV_2D_DW:
            {
                lm_ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case LM_GGML_OP_CONV_TRANSPOSE_2D:
            {
                lm_ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case LM_GGML_OP_POOL_1D:
            {
                lm_ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case LM_GGML_OP_POOL_2D:
            {
                lm_ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case LM_GGML_OP_POOL_2D_BACK:
            {
                lm_ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case LM_GGML_OP_UPSCALE:
            {
                lm_ggml_compute_forward_upscale(params, tensor);
            } break;
        case LM_GGML_OP_PAD:
            {
                lm_ggml_compute_forward_pad(params, tensor);
            } break;
        case LM_GGML_OP_PAD_REFLECT_1D:
            {
                lm_ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case LM_GGML_OP_ROLL:
            {
                lm_ggml_compute_forward_roll(params, tensor);
            } break;
        case LM_GGML_OP_ARANGE:
            {
                lm_ggml_compute_forward_arange(params, tensor);
            } break;
        case LM_GGML_OP_TIMESTEP_EMBEDDING:
            {
                lm_ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case LM_GGML_OP_ARGSORT:
            {
                lm_ggml_compute_forward_argsort(params, tensor);
            } break;
        case LM_GGML_OP_LEAKY_RELU:
            {
                lm_ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case LM_GGML_OP_FLASH_ATTN_EXT:
            {
                lm_ggml_compute_forward_flash_attn_ext(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3], tensor);
            } break;
        case LM_GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = lm_ggml_get_op_params_i32(tensor, 0);
                LM_GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                lm_ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case LM_GGML_OP_SSM_CONV:
            {
                lm_ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case LM_GGML_OP_SSM_SCAN:
            {
                lm_ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case LM_GGML_OP_WIN_PART:
            {
                lm_ggml_compute_forward_win_part(params, tensor);
            } break;
        case LM_GGML_OP_WIN_UNPART:
            {
                lm_ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case LM_GGML_OP_UNARY:
            {
                lm_ggml_compute_forward_unary(params, tensor);
            } break;
        case LM_GGML_OP_GET_REL_POS:
            {
                lm_ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case LM_GGML_OP_ADD_REL_POS:
            {
                lm_ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case LM_GGML_OP_RWKV_WKV6:
            {
                lm_ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case LM_GGML_OP_GATED_LINEAR_ATTN:
            {
                lm_ggml_compute_forward_gla(params, tensor);
            } break;
        case LM_GGML_OP_RWKV_WKV7:
            {
                lm_ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case LM_GGML_OP_MAP_CUSTOM1:
            {
                lm_ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case LM_GGML_OP_MAP_CUSTOM2:
            {
                lm_ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case LM_GGML_OP_MAP_CUSTOM3:
            {
                lm_ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case LM_GGML_OP_CUSTOM:
            {
                lm_ggml_compute_forward_custom(params, tensor);
            }
            break;
        case LM_GGML_OP_CROSS_ENTROPY_LOSS:
            {
                lm_ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                lm_ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case LM_GGML_OP_OPT_STEP_ADAMW:
            {
                lm_ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case LM_GGML_OP_NONE:
            {
                // nop
            } break;
        case LM_GGML_OP_COUNT:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!lm_ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case LM_GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case LM_GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case LM_GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct lm_ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!lm_ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int lm_ggml_get_n_tasks(struct lm_ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (lm_ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case LM_GGML_OP_CPY:
        case LM_GGML_OP_DUP:
        case LM_GGML_OP_CONT:
        case LM_GGML_OP_ADD:
        case LM_GGML_OP_ADD1:
        case LM_GGML_OP_ACC:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_SUB:
        case LM_GGML_OP_SQR:
        case LM_GGML_OP_SQRT:
        case LM_GGML_OP_LOG:
        case LM_GGML_OP_SIN:
        case LM_GGML_OP_COS:
        case LM_GGML_OP_SUM:
        case LM_GGML_OP_SUM_ROWS:
        case LM_GGML_OP_MEAN:
        case LM_GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_COUNT_EQUAL:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_REPEAT:
        case LM_GGML_OP_REPEAT_BACK:
        case LM_GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_UNARY:
            switch (lm_ggml_get_unary_op(node)) {
                case LM_GGML_UNARY_OP_ABS:
                case LM_GGML_UNARY_OP_SGN:
                case LM_GGML_UNARY_OP_NEG:
                case LM_GGML_UNARY_OP_STEP:
                case LM_GGML_UNARY_OP_TANH:
                case LM_GGML_UNARY_OP_ELU:
                case LM_GGML_UNARY_OP_RELU:
                case LM_GGML_UNARY_OP_SIGMOID:
                case LM_GGML_UNARY_OP_HARDSWISH:
                case LM_GGML_UNARY_OP_HARDSIGMOID:
                case LM_GGML_UNARY_OP_EXP:
                    {
                        n_tasks = 1;
                    } break;

                case LM_GGML_UNARY_OP_GELU:
                case LM_GGML_UNARY_OP_GELU_ERF:
                case LM_GGML_UNARY_OP_GELU_QUICK:
                case LM_GGML_UNARY_OP_SILU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    LM_GGML_ABORT("fatal error");
            }
            break;
        case LM_GGML_OP_SILU_BACK:
        case LM_GGML_OP_MUL:
        case LM_GGML_OP_DIV:
        case LM_GGML_OP_NORM:
        case LM_GGML_OP_RMS_NORM:
        case LM_GGML_OP_RMS_NORM_BACK:
        case LM_GGML_OP_L2_NORM:
        case LM_GGML_OP_GROUP_NORM:
        case LM_GGML_OP_CONCAT:
        case LM_GGML_OP_MUL_MAT:
        case LM_GGML_OP_MUL_MAT_ID:
        case LM_GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case LM_GGML_OP_SCALE:
        case LM_GGML_OP_SET:
        case LM_GGML_OP_RESHAPE:
        case LM_GGML_OP_VIEW:
        case LM_GGML_OP_PERMUTE:
        case LM_GGML_OP_TRANSPOSE:
        case LM_GGML_OP_GET_ROWS_BACK:
        case LM_GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_DIAG_MASK_ZERO:
        case LM_GGML_OP_DIAG_MASK_INF:
        case LM_GGML_OP_SOFT_MAX_BACK:
        case LM_GGML_OP_ROPE:
        case LM_GGML_OP_ROPE_BACK:
        case LM_GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case LM_GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, lm_ggml_nrows(node->src[0]));
            } break;
        case LM_GGML_OP_IM2COL:
        case LM_GGML_OP_IM2COL_BACK:
        case LM_GGML_OP_CONV_2D_DW:
        case LM_GGML_OP_CONV_TRANSPOSE_1D:
        case LM_GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_POOL_1D:
        case LM_GGML_OP_POOL_2D:
        case LM_GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_UPSCALE:
        case LM_GGML_OP_PAD:
        case LM_GGML_OP_PAD_REFLECT_1D:
        case LM_GGML_OP_ROLL:
        case LM_GGML_OP_ARANGE:
        case LM_GGML_OP_TIMESTEP_EMBEDDING:
        case LM_GGML_OP_ARGSORT:
        case LM_GGML_OP_FLASH_ATTN_EXT:
        case LM_GGML_OP_FLASH_ATTN_BACK:
        case LM_GGML_OP_SSM_CONV:
        case LM_GGML_OP_SSM_SCAN:
        case LM_GGML_OP_RWKV_WKV6:
        case LM_GGML_OP_GATED_LINEAR_ATTN:
        case LM_GGML_OP_RWKV_WKV7:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_WIN_PART:
        case LM_GGML_OP_WIN_UNPART:
        case LM_GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_MAP_CUSTOM1:
            {
                struct lm_ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == LM_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case LM_GGML_OP_MAP_CUSTOM2:
            {
                struct lm_ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == LM_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case LM_GGML_OP_MAP_CUSTOM3:
            {
                struct lm_ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == LM_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case LM_GGML_OP_CUSTOM:
            {
                struct lm_ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == LM_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case LM_GGML_OP_CROSS_ENTROPY_LOSS:
        case LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case LM_GGML_OP_OPT_STEP_ADAMW:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_COUNT:
            {
                LM_GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < LM_GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", lm_ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                LM_GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t lm_ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool lm_ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(LM_GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < LM_GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool lm_ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case LM_GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case LM_GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case LM_GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case LM_GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case LM_GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != LM_GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggresively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            LM_GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == LM_GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool lm_ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool lm_ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case LM_GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case LM_GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case LM_GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case LM_GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case LM_GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == LM_GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__gnu_linux__)
// TODO: this may not work on BSD, to be verified

static bool lm_ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < LM_GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            LM_GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool lm_ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case LM_GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case LM_GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case LM_GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case LM_GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case LM_GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == LM_GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool lm_ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool lm_ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool lm_ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < LM_GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void lm_ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, LM_GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, LM_GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < LM_GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= LM_GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= LM_GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void lm_ggml_threadpool_free(struct lm_ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads_max;

#ifndef LM_GGML_USE_OPENMP
    struct lm_ggml_compute_state* workers = threadpool->workers;

    lm_ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    lm_ggml_cond_broadcast(&threadpool->cond);
    lm_ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = lm_ggml_thread_join(workers[j].thrd, NULL);
        LM_GGML_ASSERT(rc == LM_GGML_EXIT_SUCCESS || rc == LM_GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    lm_ggml_mutex_destroy(&threadpool->mutex);
    lm_ggml_cond_destroy(&threadpool->cond);
#endif // LM_GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct lm_ggml_compute_state) * n_threads;
    lm_ggml_aligned_free(threadpool->workers, workers_size);
    lm_ggml_aligned_free(threadpool, sizeof(struct lm_ggml_threadpool));
}

#ifndef LM_GGML_USE_OPENMP
// pause/resume must be called under mutex
static void lm_ggml_threadpool_pause_locked(struct lm_ggml_threadpool * threadpool) {
    LM_GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    lm_ggml_cond_broadcast(&threadpool->cond);
}

static void lm_ggml_threadpool_resume_locked(struct lm_ggml_threadpool * threadpool) {
    LM_GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    lm_ggml_cond_broadcast(&threadpool->cond);
}
#endif

void lm_ggml_threadpool_pause(struct lm_ggml_threadpool * threadpool) {
#ifndef LM_GGML_USE_OPENMP
    lm_ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       lm_ggml_threadpool_pause_locked(threadpool);
    }
    lm_ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void lm_ggml_threadpool_resume(struct lm_ggml_threadpool * threadpool) {
#ifndef LM_GGML_USE_OPENMP
    lm_ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       lm_ggml_threadpool_resume_locked(threadpool);
    }
    lm_ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct lm_ggml_cplan lm_ggml_graph_plan(
          const struct lm_ggml_cgraph * cgraph,
                               int   n_threads,
            struct lm_ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //LM_GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads_max : LM_GGML_DEFAULT_N_THREADS;
    }

    size_t work_size = 0;

    struct lm_ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct lm_ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct lm_ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = lm_ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!lm_ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case LM_GGML_OP_CPY:
                case LM_GGML_OP_DUP:
                    {
                        if (lm_ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == LM_GGML_TYPE_F16  && node->src[1] && node->src[1]->type == LM_GGML_TYPE_BF16) ||
                            (node->src[0]->type == LM_GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == LM_GGML_TYPE_F16)) {
                            cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case LM_GGML_OP_ADD:
                case LM_GGML_OP_ADD1:
                    {
                        if (lm_ggml_is_quantized(node->src[0]->type)) {
                            cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case LM_GGML_OP_ACC:
                    {
                        if (lm_ggml_is_quantized(node->src[0]->type)) {
                            cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case LM_GGML_OP_COUNT_EQUAL:
                    {
                        cur = lm_ggml_type_size(node->type)*n_tasks;
                    } break;
                case LM_GGML_OP_MUL_MAT:
                    {
                        const enum lm_ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                        if (node->src[1]->type != vec_dot_type) {
                            cur = lm_ggml_row_size(vec_dot_type, lm_ggml_nelements(node->src[1]));
                        }
                    } break;
                case LM_GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct lm_ggml_tensor * src0 = node->src[0];
                        const struct lm_ggml_tensor * src1 = node->src[1];
                        const struct lm_ggml_tensor * ids = node->src[2];
                        const enum lm_ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += lm_ggml_row_size(vec_dot_type, lm_ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                    } break;
                case LM_GGML_OP_OUT_PROD:
                    {
                        if (lm_ggml_is_quantized(node->src[0]->type)) {
                            cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case LM_GGML_OP_SOFT_MAX:
                case LM_GGML_OP_ROPE:
                case LM_GGML_OP_ROPE_BACK:
                    {
                        cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case LM_GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        LM_GGML_ASSERT(node->src[0]->ne[3] == 1);
                        LM_GGML_ASSERT(node->src[1]->ne[2] == 1);
                        LM_GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == LM_GGML_TYPE_F16 ||
                             node->src[0]->type == LM_GGML_TYPE_BF16) &&
                            node->src[1]->type == LM_GGML_TYPE_F32) {
                            cur += sizeof(lm_ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(lm_ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == LM_GGML_TYPE_F32 &&
                                   node->src[1]->type == LM_GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            LM_GGML_ABORT("fatal error");
                        }
                    } break;
                case LM_GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                        cur += sizeof(lm_ggml_fp16_t)*ne00*ne01*ne02*ne03;
                        cur += sizeof(lm_ggml_fp16_t)*ne10*ne11*ne12;
                    } break;
                case LM_GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t ne10 = node->src[1]->ne[0]; // DK
                        const int64_t ne20 = node->src[2]->ne[0]; // DV

                        cur = sizeof(float)*(1*ne10 + 2*ne20)*n_tasks; // 1x head size K + 2x head size V (per thread)
                    } break;
                case LM_GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = lm_ggml_up(node->src[1]->ne[1], LM_GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in lm_ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == LM_GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == LM_GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == LM_GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case LM_GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = lm_ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case LM_GGML_OP_COUNT:
                    {
                        LM_GGML_ABORT("fatal error");
                    }
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}

static thread_ret_t lm_ggml_graph_compute_thread(void * data) {
    struct lm_ggml_compute_state * state = (struct lm_ggml_compute_state *) data;
    struct lm_ggml_threadpool    * tp    = state->threadpool;

    const struct lm_ggml_cgraph * cgraph = tp->cgraph;
    const struct lm_ggml_cplan  * cplan  = tp->cplan;

    set_numa_thread_affinity(state->ith);

    struct lm_ggml_compute_params params = {
        /*.ith       =*/ state->ith,
        /*.nth       =*/ atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ tp,
    };

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct lm_ggml_tensor * node = cgraph->nodes[node_n];

        lm_ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = LM_GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            lm_ggml_barrier(state->threadpool);
        }
    }

    lm_ggml_barrier(state->threadpool);

    return 0;
}

#ifndef LM_GGML_USE_OPENMP

// check if thread is active
static inline bool lm_ggml_graph_compute_thread_active(struct lm_ggml_compute_state * state) {
    struct lm_ggml_threadpool * threadpool = state->threadpool;
    int n_threads = atomic_load_explicit(&threadpool->n_threads_cur, memory_order_relaxed);
    return (state->ith < n_threads);
}

// check if thread is ready to proceed (exit from polling or sleeping)
static inline bool lm_ggml_graph_compute_thread_ready(struct lm_ggml_compute_state * state) {
    struct lm_ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int new_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    if (new_graph != state->last_graph) {
        state->pending    = lm_ggml_graph_compute_thread_active(state);
        state->last_graph = new_graph;
    }

    return state->pending;
}

// sync thread state after polling
static inline void lm_ggml_graph_compute_thread_sync(struct lm_ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef LM_GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool lm_ggml_graph_compute_poll_for_work(struct lm_ggml_compute_state * state) {
    struct lm_ggml_threadpool * threadpool = state->threadpool;

    // Skip polling for unused threads
    if (!lm_ggml_graph_compute_thread_active(state)) {
        return state->pending;
    }

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !lm_ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        lm_ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool lm_ggml_graph_compute_check_for_work(struct lm_ggml_compute_state * state) {
    struct lm_ggml_threadpool * threadpool = state->threadpool;

    if (lm_ggml_graph_compute_poll_for_work(state)) {
        lm_ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    lm_ggml_mutex_lock_shared(&threadpool->mutex);
    while (!lm_ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        LM_GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        lm_ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    lm_ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t lm_ggml_graph_compute_secondary_thread(void* data) {
    struct lm_ggml_compute_state * state = (struct lm_ggml_compute_state *) data;
    struct lm_ggml_threadpool * threadpool = state->threadpool;

    lm_ggml_thread_apply_priority(threadpool->prio);
    if (lm_ggml_thread_cpumask_is_valid(state->cpumask)) {
        lm_ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            LM_GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            lm_ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                lm_ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            LM_GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            lm_ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        lm_ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;

            lm_ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void lm_ggml_graph_compute_kickoff(struct lm_ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    lm_ggml_mutex_lock(&threadpool->mutex);

    LM_GGML_PRINT_DEBUG("threadpool: n_threads_cur %d n_threads %d\n", threadpool->n_threads_cur, n_threads);

    // Update the number of active threads
    atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_fetch_add_explicit(&threadpool->n_graph, 1, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       lm_ggml_thread_apply_priority(threadpool->prio);
       if (lm_ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           lm_ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       lm_ggml_threadpool_resume_locked(threadpool);
    } else {
       lm_ggml_cond_broadcast(&threadpool->cond);
    }

    lm_ggml_mutex_unlock(&threadpool->mutex);
}

#endif // LM_GGML_USE_OPENMP

static struct lm_ggml_threadpool * lm_ggml_threadpool_new_impl(
    struct lm_ggml_threadpool_params * tpp,
               struct lm_ggml_cgraph * cgraph,
                struct lm_ggml_cplan * cplan) {

    struct lm_ggml_threadpool * threadpool =
        lm_ggml_aligned_malloc(sizeof(struct lm_ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads_max    = tpp->n_threads;
        threadpool->n_threads_cur    = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = LM_GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct lm_ggml_compute_state) * tpp->n_threads;
    struct lm_ggml_compute_state * workers = lm_ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifndef LM_GGML_USE_OPENMP
    lm_ggml_mutex_init(&threadpool->mutex);
    lm_ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        lm_ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = lm_ggml_thread_create(&workers[j].thrd, NULL, lm_ggml_graph_compute_secondary_thread, &workers[j]);
        LM_GGML_ASSERT(rc == 0);
    }

    lm_ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        lm_ggml_thread_apply_priority(threadpool->prio);
        if (lm_ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            lm_ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // LM_GGML_USE_OPENMP

    return threadpool;
}

struct lm_ggml_threadpool * lm_ggml_threadpool_new(struct lm_ggml_threadpool_params * tpp) {
    return lm_ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum lm_ggml_status lm_ggml_graph_compute(struct lm_ggml_cgraph * cgraph, struct lm_ggml_cplan * cplan) {
    lm_ggml_cpu_init();

    LM_GGML_ASSERT(cplan);
    LM_GGML_ASSERT(cplan->n_threads > 0);
    LM_GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct lm_ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //LM_GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct lm_ggml_threadpool_params ttp = lm_ggml_threadpool_params_default(n_threads);
        threadpool = lm_ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = LM_GGML_STATUS_SUCCESS;
    }

#ifdef LM_GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);
            }

            lm_ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_threads_cur, 1, memory_order_relaxed);
        lm_ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads_max) {
        LM_GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads_max);
        n_threads = threadpool->n_threads_max;
    }

    // Kick all threads to start the new graph
    lm_ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    lm_ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum lm_ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        lm_ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum lm_ggml_status lm_ggml_graph_compute_with_ctx(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * cgraph, int n_threads) {
    struct lm_ggml_cplan cplan = lm_ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)lm_ggml_new_buffer(ctx, cplan.work_size);

    return lm_ggml_graph_compute(cgraph, &cplan);
}

void lm_ggml_cpu_fp32_to_fp16(const float * x, lm_ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__NNPA__)
    for (; i + 7 < n; i += 8) {
        float32x4_t v_xh = vec_xl(0, (const float *)(x + i + 0));
        float32x4_t v_xl = vec_xl(0, (const float *)(x + i + 4));
        uint16x8_t v_yd = vec_round_from_fp32(v_xh, v_xl, 0);
        uint16x8_t v_y = vec_convert_to_fp16(v_yd, 0);
        vec_xst(v_y, 0, (lm_ggml_fp16_t *)(y + i));
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t v_x = vec_xl(0, (const float *)(x + i));
        float32x4_t v_zero = vec_splats(0.0f);
        uint16x8_t v_yd = vec_round_from_fp32(v_x, v_zero, 0);
        uint16x8_t v_y = vec_convert_to_fp16(v_yd, 0);
        vec_xst(v_y, 0, (lm_ggml_fp16_t *)(y + i));
    }
#endif
    for (; i < n; ++i) {
        y[i] = LM_GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void lm_ggml_cpu_fp16_to_fp32(const lm_ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }
#elif defined(__NNPA__)
    for (; i + 7 < n; i += 8) {
        uint16x8_t v_x = vec_xl(0, (const lm_ggml_fp16_t *)(x + i));
        uint16x8_t v_yd = vec_convert_from_fp16(v_x, 0);
        float32x4_t v_yh = vec_extend_to_fp32_hi(v_yd, 0);
        float32x4_t v_yl = vec_extend_to_fp32_lo(v_yd, 0);
        vec_xst(v_yh, 0, (float *)(y + i + 0));
        vec_xst(v_yl, 0, (float *)(y + i + 4));
    }
    for (; i + 3 < n; i += 4) {
        uint16x8_t v_x = vec_xl(0, (const lm_ggml_fp16_t *)(x + i));
        uint16x8_t v_yd = vec_convert_from_fp16(v_x, 0);
        float32x4_t v_yh = vec_extend_to_fp32_hi(v_yd, 0);
        vec_xst(v_yh, 0, (float *)(y + i));
    }
#endif

    for (; i < n; ++i) {
        y[i] = LM_GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void lm_ggml_cpu_fp32_to_bf16(const float * x, lm_ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_BF16(x[i]);
    }
}

void lm_ggml_cpu_bf16_to_fp32(const lm_ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#endif
    for (; i < n; i++) {
        y[i] = LM_GGML_BF16_TO_FP32(x[i]);
    }
}

int lm_ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_llamafile(void) {
#if defined(LM_GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_nnpa(void) {
#if defined(LM_GGML_NNPA)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return lm_ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

void lm_ggml_cpu_init(void) {
    // needed to initialize lm_ggml_time
    {
        struct lm_ggml_init_params params = { 0, NULL, false };
        struct lm_ggml_context * ctx = lm_ggml_init(params);
        lm_ggml_free(ctx);
    }

    lm_ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = lm_ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    lm_ggml_fp16_t fp16;
                } u = {i};
                float f = LM_GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                lm_ggml_table_f32_f16[i] = f;
                lm_ggml_table_gelu_f16[i] = LM_GGML_CPU_FP32_TO_FP16(lm_ggml_gelu_f32(f));
                lm_ggml_table_gelu_quick_f16[i] = LM_GGML_CPU_FP32_TO_FP16(lm_ggml_gelu_quick_f32(f));
            }

            const uint64_t t_end = lm_ggml_time_us(); UNUSED(t_end);

            LM_GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef LM_GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    putenv("OMP_WAIT_POLICY=active");
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
                putenv("KMP_BLOCKTIME=200"); // 200ms
            }
#endif
        }

#if defined(__ARM_ARCH)
        lm_ggml_init_arm_arch_features();
#endif

        is_first_call = false;
    }

    lm_ggml_critical_section_end();
}
