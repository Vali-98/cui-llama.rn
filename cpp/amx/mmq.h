#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t lm_ggml_backend_amx_get_alloc_size(const struct lm_ggml_tensor * tensor);

void lm_ggml_backend_amx_convert_weight(struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);

void lm_ggml_backend_amx_mul_mat(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
