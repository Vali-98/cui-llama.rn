#include "ggml-backend.h"
#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

lm_ggml_backend_buffer_type_t lm_ggml_backend_amx_buffer_type(void);
bool lm_ggml_backend_amx_buft_is_amx(lm_ggml_backend_buffer_type_t buft);
bool lm_ggml_backend_amx_device_supports_op(const struct lm_ggml_tensor * op);
void lm_ggml_backend_amx_mul_mat(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
size_t lm_ggml_backend_amx_desired_wsize(const struct lm_ggml_tensor * dst);

#endif

#ifdef __cplusplus
}
#endif
