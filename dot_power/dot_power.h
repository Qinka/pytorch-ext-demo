#pragma once

#ifndef _DOT_POWER_H_
#define _DOT_POWER_H_



#ifdef __cplusplus
extern "C" {
#endif

# ifdef WIN32
#  ifdef _DOT_POWER_C_
#   define DP_C_API _declspec(dllexport)
#  else
#   define DP_C_API _declspec(dllimport)
#  endif
# else
#  define DP_C_API 
# endif

#include <stdint.h>

/**
 * dot_power_forward
 * @param base base values
 * @param exponent exponent values
 * @param output
 * @param n size
 * (all device)
 */
DP_C_API
cudaError_t dot_power_forward(const float* base, const float* exponent,
                              float* output,
                              int n, cudaStream_t stream);

/**
 * dot_power_backward
 * @param base base values
 * @param exponent exponent values
 * @param grad_output grad of output
 * @param grad_base grad of base
 * @param grad_exponent grad of exponent
 * @param n size
 */
DP_C_API
cudaError_t dot_power_backward(const float* base, const float* exponent,
                               const float* grad_output,
                               float* grad_base, float* grad_exponent,
                               int n, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif
