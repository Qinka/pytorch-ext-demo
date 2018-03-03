#include <THC/THC.h>
#include <ATen/ATen.h>
//#include <THC/THCGeneral.h>
#include <dot_power.h>
#include <stdio.h>

extern at::Context &at::globalContext();
THCState *state = at::globalContext().thc_state;

extern "C"
int dot_power_forward_cuda(THCudaTensor *base, THCudaTensor *exponent, THCudaTensor *output) {
  THCudaTensor_resizeAs(state, output, base);
  THCudaTensor_resizeAs(state, output, exponent);

  int n = THCudaTensor_nElement(state,output);

  cudaError_t rt = dot_power_forward(THCudaTensor_data(state,base),
                                      THCudaTensor_data(state,exponent),
                                      THCudaTensor_data(state,output),
                                      n,THCState_getCurrentStream(state));

  if (rt == cudaSuccess)
    return 1;
  else {
    printf("%s\n",cudaGetErrorString(rt));
    return 0;
  }

}

extern "C"
int dot_power_backward_cuda(THCudaTensor *base, THCudaTensor *exponent,
                            THCudaTensor *grad_output,
                            THCudaTensor *grad_base, THCudaTensor *grad_exponent){
  THCudaTensor_resizeAs(state, base, exponent);
  THCudaTensor_resizeAs(state, exponent, grad_output);
  THCudaTensor_resizeAs(state, grad_output, grad_base);
  THCudaTensor_resizeAs(state, grad_exponent, grad_exponent);

  int n = THCudaTensor_nElement(state,grad_output);

  cudaError_t rt = dot_power_backward(THCudaTensor_data(state,base),
                                       THCudaTensor_data(state,exponent),
                                       THCudaTensor_data(state,grad_output),
                                       THCudaTensor_data(state,grad_base),
                                       THCudaTensor_data(state,grad_exponent),
                                       n,THCState_getCurrentStream(state));

  if (rt == cudaSuccess)
    return 1;
  else {
    printf("%s\n",cudaGetErrorString(rt));
    return 0;
  }
}
