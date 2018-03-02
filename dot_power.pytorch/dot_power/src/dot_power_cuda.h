int dot_power_forward_cuda(THCudaTensor *base, THCudaTensor *exponent, THCudaTensor *output);
int dot_power_backward_cuda(THCudaTensor *base, THCudaTensor *exponent,
                            THCudaTensor *grad_output,
                            THCudaTensor *grad_base, THCudaTensor *grad_exponent);
